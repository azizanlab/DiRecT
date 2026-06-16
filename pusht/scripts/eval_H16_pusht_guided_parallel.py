"""
Guided evaluation sub-script for PushT (coupling / coupling+projection).
Invoked programmatically by eval_H16_seq.py via run_eval(cfg).
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import logging
import time
import numpy as np
from copy import deepcopy
from typing import Union
from diffusion_policy.policy.diffusion_unet_lowdim_cost_policy import DiffusionUnetLowdimCostPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_proj_coup_policy import DiffusionUnetLowdimProjCoupPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_direct_policy import DiffusionUnetLowdimDiRecTPolicy
from diffusion_policy.common.env_util import rand_init_state, regroup_trajectories
from diffusion_policy.coupling_cost.cost_functions import cost_registry
from omegaconf import OmegaConf
from datetime import datetime
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


logger = logging.getLogger(__name__)

def get_policy(cfg):
    ## Resolve config to get main workspace
    OmegaConf.resolve(cfg)

    policy:Union[DiffusionUnetLowdimCostPolicy, DiffusionUnetLowdimProjCoupPolicy, DiffusionUnetLowdimDiRecTPolicy] = hydra.utils.instantiate(cfg.policy)

    ## load diffusion model checkpoint
    logger.info(f"Loading diffusion model from checkpoint: \'{cfg.diffusion_checkpoint}\'")
    diffusion_payload = torch.load(open(cfg.diffusion_checkpoint, 'rb'), pickle_module=dill)
    diffusion_cfg = diffusion_payload['cfg']
    train_diffusion_workspace_cls = hydra.utils.get_class(diffusion_cfg._target_)
    diffusion_workspace = train_diffusion_workspace_cls(diffusion_cfg, output_dir=cfg.output_dir)
    diffusion_workspace.load_payload(diffusion_payload, exclude_keys=None, include_keys=None)
    ## get diffusion model from workspace
    diffusion_policy = diffusion_workspace.model
    if diffusion_cfg.training.use_ema:
        diffusion_policy = diffusion_workspace.ema_model
    #end if
    diffusion_model = deepcopy(diffusion_policy.model)
    
    ## Replacing diffusion model with the pretrained model
    policy.model = diffusion_model
    
    if cfg.guider == 'coupling' or 'coupling_ps':
        policy.guide.set_cost_function(cost_registry[cfg.cost_func_key],)
    else:
        raise ValueError(f"Unknown guide type: \'{cfg.guider}\'")

    ## Set new policy's normalizer as the one in diffusion policy.
    policy.set_normalizer(diffusion_workspace.model.normalizer)

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    return policy


def run_eval(cfg):
    base_output_dir = cfg.output_dir
    log_pth = cfg.log_pth
    time_str = datetime.now().strftime('%y%m%d-%H%M%S')
    output_dir = os.path.join(base_output_dir, time_str)
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)

    #--------------------------- policy ----------------------------#
    policy = get_policy(cfg)
    
    if cfg.guider not in ['coupling', 'coupling_ps']:
        raise ValueError(f"Unknown guide type: \'{cfg.guider}\'")
    if cfg.guider == 'coupling_ps':
        assert policy.pt_est , \
            "Coupling PS guide requires the policy to have `point_estimate`=True " 
    elif cfg.guider == 'coupling':
        assert not policy.pt_est, \
            "Coupling guide requires the policy to have `point_estimate`=False "
    
    if hasattr(cfg, 'projector'): 
        if cfg.projector not in ['max_vel_cvxpy', 'max_vel_admm']:
            raise ValueError(f"Unknown projector type: \'{cfg.projector}\'")
    else:
        logger.info("Projector setting no found; proceeding w/o projection.")
    

    #--------------------------- Logging Directory ----------------------------#
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    ## Dump the entire config to the output directory as a YAML file
    config_out_path = os.path.join(output_dir, 'config.yaml')
    with open(config_out_path, 'w') as f:
        OmegaConf.save(cfg, f)

    #--------------------------- Run Env Evaluation ----------------------------#
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
    )
    start = time.time()
    if cfg.guider in ['coupling', 'coupling_ps']:
        runner_log:dict = env_runner.run(
            policy=policy, 
            init_states = rand_init_state(cfg),
        )
    else:
        raise ValueError(f"Unknown guide type: \'{cfg.guider}\'")
    end = time.time()
    logger.info(f"Evaluation completed in {end - start:.6f} seconds.")

    #--------------------------- Regroup Trajectories ----------------------------#
    ## Post-process the trajectories (Regrouping)
    regrouped_trj_dict = dict()
    for key in ['all_pred_action_trjs', 'all_roll_action_trjs', 'all_state_trjs', 'all_rewards', 'all_guide_grad_trjs']:
        if key in runner_log:
            regrouped_trj_dict[key] = regroup_trajectories(
                trajectories=runner_log[key],
                tup_size=getattr(cfg, 'group_size', 2),
                n_chunks=runner_log["n_chunks"].item(),
            )
        else: 
            logger.warning(f"Expected key '{key}' in `runner_log`, but not found.")
    # end for
    ## Update the runner_log with regrouped trajectories
    runner_log.update(regrouped_trj_dict)

    # dump log to json
    json_log = dict()
    raw_data_to_save = {}

    ## Log Raw data
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video): # type: ignore
            json_log[key] = value._path
        elif isinstance(value, np.ndarray):
            raw_data_to_save[key] = value
            json_log[key] = os.path.join(output_dir, f"raw_data.npz/{key}")
        else:
            json_log[key] = value
    
    ## save data to npz
    np.savez_compressed(os.path.join(output_dir, 'raw_data.npz'), **raw_data_to_save)

    json_out_path = os.path.join(output_dir, f'eval_DP{str(cfg.guider).upper()}_log.json')
    json.dump(json_log, open(json_out_path, 'w'), indent=2, sort_keys=True)

    ## Write the output directory to a .log file
    outer_log_file = log_pth
    with open(outer_log_file, 'a' if os.path.exists(outer_log_file) else 'w') as f:
        f.write(f'{output_dir}\n')

#------------------------------------ main -----------------------------------#
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg):
    return run_eval(cfg)


if __name__ == '__main__':
    main()
