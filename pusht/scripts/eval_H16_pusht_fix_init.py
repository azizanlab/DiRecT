"""
Baseline evaluation sub-script for PushT with fixed initial states.
Invoked programmatically by eval_H16_seq.py via run_eval(cfg).
"""

import sys
import time
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
import numpy as np
from copy import deepcopy
from diffusion_policy.workspace.eval_pcdiff_cc_workspace import EvalPCDiffCCWorkspace
from diffusion_policy.common.env_util import rand_init_state, regroup_trajectories
from omegaconf import OmegaConf
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


logger = logging.getLogger(__name__)



def get_policy(cfg):
    ## Resolve config to get main workspace
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: EvalPCDiffCCWorkspace = cls(cfg)
    ## get policy from workspace
    policy = workspace.model

    ## load diffusion model checkpoint
    logger.info(f"Loading diffusion model from checkpoint: \'{cfg.diffusion_checkpoint}\'")
    diffusion_payload = torch.load(open(cfg.diffusion_checkpoint, 'rb'), pickle_module=dill)
    diffusion_cfg = diffusion_payload['cfg']
    train_diffusion_workspace_cls = hydra.utils.get_class(diffusion_cfg._target_)
    diffusion_workspace = train_diffusion_workspace_cls(diffusion_cfg, output_dir=cfg.output_dir)
    diffusion_workspace: EvalPCDiffCCWorkspace
    diffusion_workspace.load_payload(diffusion_payload, exclude_keys=None, include_keys=None)
    ## get diffusion model from workspace
    diffusion_policy = diffusion_workspace.model
    if diffusion_cfg.training.use_ema:
        diffusion_policy = diffusion_workspace.ema_model
    #end if
    diffusion_model = deepcopy(diffusion_policy.model)
    
    ## Replacing model with the pretrained diffusion
    policy.model = diffusion_model
    
    ## Set new policy's normalizer as the one in diffusion policy.
    policy.set_normalizer(diffusion_workspace.model.normalizer)

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    return policy


def run_eval(cfg):
    output_dir = cfg.output_dir
    log_pth = cfg.log_pth

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    #--------------------------- policy ----------------------------#
    policy = get_policy(cfg)
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    start = time.time()
    runner_log:dict = env_runner.run(
        policy=policy, 
        init_states = rand_init_state(cfg),
    )
    end = time.time()
    logger.info(f"Evaluation completed in {end - start:.6f} seconds.")

    #--------------------------- Regroup Trajectories ----------------------------#
    ## Post-process the trajectories (Regrouping)
    regrouped_trj_dict = dict()
    for key in ['all_pred_action_trjs', 'all_roll_action_trjs', 'all_state_trjs', 'all_rewards', 'all_obs_trjs']:
        if key in runner_log:
            regrouped_trj_dict[key] = regroup_trajectories(
                trajectories=runner_log[key],
                tup_size=getattr(cfg, 'group_size', 2),
                n_chunks=runner_log["n_chunks"].item(),
            )
        else: 
            print(f"Expected key {key} in runner_log, but not found.")
    # end for
    ## Update the runner_log with regrouped trajectories
    runner_log.update(regrouped_trj_dict)

    
    # dump log to json
    json_log = dict()
    raw_data_to_save = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, np.ndarray):
            raw_data_to_save[key] = value
            json_log[key] = os.path.join(output_dir, f"raw_data.npz/{key}")
        else:
            try:
                json_log[key] = value
            except RuntimeError as err:
                print(f"[WARNING] Logging error:", err)
    out_path = os.path.join(output_dir, 'eval_baseline_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    ## save data to npz
    np.savez_compressed(os.path.join(output_dir, 'raw_data.npz'), **raw_data_to_save)

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
