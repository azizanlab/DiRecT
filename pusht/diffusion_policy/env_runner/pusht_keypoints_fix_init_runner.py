import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import pdb
import time
import wandb.sdk.data_types.video as wv
from typing import List
from copy import copy, deepcopy
from diffusion_policy.env.pusht.pusht_keypoints_sttrj_env import PushTKeypointsStTrjEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper_state import MultiStepWrapperPushTFullState
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner


logger = logging.getLogger(__name__)

class PushTKeypointsFixInitRunner(PushTKeypointsRunner):
    def __init__(self,
            output_dir,
            # constraint_key, 
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=1.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsStTrjEnv.genenerate_keypoint_manager_params()


        def env_fn():
            # return MultiStepWrapper(
            return MultiStepWrapperPushTFullState(
                VideoRecordingWrapper(
                    PushTKeypointsStTrjEnv(
                        # constraint_key=constraint_key,
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        render_action=False,
                        draw_keypoints=True,
                        render_size=192,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render, prefix:str=''):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', prefix + wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapperPushTFullState)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render, prefix:str=''):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', prefix + wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapperPushTFullState)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)


        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, 
            policy: BaseLowdimPolicy, 
            init_states: List[np.ndarray],
    ):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_init_states = len(init_states)
        n_chunks = math.ceil(n_inits / n_envs)

        ## allocate data
        all_video_paths = [[None]* n_inits ] * n_init_states
        all_rewards = np.empty((n_inits, self.max_steps, n_init_states))
        all_state_trjs = np.empty((n_inits, self.max_steps+1, 5, n_init_states))
        n_act_pred_steps = ((self.max_steps + policy.n_action_steps - 1) // policy.n_action_steps) * policy.horizon 
        all_pred_action_trjs = np.empty((n_inits, n_act_pred_steps, 2, n_init_states))
        all_roll_action_trjs = np.empty((n_inits, self.max_steps, 2, n_init_states))
        total_obs_groups = (self.max_steps + policy.n_action_steps - 1) // policy.n_action_steps
        all_obs_trjs = np.empty((n_inits, total_obs_groups, self.n_obs_steps, 20, n_init_states), dtype=np.float32)

        T_grad = policy.num_inference_steps - policy.t_stopgrad if hasattr(policy, 't_stopgrad') else 0
        all_guide_grad_trjs = np.empty((n_inits, T_grad, self.max_steps, 2, n_init_states), dtype=np.float32)

        ## Policy call timing
        policy_times = []

        ## Fill with NaN to protect missing data
        all_rewards.fill(np.nan)
        all_state_trjs.fill(np.nan)
        all_pred_action_trjs.fill(np.nan)
        all_roll_action_trjs.fill(np.nan)
        all_obs_trjs.fill(np.nan)
        all_guide_grad_trjs.fill(np.nan)

        logger.info(f"Running evaluation for {n_inits} envs, {n_init_states} init states, {n_chunks} chunks")
        for init_st_idx in tqdm.tqdm(range(n_init_states)):
            
            for chunk_idx in range(n_chunks):
                start = chunk_idx * n_envs
                end = min(n_inits, start + n_envs)
                this_global_slice = slice(start, end)
                this_n_active_envs = end - start
                this_local_slice = slice(0,this_n_active_envs)

                action_pred_trjs = np.empty((n_envs, 0, 2))
                action_roll_trjs = np.empty((n_envs, 0, 2))
                obs_roll_trjs = np.empty((n_envs, 0, self.n_obs_steps, 20), np.float32)
                guide_grad_trjs = np.empty((n_envs, T_grad, 0, 2))
                
                this_init_fns = self.env_init_fn_dills[this_global_slice]
                n_diff = n_envs - len(this_init_fns)
                if n_diff > 0:
                    this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
                assert len(this_init_fns) == n_envs

                # init envs
                env.call_each(
                    'run_dill_function', 
                    args_list=[(x,) for x in this_init_fns], 
                    kwargs_list=[{
                        'prefix': f"init_st_{init_st_idx:03d}_", 
                    }] * n_envs
                )

                # start rollout
                _ = env.reset()
                obs = np.atleast_2d(
                    env.call(
                        'set_to_state', 
                        init_states[init_st_idx],
                ))
                obs = np.stack([obs] * self.n_obs_steps, axis=1)
                
                past_action = None
                policy.reset()

                pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsFixInitRunner {chunk_idx+1}/{n_chunks}", 
                    leave=False, mininterval=self.tqdm_interval_sec)
                done = False
                while not done:
                    Do = obs.shape[-1] // 2
                    # create obs dict
                    np_obs_dict = {
                        # handle n_latency_steps by discarding the last n_latency_steps
                        'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32), 
                        'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                    }
                    if self.past_action and (past_action is not None):
                        # TODO: not tested
                        np_obs_dict['past_action'] = past_action[
                            :,-(self.n_obs_steps-1):].astype(np.float32)
                    
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, 
                        lambda x: torch.from_numpy(x).to(
                            device=device))

                    # run policy
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_policy_start = time.perf_counter()
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_policy_end = time.perf_counter()
                    policy_times.append(t_policy_end - t_policy_start)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    # handle latency_steps, we discard the first n_latency_steps actions
                    # to simulate latency
                    action = np_action_dict['action'][:,self.n_latency_steps:]

                    # step env
                    obs, reward, done, info = env.step(action)
                    done = np.all(done)
                    past_action = action

                    # record actions
                    action_pred = np_action_dict['action_pred']
                    action_pred_trjs = np.concatenate([action_pred_trjs, action_pred], axis=1)
                    ## Note: rollout action is restricted by max_steps
                    delta_t_act_roll = min(action.shape[1], self.max_steps - pbar.n)
                    action_roll_trjs = np.concatenate([action_roll_trjs, action[:,:delta_t_act_roll,...]], axis=1)
                    obs_roll_trjs = np.concatenate([obs_roll_trjs, np_obs_dict['obs'][..., None, :, :]], axis=-3)
                    # record guide gradients
                    if "guidance_grads" in np_action_dict:
                        guide_grad = np_action_dict['guidance_grads']
                        guide_grad = guide_grad[:,self.n_latency_steps:]
                        guide_grad_trjs = np.concatenate([guide_grad_trjs, guide_grad], axis=2)

                    ## update pbar
                    pbar.update(action.shape[1])
                pbar.close()

                ## collect data for this round
                # all_video_paths[this_global_slice] = env.render()[this_local_slice]
                ### Prefix the filenames, so that we can distinguish different LTLs
                ### Do it the dump way 
                this_video_paths = env.render()[this_local_slice]
                all_video_paths[init_st_idx][this_global_slice] = self.prefix_filenames(
                    file_paths = this_video_paths, 
                    # prefixes   = [f'ltl{ltl_idx:03}_'] * this_n_active_envs
                    # prefixes = [f'init_st_{init_st_idx:03d}_'] * this_n_active_envs
                    prefixes   = [''] * len(this_video_paths)
                )
                
                # all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
                reward_trjs = env.call('get_attr', 'reward')[this_local_slice]
                reward_trjs = self.pad(reward_trjs, self.max_steps)
                # reward_trjs = np.concatenate(reward_trjs, axis=-1)
                reward_trjs = np.stack(reward_trjs, axis=0)
                all_rewards[this_global_slice, :, init_st_idx] = reward_trjs

                ## evaluate this chunk's value
                state_trjs = deepcopy(env.call('get_attr', 'state_trj')[this_local_slice])
                state_trjs = self.pad(state_trjs, self.max_steps+1)
                state_trjs = np.stack(state_trjs, axis=0)   # (bs, max_steps, state_dim)
                all_state_trjs[this_global_slice, :, :, init_st_idx] = state_trjs

                ## Nod padding and keep the NaN values
                all_pred_action_trjs[this_global_slice, :action_pred_trjs.shape[1], :, init_st_idx] = action_pred_trjs
                all_roll_action_trjs[this_global_slice, :action_roll_trjs.shape[1], :, init_st_idx] = action_roll_trjs

                all_obs_trjs[this_global_slice, :obs_roll_trjs.shape[1], :, :, init_st_idx] = obs_roll_trjs
                if "guidance_grads" in np_action_dict:
                    all_guide_grad_trjs[this_global_slice, ..., init_st_idx] = guide_grad_trjs

            #end for [chunk]
        # end for [init_states]

        # Sanity Check
        assert not np.any(np.isnan(all_pred_action_trjs)), \
            f"all_pred_action_trjs contains NaN values! {np.isnan(all_pred_action_trjs).sum()} NaNs found"
        assert not np.any(np.isnan(all_roll_action_trjs)), \
            f"all_roll_action_trjs contains NaN values! {np.isnan(all_roll_action_trjs).sum()} NaNs found"
        for _ in range(n_init_states):
            for i in range(n_inits-1):
                try:
                    assert np.allclose(
                            all_state_trjs[i, 0, :, _], 
                            all_state_trjs[i+1, 0, :, _], 
                            atol=1.0,
                            rtol=1e-2,
                        ), \
                        f"Initial state {i} and {i+1} are different! \n"\
                        f"\tall_state_trjs[{i}, 0, :, {_}]: {all_state_trjs[i, 0, :, _]}\n"\
                        f"\tall_state_trjs[{i+1}, 0, :, {_}]: {all_state_trjs[i+1, 0, :, _]}" 
                except AssertionError as err:
                    logger.error(err)
                    # raise err
                    import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        mean_satisfactions = collections.defaultdict(list)
        log_data = dict()


        log_data['n_chunks'] = np.array(n_chunks)
        log_data['n_initial_states'] = np.array(n_init_states)
        log_data['n_envs'] = np.array(n_envs)
        log_data['n_inits'] = np.array(n_inits)
        log_data['all_rewards'] = all_rewards
        log_data['all_state_trjs'] = all_state_trjs
        log_data['all_pred_action_trjs'] = all_pred_action_trjs
        log_data['all_roll_action_trjs'] = all_roll_action_trjs
        log_data['all_obs_trjs'] = all_obs_trjs
        policy_times_arr = np.array(policy_times)
        log_data['policy_time_mean'] = np.array(policy_times_arr.mean())
        log_data['policy_time_std'] = np.array(policy_times_arr.std())
        log_data['policy_times'] = policy_times_arr
        logger.info(f"Policy call time: {policy_times_arr.mean():.4f} ± {policy_times_arr.std():.4f} s "
                     f"({len(policy_times_arr)} calls)")
        if "guidance_grads" in np_action_dict:
            log_data['all_guide_grad_trjs'] = all_guide_grad_trjs

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            
            rewards_per_init_st = np.max(all_rewards[i], axis=0)
            assert rewards_per_init_st.shape[0] == n_init_states, \
                f"rewards_per_ltl.shape[0]: {rewards_per_init_st.shape[0]}, not equal to n_ltls: {n_init_states}"
            avg_max_reward = np.mean(rewards_per_init_st)
            max_rewards[prefix].append(avg_max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = avg_max_reward

            ## visualize sim
            for j in range(n_init_states):
                video_path = all_video_paths[j][i]
                if video_path is not None:
                    sim_video = wandb.Video(video_path)
                    log_data[prefix+f'init_st_{j}/'+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            avg = prefix+'mean_score'
            avg_value = np.mean(value)
            log_data[avg] = avg_value
            
            std = prefix+'score_std'
            std_value = np.std(value)
            log_data[std] = std_value

            if hasattr(policy, "n_guide_steps") and hasattr(policy, "grad_scale"):
                logger.info(
                    f"stp{policy.n_guide_steps}-scl{policy.grad_scale}-t{n_inits}"\
                    f"--mScr: {avg_value:.4g} ± {std_value:.4g}"
                )
            else:
                logger.info(
                    f"t{n_inits}"\
                    f"--mScr: {avg_value:.4g} ± {std_value:.4g}"
                )

        for prefix, value in mean_satisfactions.items():
            avg = prefix+'mean_satisfaction_rate'
            avg_value = np.mean(value)
            log_data[avg] = avg_value
            
            std = prefix+'satisfaction_rate_std'
            std_value = np.std(value)
            log_data[std] = std_value

            if hasattr(policy, "n_guide_steps") and hasattr(policy, "grad_scale"):
                logger.info(
                    f"stp{policy.n_guide_steps}-scl{policy.grad_scale}-t{n_inits}"\
                    f"--mSR: {avg_value:.4g} ± {std_value:.4g}"
                )
            else:
                logger.info(
                    f"t{n_inits}"\
                    f"--mSR: {avg_value:.4g} ± {std_value:.4g}"
                )

        return log_data


    def pad(self, trjs, max_steps) -> list:
        trjs = list(trjs)
        for i in range(len(trjs)):
            max_steps = max(max_steps, len(trjs[i]))
        for i, trj in enumerate(trjs):
            l = len(trj)
            if l < max_steps:
                # print(f"l before padding: {l}")
                last = trj[-1] * np.ones_like(trj[-1])
                trjs[i] = np.concatenate([trj, [last]*(max_steps-l)], axis=0)
                # print(f"after padding: {trjs[i].shape}")
    
        return trjs



    def extract_state(self, info):
        states = []
        for i in range(len(info)):
            states.append(info[i]['state'])
        states_np = np.array(states)
        return states_np
    

    def prefix_filenames(self, file_paths:List[str], prefixes:List[str]):
        # Result list to hold the new filenames
        result = []

        # Iterate over the file_paths and prefixes together
        for file, prefix in zip(file_paths, prefixes):
            if file is None:
                result.append(None)
                continue
            # Split the path into directory and file name
            path_parts = file.rsplit('/', 1)
            # Append the prefixed filename to the path
            new_file = f'{path_parts[0]}/{prefix}{path_parts[1]}'
            # Add the new file path to the result list
            result.append(new_file)

        return result
