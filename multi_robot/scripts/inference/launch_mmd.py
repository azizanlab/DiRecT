"""
Modified from 
- https://github.com/yoraish/mmd 

MIT License

Copyright (c) 2024 Yorai Shaoul
Copyright (c) 2025 Hao Luan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.

# Project includes.
from torch import fix
from mmd.common.experiments.experiment_utils import *
from inference_mmd import run_multi_agent_trial


def run_multi_agent_experiment(experiment_config: MultiAgentPlanningExperimentConfig):
    # Run the multi-agent planning experiment.
    startt = time.time()
    # Create the experiment config.
    experiment_config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Get single trial configs from the experiment config.
    single_trial_configs = experiment_config.get_single_trial_configs_from_experiment_config()
    # So let's run sequentially.
    for single_trial_config in single_trial_configs:
        print(single_trial_config)
        try:
            run_multi_agent_trial(single_trial_config)

            # Aggregate and save results after each trial.
            combine_and_save_results_for_experiment(experiment_config)
        except Exception as e:
            print("Error in run_multi_agent_experiment: ", e)
            # Save the error details for this trial.
            with open(f"error_{experiment_config.time_str}.txt", "a") as f:
                f.write(str(e))
                f.write("This is for single_trial_config: ")
                f.write(str(single_trial_config))
                f.write("\n")
            continue

    # Print the runtime.
    print("Runtime: ", time.time() - startt)
    print("Run: OK.")


if __name__ == "__main__":
    experiment_config = MultiAgentPlanningExperimentConfig()
    experiment_config.num_agents_l = [4]
    experiment_config.instance_name = "EnvConveyor2DRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvHighways2DRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvDropRegion2DRobotPlanarDiskRandom"

    experiment_config.stagger_start_time_dt = 0
    experiment_config.multi_agent_planner_class_l = ["CBS"]
    experiment_config.single_agent_planner_class = "MPD"
    experiment_config.runtime_limit = 60 
    experiment_config.num_trials_per_combination = 100
    # experiment_config.num_trials_per_combination = 5
    experiment_config.render_animation = False
    run_multi_agent_experiment(experiment_config)
