"""
Modified from 
- https://github.com/yoraish/mmd 

MIT License

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
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import logging
from typing import Tuple, List
from enum import Enum

from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes

from mmd.common.experiments import TrialSuccessStatus
from mmd.common.conflicts import VertexConflict, Conflict
from mmd.common.constraints import MultiPointConstraint
from mmd.common.experiences import PathExperience, PathBatchExperience
from mmd.common.pretty_print import *
from mmd.common import densify_trajs, smooth_trajs, is_multi_agent_start_goal_states_valid, global_pad_paths
from mmd.config import MMDParams as params
from mmd.planners.multi_agent.cbs import SearchState  # Holding multi-agent paths and constraints information.
from mmd.planners.multi_agent.prioritized_planning import PrioritizedPlanning
from mmd.planners.single_agent.common import PlannerOutput


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class DummyPlanning(PrioritizedPlanning):
    """
    Wrapper around a low-level planner with no high-level coordination.
    """
    def __init__(self, 
        low_level_planner_l,
        start_l: List[torch.Tensor],
        goal_l: List[torch.Tensor],
        start_time_l: List[int] = None,
        reference_robot=None,
        reference_task=None,
        **kwargs
    ) -> None:
        ## Only keep the 1st low-level planner.
        if len(low_level_planner_l) > 1:
            logger.warning(
                "DummyPlanning is designed to work with a single low-level planner. "
                "Multiple planners are provided, but only the first one will be used."
            )
            low_level_planner_l = low_level_planner_l[:1]
        super().__init__(
            low_level_planner_l=low_level_planner_l,
            start_l=start_l,
            goal_l=goal_l,
            start_time_l=None,
            reference_robot=None,
            reference_task=None,
        )
        # Default to zero start times when none are provided.
        if start_time_l is None:
            start_time_l = [0] * self.num_agents
        self.start_time_l = start_time_l

        self.start_poss_tensor = torch.stack(start_l)
        self.goal_poss_tensor = torch.stack(goal_l)

    def update_start_goal_states(self, 
        start_l: List[torch.Tensor],
        goal_l: List[torch.Tensor]
    ):
        """
        Update the start and goal states of the planner.
        This is useful when the start and goal states change during planning.
        """
        self.start_poss_tensor = torch.stack(start_l)
        self.goal_poss_tensor = torch.stack(goal_l)
        ## For compatibility of all legacy methods (e.g. rendering)
        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        # Update the low-level planner with the new start and goal states.
        for planner in self.low_level_planner_l:
            planner.update_start_goal_states(self.start_poss_tensor, self.goal_poss_tensor)

    def plan(self, runtime_limit=100)->Tuple[List[torch.Tensor], int, TrialSuccessStatus, int]:
        """
        Plan a path from start to goal. Do it for one agent at a time.
        """
        success_status = TrialSuccessStatus.UNKNOWN

        root = SearchState([], [])
        planner_output:PlannerOutput = self.low_level_planner_l[0](
            self.start_poss_tensor,
            self.goal_poss_tensor,
        )

        best_path_l = [*planner_output.trajs_final]
        success_status = TrialSuccessStatus.SUCCESS

        best_path_l = global_pad_paths(best_path_l, self.start_time_l)
        return best_path_l, planner_output.t_profile, 0, success_status, len(root.conflict_l)
