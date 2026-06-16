import numpy as np


def execute_controller(
    robots: list,
    controllers: list,
    max_duration: float = 4.0,
):
    start_times = [rb.time_stamp for rb in robots]
    robot_ctrl_matches = list(zip(robots, controllers, start_times))

    for rb, ctrl, _ in robot_ctrl_matches:
        ctrl.initController(rb, max_duration)

    while (
        not np.all([ctrl.isFinished(rb) for rb, ctrl, _ in robot_ctrl_matches])
    ) and (
        np.all([rb.time_stamp - st < max_duration for rb, _, st in robot_ctrl_matches])
    ):
        for rb, ctrl, _ in robot_ctrl_matches:
            ctrl_action = ctrl.getControl(rb)
            rb.command = ctrl_action

        robots[0].nextStep()

    for rb in robots:
        rb.hold_joint_position()


def execute_controller_timesteps(
    robots: list,
    controllers: list,
    time_steps: int = 10,
):
    robot_ctrl_matches = list(zip(robots, controllers))

    for rb, ctrl in robot_ctrl_matches:
        ctrl.initController(rb, time_steps * rb.dt)

    for i in range(time_steps):
        for rb, ctrl in robot_ctrl_matches:
            ctrl_action = ctrl.getControl(rb)
            rb.command = ctrl_action

        robots[0].nextStep()

    for rb in robots:
        rb.hold_joint_position()
