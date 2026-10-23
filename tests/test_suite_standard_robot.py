"""Every task preserves the pinned standard YAM's mechanical parameters."""

import mujoco
import numpy as np
import pytest

from robo_mj_bench.robot_assets import MODEL_PATH
from robo_mj_bench.tasks.registry import TASK_NAMES, BIMANUAL_TASKS, create_simulation


@pytest.mark.parametrize("task_name", [name for name in TASK_NAMES if name not in BIMANUAL_TASKS])
def test_task_preserves_standard_robot_parameters(task_name):
    original = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    simulation = create_simulation(task_name, 23)
    model = simulation.model
    try:
        for field in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange"):
            np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
        for index in range(1, original.nbody):
            name = original.body(index).name
            body = model.body(name).id
            for field in ("body_mass", "body_inertia", "body_gravcomp", "body_pos", "body_quat"):
                np.testing.assert_array_equal(getattr(model, field)[body], getattr(original, field)[index])
        for index in range(1, 7):
            name = f"joint{index}"
            np.testing.assert_array_equal(model.joint(name).range, original.joint(name).range)
        assert len(simulation.robot_state()["joint_positions"]) == 6
    finally:
        simulation.close()
