"""Public-model IK must solve its requested pose without a task environment."""

import mujoco
import numpy as np
import pytest

from agent.robot_kinematics import RobotKinematics


@pytest.mark.parametrize("target", [(0.4, 0.0, 0.12), (0.42, -0.05, 0.06), (0.46, 0.06, 0.045)])
def test_public_yam_ik_reaches_pose_with_six_bounded_joints(target):
    solver = RobotKinematics()
    joints = solver.downward_pose(*target)
    assert len(joints) == 6
    assert np.isfinite(joints).all()
    assert np.all(np.asarray(joints) >= solver.limits[:, 0])
    assert np.all(np.asarray(joints) <= solver.limits[:, 1])
    solver.data.qpos[solver.qadr] = joints
    mujoco.mj_forward(solver.model, solver.data)
    np.testing.assert_allclose(solver.data.site_xpos[solver.site], target, atol=2e-5)
    # Button pressing needs a downward tool axis; yaw is unconstrained.
    np.testing.assert_allclose(solver.data.site_xmat[solver.site].reshape(3, 3)[:, 2],
                               [0, 0, -1], atol=2e-4)
    assert solver.model.nq == 8  # Six arm coordinates and two standard fingers.
    with pytest.raises(KeyError):
        solver.model.joint("button_travel")
