"""Reference playback restores saved motion rather than executing a policy."""
from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest
import torch

from supermimic.wf.reference_view_data import ReferenceSequence


@pytest.fixture(scope="module")
def physical_model():
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf.robot import build_model

    return build_model()


def _quaternion(yaw, pitch=0., roll=0.):
    def axis(angle, component):
        q = np.zeros(4)
        q[0], q[component] = np.cos(angle / 2), np.sin(angle / 2)
        return q

    result = np.empty(4)
    mujoco.mju_mulQuat(result, axis(yaw, 3), axis(pitch, 2))
    mujoco.mju_mulQuat(result, result.copy(), axis(roll, 1))
    return result


def _rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _frame(state, kind):
    if kind == "body":
        return state[:3], state[3:7]
    rotation = _rotation(state[3:7])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.r_[state[:2], 0.], _quaternion(yaw)


def _state(spec):
    state = np.zeros(57, dtype=np.float64)
    state[:3] = [.123456789012, -.234567890123, .712345678901]
    state[3:7] = _quaternion(.35, -.14, .08)
    state[7:29] = spec.q0
    state[29:] = np.linspace(-.7, .8, 28)
    return state


@pytest.mark.private_assets
@pytest.mark.parametrize("mode", [0., 1.])
@pytest.mark.parametrize("follow_frame", ["body", "heading"])
def test_physical_replay_preserves_states_and_transports_following_targets(
        physical_model, mode, follow_frame):
    from supermimic.wf.reference_play import prepare_reference
    from supermimic.wf.task import WholeBodyRewardCfg, WholeBodyTask

    model, spec = physical_model
    task = WholeBodyTask(model, spec, "cpu", WholeBodyRewardCfg(
        base_command_conditioning=True, goal_follow_frame=follow_frame))
    states = np.repeat(_state(spec)[None], 3, axis=0)
    states[1, :3] += [.4, -.2, .1]
    states[2, :3] += [-.1, .3, -.05]
    states[1, 3:7] = _quaternion(1.1, .17, -.09)
    states[2, 3:7] = _quaternion(-.7, -.2, .12)
    local_position = np.array([.43, -.11, .24])
    local_quaternion = _quaternion(.2, .1)
    initial_position, initial_quaternion = _frame(states[0], follow_frame)
    target_quaternion = np.empty(4)
    mujoco.mju_mulQuat(target_quaternion, initial_quaternion, local_quaternion)
    goal = np.r_[initial_position + _rotation(initial_quaternion) @ local_position,
                 target_quaternion, .12, -.03, .04, mode]
    before_states, before_goal = states.copy(), goal.copy()
    states.setflags(write=False)
    goal.setflags(write=False)
    sequence = ReferenceSequence(0, "recorded", 0, .02, states=states, goal=goal)

    replay_states, replay_goals, _, _, route = prepare_reference(
        sequence, task, plot_frame="world")

    # Saved double-precision joint/base motion must not be rounded for rendering.
    np.testing.assert_array_equal(replay_states, before_states)
    np.testing.assert_array_equal(states, before_states)
    np.testing.assert_array_equal(goal, before_goal)
    for index, state in enumerate(states):
        if mode == 0:
            expected = goal[:7]
        else:
            position, quaternion = _frame(state, follow_frame)
            orientation = np.empty(4)
            mujoco.mju_mulQuat(orientation, quaternion, local_quaternion)
            expected = np.r_[position + _rotation(quaternion) @ local_position, orientation]
        np.testing.assert_allclose(replay_goals[index, :7].numpy(), expected, atol=2e-7)
        np.testing.assert_allclose(replay_goals[index, 7:].numpy(), goal[7:], atol=1e-8)
    np.testing.assert_allclose(route["target_positions"], replay_goals[:, :3], atol=0.)


@pytest.mark.private_assets
@pytest.mark.parametrize("follow_frame", ["body", "heading"])
def test_command_replay_reconstructs_saved_arm_joints_and_matches_mujoco_fk(
        physical_model, follow_frame):
    from supermimic.wf.reference_play import prepare_reference
    from supermimic.wf.task import WholeBodyRewardCfg, WholeBodyTask

    model, spec = physical_model
    task = WholeBodyTask(model, spec, "cpu", WholeBodyRewardCfg(
        base_command_conditioning=True, goal_follow_frame=follow_frame))
    nominal = _state(spec)
    joints = np.repeat(nominal[None, 23:29], 4, axis=0)
    joints[:, 0] += np.linspace(-.12, .15, 4)
    joints[:, 2] += np.linspace(.08, -.1, 4)
    data = mujoco.MjData(model)
    frame_position, frame_quaternion = _frame(nominal, follow_frame)
    inverse = frame_quaternion * np.array([1., -1., -1., -1.])
    poses, world_poses = [], []
    for arm in joints:
        data.qpos[:] = model.qpos0
        data.qpos[spec.base_qpos_index:spec.base_qpos_index + 7] = nominal[:7]
        data.qpos[spec.qpos_indices] = nominal[7:29]
        data.qpos[spec.qpos_indices[16:]] = arm
        mujoco.mj_kinematics(model, data)
        position = data.site_xpos[spec.tcp_site].copy()
        quaternion, relative = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[spec.tcp_site])
        mujoco.mju_mulQuat(relative, inverse, quaternion)
        poses.append(np.r_[_rotation(frame_quaternion).T @ (position - frame_position), relative])
        world_poses.append(np.r_[position, quaternion])
    poses, world_poses = np.array(poses), np.array(world_poses)
    original = [value.copy() for value in (nominal, joints, poses)]
    for value in (nominal, joints, poses):
        value.setflags(write=False)
    sequence = ReferenceSequence(0, "command", 3, .02, joints=joints, poses=poses)

    states, goals, position_error, orientation_error, route = prepare_reference(
        sequence, task, nominal=nominal, follow_frame=follow_frame, plot_frame="world")

    np.testing.assert_array_equal(states[:, 23:29], joints)
    np.testing.assert_array_equal(states[:, :23], np.broadcast_to(nominal[:23], (4, 23)))
    np.testing.assert_array_equal(states[:, 29:], 0.)
    np.testing.assert_allclose(goals[:, :3], world_poses[:, :3], atol=2e-7)
    # Quaternion signs can differ while representing the same orientation.
    dots = np.abs(np.sum(goals[:, 3:7].numpy() * world_poses[:, 3:7], axis=-1))
    np.testing.assert_allclose(dots, 1., atol=2e-7)
    assert float(position_error.max()) < 1e-6
    assert float(orientation_error.max()) < 1e-5
    np.testing.assert_allclose(route["positions"], world_poses[:, :3], atol=2e-7)
    assert "target_positions" not in route
    for value, before in zip((nominal, joints, poses), original):
        np.testing.assert_array_equal(value, before)


@pytest.mark.private_assets
@pytest.mark.parametrize("quaternion_scale", [1., 2.3])
def test_restoring_canonical_state_maps_qpos_qvel_and_never_steps(
        physical_model, monkeypatch, quaternion_scale):
    from supermimic.wf.reference_play import write_reference_state

    model, spec = physical_model
    data = mujoco.MjData(model)
    state = _state(spec)
    state[3:7] *= quaternion_scale
    state[7:29] += np.linspace(-.01, .02, 22)
    before = state.copy()
    state.setflags(write=False)
    def unexpected_step(*args, **kwargs):
        raise AssertionError("saved reference playback must not run dynamics")
    monkeypatch.setattr(mujoco, "mj_step", unexpected_step)
    monkeypatch.setattr(mujoco, "mj_forward", unexpected_step)

    write_reference_state(model, spec, data, state, 3.14)

    np.testing.assert_array_equal(data.qpos[spec.base_qpos_index:spec.base_qpos_index + 3], state[:3])
    np.testing.assert_allclose(data.qpos[spec.base_qpos_index + 3:spec.base_qpos_index + 7],
                               state[3:7] / np.linalg.norm(state[3:7]), rtol=0., atol=0.)
    np.testing.assert_array_equal(data.qpos[spec.qpos_indices], state[7:29])
    np.testing.assert_array_equal(data.qvel[spec.base_dof_index:spec.base_dof_index + 6], state[29:35])
    np.testing.assert_array_equal(data.qvel[spec.dof_indices], state[35:57])
    np.testing.assert_array_equal(state, before)
    assert data.time == 3.14
    assert np.isfinite(data.site_xpos[spec.tcp_site]).all()
