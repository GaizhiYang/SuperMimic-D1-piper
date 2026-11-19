"""A policy preview follows a whole path from the actual settled TCP pose."""
from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.wf.config import TrajectoryConfig
from supermimic.wf.goal_trajectory import TrajectorySchedule
from supermimic.wf.math import following_frame, quat_apply, quat_from_rotvec, quat_mul
from supermimic.wf.preview_trajectory import (
    prepare_preview,
    rank_trajectories,
    resample_joints,
    smooth_joint_approach,
)


def test_rank_uses_whole_path_extent_exact_level_and_stable_ties():
    poses = torch.zeros(4, 5, 7)
    poses[..., 3] = 1.
    # All paths return to their start: endpoint displacement cannot rank them.
    poses[0, 2, :2] = torch.tensor([1., 1.])
    poses[1, 2, 0] = 1.5
    poses[2, 2, 1] = 1.5
    poses[3, 2, 0] = 100.  # A larger excursion at another curriculum level.
    library = SimpleNamespace(poses=poses, levels=torch.tensor([2, 2, 2, 1]))

    assert rank_trajectories(library, 2) == [1, 2, 0]
    with pytest.raises(ValueError, match="no paths at level"):
        rank_trajectories(library, 3)


@pytest.mark.parametrize("mode", [0., 1.])
def test_full_preview_schedule_starts_at_zero_and_never_switches_or_visits_padding(mode):
    steps = 24
    t = torch.linspace(0., 1., steps + 1)
    poses = torch.zeros(1, steps + 2, 7)
    poses[0, :-1, :3] = torch.stack((.3 + .2*t, .1*torch.sin(t*3.), .5 + .03*t), -1)
    poses[0, :-1, 3:] = quat_from_rotvec(torch.stack((t*0., t*.2, t*.1), -1))
    poses[0, -1] = poses[0, -2]  # The sole scheduler pad is outside the rollout.
    library = SimpleNamespace(poses=poses, levels=torch.tensor([2]), length=steps+2)
    cfg = TrajectoryConfig(library="unused", fraction=1., segment_steps=steps+1,
        update_steps=1, continuity_candidates=1, lower_level_fraction=0.)
    schedule = TrajectorySchedule(library, cfg, "body", 1, "cpu",
                                  generator=torch.Generator().manual_seed(32))
    initial = torch.zeros(1, 57)
    initial[0, :3] = torch.tensor([1., -.2, .4])
    initial[:, 3:7] = quat_from_rotvec(torch.tensor([[.1, -.2, .4]]))
    goals = torch.zeros(1, 11)
    goals[:, 7:11] = torch.tensor([[.1, .02, -.03, mode]])
    tcp = torch.cat((initial[:, :3] + quat_apply(initial[:, 3:7], poses[:, 0, :3]),
                     quat_mul(initial[:, 3:7], poses[:, 0, 3:])), -1)
    goals = schedule.start(torch.tensor([0]), initial, goals, 2, tcp)

    for step in range(steps + 1):
        current = initial.clone()
        current[:, 0] += step * .001  # Mode 0 retains its segment-start frame.
        if step:
            goals = schedule.step(current, goals, 2, tcp)
        frame = current if mode else initial
        fp, fq = following_frame(frame, "body")
        expected = torch.cat((fp + quat_apply(fq, poses[:, step, :3]),
                              quat_mul(fq, poses[:, step, 3:])), -1)
        torch.testing.assert_close(goals[:, :7], expected, atol=1e-6, rtol=1e-6)
        assert int(schedule.index[0]) == int(schedule.age[0]) == step
        assert int(schedule.traj[0]) == 0
        assert bool(schedule.active[0])
        torch.testing.assert_close(goals[:, 7:11], torch.tensor([[.1, .02, -.03, mode]]))


def test_slowing_and_lead_in_preserve_endpoints_without_joint_overshoot():
    start = np.array([.1, -.2, .3, -.4, .5, -.6])
    end = -start
    approach = smooth_joint_approach(start, end, 100)
    np.testing.assert_array_equal(approach[0], start)
    np.testing.assert_array_equal(approach[-1], end)
    assert np.all(approach >= np.minimum(start, end))
    assert np.all(approach <= np.maximum(start, end))
    # The initial/final target increments are small compared with the middle.
    increments = np.linalg.norm(np.diff(approach, axis=0), axis=-1)
    assert max(increments[0], increments[-1]) < increments[50] * .01

    joints = np.stack((start, end, start))
    slowed = resample_joints(joints, .5)
    assert len(slowed) == 5
    np.testing.assert_array_equal(slowed[[0, 2, 4]], joints)
    assert np.all(slowed >= np.minimum(start, end))
    assert np.all(slowed <= np.maximum(start, end))


@pytest.mark.parametrize("speed", [0., -1., 1.1, float("nan")])
def test_preview_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="speed"):
        resample_joints(np.zeros((3, 6)), speed)


def test_preview_rechecks_source_bytes_before_using_joint_path(tmp_path):
    source = tmp_path / "trajectories.npz"
    np.savez_compressed(source, joints=np.zeros((1, 3, 6)))
    library = SimpleNamespace(count=1, manifest=dict(
        dt=.02, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    with source.open("ab") as stream:
        stream.write(b"changed since validation")

    with pytest.raises(ValueError, match="bytes changed"):
        prepare_preview(library, tmp_path, 0, SimpleNamespace(dt=.02))


@pytest.fixture
def physical_preview_fixture(tmp_path):
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    import mujoco

    from supermimic.wf.agent_references import ArmPathPlanner
    from supermimic.wf.math import quat_apply_inverse, quat_conjugate, quat_normalize
    from supermimic.wf.simulator import BatchedSimulator

    sim = BatchedSimulator(asset_dir=os.environ["WFARM_ASSET_DIR"], arm_action_scale=None,
                           num_envs=1, friction=1., dt=.02)
    try:
        sim.settle(steps=100)
        state = sim.read_state()[0]
        spec = sim.spec
        lower = np.maximum(spec.low, spec.q0-spec.action_scale)[16:]
        upper = np.minimum(spec.high, spec.q0+spec.action_scale)[16:]
        planner = ArmPathPlanner(sim, state, lower, upper, clearance=0.)
        origin = state[23:29].copy()
        assert planner.valid(origin), "the settled fixture must be a valid arm-path origin"
        # Select a small, feasible excursion rather than depend on a stored library.
        joints = None
        for joint in range(6):
            for direction in (1., -1.):
                candidate = np.repeat(origin[None], 9, axis=0)
                candidate[:, joint] += direction*np.linspace(.005, .025, 9)
                if all(planner.valid(q) for q in candidate):
                    joints = candidate
                    break
            if joints is not None:
                break
        assert joints is not None, "fixture has no small collision-free arm excursion"
        frame_pos, frame_quat = following_frame(torch.as_tensor(state), "body")
        poses = []
        for q in joints:
            assert planner.valid(q)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, planner.data.site_xmat[spec.tcp_site])
            local = quat_apply_inverse(frame_quat,
                torch.as_tensor(planner.data.site_xpos[spec.tcp_site])-frame_pos)
            relative = quat_normalize(quat_mul(quat_conjugate(frame_quat), torch.as_tensor(quat)))
            poses.append(torch.cat((local, relative)))
        source = tmp_path / "trajectories.npz"
        np.savez_compressed(source, joints=joints[None])
        library = SimpleNamespace(count=1, length=len(joints), levels=torch.tensor([2]),
            poses=torch.stack(poses).float()[None], fingerprint="synthetic-fixture-library",
            manifest=dict(dt=sim.dt, follow_frame="body", options=dict(clearance=0.),
                          sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        yield sim, library, tmp_path
    finally:
        sim.close()


@pytest.mark.private_assets
def test_physical_preview_starts_at_actual_tcp_and_preserves_fixture(physical_preview_fixture):
    import mujoco

    sim, library, directory = physical_preview_fixture
    before = sim.snapshot()
    margin = sim.model.geom_margin.copy()
    friction = sim.model.geom_friction.copy()
    actual_quat = np.empty(4)
    mujoco.mju_mat2Quat(actual_quat, sim.data[0].site_xmat[sim.spec.tcp_site])
    actual_tcp = np.r_[sim.data[0].site_xpos[sim.spec.tcp_site], actual_quat]

    result = prepare_preview(library, directory, 0, sim, speed=.5, lead_in_s=.2, hold_s=.08)

    assert result.count == 1 and result.length == result.steps + 2
    assert result.poses.shape == (1, result.length, 7)
    assert result.lead_in_steps == 10
    assert result.steps == 10 + 16 + 4
    assert result.levels.tolist() == [2]
    assert torch.isfinite(result.poses).all()
    torch.testing.assert_close(torch.linalg.vector_norm(result.poses[..., 3:], dim=-1),
                               torch.ones(1, result.length))
    torch.testing.assert_close(result.poses[:, -1], result.poses[:, -2])
    torch.testing.assert_close(result.poses[:, result.lead_in_steps], library.poses[:, 0])
    torch.testing.assert_close(result.poses[:, -1], library.poses[:, -1])
    assert result.metadata["source_fingerprint"] == library.fingerprint
    assert result.metadata["fingerprint"] == result.fingerprint

    state = torch.as_tensor(sim.read_state(), dtype=torch.float32)
    fp, fq = following_frame(state, "body")
    first_world = torch.cat((fp+quat_apply(fq, result.poses[:, 0, :3]),
                             quat_mul(fq, result.poses[:, 0, 3:])), -1)
    np.testing.assert_allclose(first_world[0, :3].numpy(), actual_tcp[:3], atol=1e-6)
    assert abs(float(torch.dot(first_world[0, 3:], torch.as_tensor(actual_tcp[3:]).float()))) > 1.-1e-6
    after = sim.snapshot()
    for key, expected in before.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(after[key], expected, err_msg=key)
        else:
            assert after[key] == expected, key
    np.testing.assert_array_equal(sim.model.geom_margin, margin)
    np.testing.assert_array_equal(sim.model.geom_friction, friction)
