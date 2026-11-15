"""Dedicated broad preview commands preserve robot bounds and reachability."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.wf.preview_trajectory import generate


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kwargs", [dict(duration_s=0), dict(span_scale=-1),
    dict(tcp_speed=float("nan")), dict(angular_speed=0), dict(joint_speed=-1),
    dict(clearance=.06), dict(margin=-.01), dict(level=4)])
def test_bad_preview_parameters_fail_before_building_robot(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate(SimpleNamespace(), tmp_path / "route", **kwargs)
    assert not (tmp_path / "route").exists()


def test_preview_does_not_overwrite_existing_commands(tmp_path):
    commands = tmp_path / "commands.csv"
    commands.write_text("existing commands\n")
    with pytest.raises(FileExistsError, match="not empty"):
        generate(SimpleNamespace(), tmp_path)
    assert commands.read_text() == "existing commands\n"


@pytest.mark.private_assets
def test_broad_route_commands_match_checked_fk_and_actor_bounds(tmp_path):
    assets = os.environ.get("WFARM_ASSET_DIR")
    if not assets:
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf.agent_references import ArmPathPlanner
    from supermimic.wf.evaluate import config_from_dict
    from supermimic.wf.goal_trajectory import TrajectoryLibrary
    from supermimic.wf.math import following_frame, quat_angle, quat_error
    from supermimic.wf.simulator import BatchedSimulator
    from supermimic.wf.task import WholeBodyTask

    torch.set_num_threads(1)
    cfg = config_from_dict(json.loads((ROOT / "configs/examples/wf_cmd_smoke.json").read_text()))
    cfg.asset_dir = assets
    output = tmp_path / "wide"
    manifest = generate(cfg, output, duration_s=60.)
    assert manifest["profile"] == "preview_wide_sweep"
    assert manifest["levels"] == [3]
    stats = manifest["preview_stats"]
    np.testing.assert_allclose(stats["xyz_span_m"], [.5, .6, .5], atol=.015)
    assert stats["duration_s"] >= 60 and stats["path_length_m"] > 2.5
    assert stats["max_tcp_speed_m_s"] <= .12
    assert stats["max_tcp_angular_rad_s"] <= .5
    assert stats["max_joint_speed_rad_s"] <= .65
    with (output / "commands.csv").open() as stream:
        commands = list(csv.reader(stream))
    assert commands[0] == ["time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"]
    assert len(commands) == manifest["steps"] + 1
    assert [point["name"] for point in manifest["waypoints"]] == [
        "start", "left", "right", "high", "low", "extend", "retract", "return"]

    c = cfg.sim
    sim = BatchedSimulator(num_envs=1, seed=0, friction=1., dt=c.dt, timestep=c.timestep,
        asset_dir=assets, leg_action_scale=c.leg_action_scale, arm_action_scale=c.arm_action_scale,
        wheel_action_scale=c.wheel_action_scale, arm_initial_position=c.arm_initial_position)
    try:
        sim.reset(); sim.settle(steps=c.reset_settle_steps)
        nominal = sim.read_state()[0].copy()
        bounds = manifest["limits"]
        lower, upper = np.array(bounds["arm_lower_rad"]), np.array(bounds["arm_upper_rad"])
        planner = ArmPathPlanner(sim, nominal, lower, upper, clearance=.01)
        task = WholeBodyTask(sim.model, sim.spec, reward_cfg=cfg.reward)
        library = TrajectoryLibrary(output, task, manifest["provenance"]["physics_hash"], c.dt)
        with np.load(output / "trajectories.npz") as archive:
            joints = archive["joints"][0]
        assert (joints >= lower - 1e-6).all() and (joints <= upper + 1e-6).all()
        actions = (joints - sim.spec.q0[16:]) / sim.spec.action_scale[16:]
        assert (actions >= -1. + 1e-6).all() and (actions <= 1. - 1e-6).all()
        np.testing.assert_allclose(joints[0], joints[-1], atol=1e-7)
        np.testing.assert_allclose(library.poses[0, 0], library.poses[0, -1], atol=1e-7)
        np.testing.assert_allclose(np.asarray(commands[1:], dtype=float)[:, 1:], library.poses[0], atol=1e-6)
        fp, fq = following_frame(torch.tensor(nominal[None], dtype=torch.float64), cfg.reward.goal_follow_frame)
        matrix = np.empty(9); mujoco.mju_quat2Mat(matrix, fq[0].numpy())
        # Verify every saved point against independent FK and planning checks,
        # rather than accepting only the generator's own summary statistics.
        quaternions = []
        for index, q in enumerate(joints):
            assert planner.valid(q)
            world = planner.data.site_xpos[sim.spec.tcp_site]
            body = matrix.reshape(3, 3).T @ (world - fp[0].numpy())
            np.testing.assert_allclose(body, library.poses[0, index, :3], atol=2e-7)
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, planner.data.site_xmat[sim.spec.tcp_site])
            quaternions.append(quat)
        q = torch.tensor(np.asarray(quaternions))
        assert float(quat_angle(quat_error(q[1:], q[:-1])).max()) / c.dt <= .5
    finally:
        sim.close()
