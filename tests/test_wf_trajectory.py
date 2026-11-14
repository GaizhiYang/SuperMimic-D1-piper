"""Ta/Tb TCP trajectory schedule and guard tolerance (no robot assets needed)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.wf.config import TrajectoryConfig
from supermimic.wf.goal_trajectory import TrajectorySchedule
from supermimic.wf.guard import PolicyGuard
from supermimic.wf.math import quat_from_rotvec

ROOT = Path(__file__).resolve().parents[1]


def library(n=3, length=40):
    # Trajectory i moves along body x from x0 = i at 1 cm per step.
    t = torch.arange(length, dtype=torch.float32)
    poses = torch.zeros(n, length, 7)
    for i in range(n):
        poses[i, :, 0] = i + .01 * t
        poses[i, :, 2] = .5
    poses[..., 3] = 1.
    return SimpleNamespace(poses=poses, levels=torch.zeros(n, dtype=torch.long), length=length)


def state(n, x=0., yaw=0.):
    s = torch.zeros(n, 57)
    s[:, 0] = x
    s[:, 3:7] = quat_from_rotvec(torch.tensor([[0., 0., yaw]])).expand(n, 4)
    return s


def goals(n, mode):
    g = torch.zeros(n, 11)
    g[:, 3] = 1.
    g[:, 10] = mode
    return g


def tcp_at(x):
    p = torch.zeros(1, 7); p[:, 0] = x; p[:, 2] = .5; p[:, 3] = 1.
    return p


def make(segment=10, update=3, candidates=8):
    cfg = TrajectoryConfig(library="unused", fraction=1., segment_steps=segment, update_steps=update,
                           continuity_candidates=candidates, lower_level_fraction=0.)
    return TrajectorySchedule(library(), cfg, "body", 1, "cpu", generator=torch.Generator().manual_seed(0))


def test_continuity_picks_window_nearest_the_tcp():
    sched = make(candidates=64)
    g = sched.start(torch.arange(1), state(1), goals(1, 0.), 0, tcp_at(1.05))
    # Trajectory 1 starts at x=1 and reaches 1.05 at index 5: nearest candidate.
    assert int(sched.traj[0]) == 1
    assert abs(float(g[0, 0]) - 1.05) < .03


def test_update_cadence_and_world_fixed_mode0():
    sched = make(segment=10, update=3)
    s = state(1)
    g = sched.start(torch.arange(1), s, goals(1, 0.), 0, tcp_at(0.))
    history = [float(g[0, 0])]
    moved = state(1, x=2., yaw=.5)  # base moves: a mode-0 target stays world-fixed
    for _ in range(9):
        g = sched.step(moved, g, 0, tcp_at(0.))
        history.append(float(g[0, 0]))
    changes = [i for i in range(1, len(history)) if history[i] != history[i-1]]
    assert changes == [3, 6, 9]
    # World-fixed: anchored at the segment-start frame, advancing 3 cm per update.
    np.testing.assert_allclose(np.diff([history[0], history[3], history[6], history[9]]), .03, atol=1e-5)


def test_mode1_targets_follow_the_current_base_frame():
    sched = make(segment=10, update=1)
    g = sched.start(torch.arange(1), state(1), goals(1, 1.), 0, tcp_at(0.))
    body = sched.poses[sched.traj[0], sched.index[0] + 1].clone()
    g = sched.step(state(1, x=2., yaw=np.pi / 2), g, 0, tcp_at(0.))
    # Rotated 90 deg about z and translated by 2 m in x.
    np.testing.assert_allclose(g[0, :3].numpy(), [2. - float(body[1]), float(body[0]), .5], atol=1e-5)


def test_new_segment_after_ta_steps():
    sched = make(segment=4, update=100)
    g = sched.start(torch.arange(1), state(1), goals(1, 0.), 0, tcp_at(0.))
    for _ in range(4):
        g = sched.step(state(1), g, 0, tcp_at(1.5))
    assert int(sched.age[0]) == 0
    assert int(sched.traj[0]) in (1, 2)  # re-chosen near x=1.5


def test_inactive_and_dead_envs_are_not_advanced():
    sched = make()
    sched.start(torch.arange(1), state(1), goals(1, 0.), 0, tcp_at(0.))
    index = int(sched.index[0])
    sched.step(state(1), goals(1, 0.), 0, tcp_at(0.), alive=torch.zeros(1, dtype=torch.bool))
    assert int(sched.index[0]) == index


def test_guard_success_tolerance_band():
    guard = PolicyGuard(return_tolerance=.1, prefer_success=True, prefer_safety=True,
                        success_key="tracking_success_rate", success_tolerance=.03)
    actor = torch.nn.Linear(1, 1); opt = torch.optim.Adam(actor.parameters())
    base = dict(budgeted_reward_mean=1., failure_rate=0., tracking_success_rate=.5, target_hash="x")
    guard.capture_baseline(base, actor, opt)
    assert guard.assess(dict(base, tracking_success_rate=.48))[0]
    assert not guard.assess(dict(base, tracking_success_rate=.46))[0]
    assert guard.assess(dict(base, failure_rate=0., budgeted_reward_mean=.95))[0]


@pytest.mark.parametrize("name", ["wf_cmd_gpu4096.json", "wf_cmd_smoke.json"])
def test_trajectory_configs(name):
    from supermimic.wf.evaluate import config_from_dict
    cfg = config_from_dict(json.loads((ROOT / "configs/examples" / name).read_text()))
    cfg.validate()
    assert cfg.trajectory.library and cfg.curriculum.metric == "tracking_success_rate"
    assert cfg.trajectory.update_steps < cfg.trajectory.segment_steps < cfg.sim.episode_steps
    assert cfg.guard.prefer_safety and cfg.guard.prefer_success


@pytest.mark.private_assets
def test_generated_library_is_reachable_and_level_ordered(tmp_path):
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf import goal_trajectory as gt
    from supermimic.wf.evaluate import config_from_dict
    raw = json.loads((ROOT / "configs/examples/wf_cmd_smoke.json").read_text())
    cfg = config_from_dict(raw)
    manifest = gt.generate(cfg, raw, tmp_path / "traj", per_level=2, levels=[0, 2], seed=1, workers=2,
                           duration_s=4., min_segment_s=1., margin=.05, clearance=.01)
    stats = manifest["stats"]
    assert stats["0"]["reach_m_p50"] < stats["2"]["reach_m_p50"]
    assert stats["0"]["max_tcp_speed_m_s"] <= gt.LEVEL_TCP_SPEED[0] + 1e-6
    with np.load(tmp_path / "traj/trajectories.npz") as d:
        assert d["poses"].shape == (4, 200, 7) and np.isfinite(d["poses"]).all()
