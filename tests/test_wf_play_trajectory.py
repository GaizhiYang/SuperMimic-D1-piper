"""Playback selects compatible trajectory commands without robot assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.wf.config import TrainConfig
from supermimic.wf import play_trajectory as playback


@pytest.fixture
def context(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(playback, "REPO_ROOT", repo)
    monkeypatch.chdir(repo)
    checkpoint = tmp_path / "saved" / "outputs" / "run" / "ckpt_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    cfg = TrainConfig()
    cfg.trajectory.segment_steps = 3
    cfg.trajectory.update_steps = 1
    cfg.trajectory.fraction = .2
    task = SimpleNamespace(cfg=SimpleNamespace(goal_follow_frame="body"), device=torch.device("cpu"))
    return cfg, task, checkpoint


def make_library(directory, *, marker=0., levels=(0, 1), **metadata):
    directory.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((len(levels), 8, 7), dtype=np.float32)
    poses[..., 0] = marker + np.arange(8) * .01
    poses[..., 2] = .5
    poses[..., 3] = 1.
    data = directory / "trajectories.npz"
    np.savez(data, poses=poses, levels=np.asarray(levels))
    manifest = dict(format=playback.FORMAT, complete=True,
                    provenance={"physics_hash": "physics"}, follow_frame="body", dt=.02,
                    levels=list(levels), arm_initial_position=None,
                    sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    manifest.update(metadata)
    raw = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load(context, **kwargs):
    cfg, task, checkpoint = context
    return playback.load_playback_trajectories(cfg, task, "physics", checkpoint, **kwargs)


def test_full_playback_visits_every_sample_in_order_including_last(context, tmp_path):
    from supermimic.wf.goal_trajectory import TrajectorySchedule

    directory = tmp_path / "long_route"
    make_library(directory, levels=(0,))
    library, cfg = load(context, explicit=directory)
    original = library.poses.clone()
    sequential, cfg = playback.sequential_trajectory_playback(library, cfg)
    schedule = TrajectorySchedule(sequential, cfg, "body", 1, "cpu",
                                  generator=torch.Generator().manual_seed(44))
    state = torch.zeros(1, 57)
    state[:, 3] = 1.
    goals = torch.zeros(1, 11)
    goals[:, 3] = goals[:, 10] = 1.
    tcp = goals[:, :7].clone()
    goals = schedule.start(torch.arange(1), state, goals, 0, tcp)
    actual = [goals[0, :7].clone()]
    for _ in range(library.length - 1):
        goals = schedule.step(state, goals, 0, tcp)
        actual.append(goals[0, :7].clone())
    torch.testing.assert_close(torch.stack(actual), original[0], rtol=0., atol=0.)
    torch.testing.assert_close(library.poses, original, rtol=0., atol=0.)
    assert library.length == original.shape[1]


def test_explicit_library_overrides_saved_path_and_fingerprint(context, tmp_path):
    cfg, _, _ = context
    saved, chosen = tmp_path / "saved_library", tmp_path / "chosen_library"
    fingerprint = make_library(saved, marker=1.)
    make_library(chosen, marker=2.)
    cfg.trajectory.library = str(saved)

    library, schedule = load(context, explicit=chosen, expected_fingerprint=fingerprint)

    assert float(library.poses[0, 0, 0]) == 2.
    assert schedule.library == str(chosen.resolve())
    assert schedule.fraction == 1.
    assert schedule.segment_steps == cfg.trajectory.segment_steps
    assert cfg.trajectory.library == str(saved) and cfg.trajectory.fraction == .2


def test_saved_library_takes_priority_over_newer_discovered_library(context):
    cfg, _, checkpoint = context
    root = checkpoint.parent.parent
    original, newer = root / "refs_20260101" / "trajectories", root / "refs_20260102" / "trajectories"
    fingerprint = make_library(original, marker=1.)
    make_library(newer, marker=2.)
    cfg.trajectory.library = str(original)

    library, schedule = load(context, expected_fingerprint=fingerprint)

    assert library.fingerprint == fingerprint
    assert schedule.library == str(original.resolve())


@pytest.mark.parametrize("metadata, message", [
    ({"provenance": {"physics_hash": "other"}}, "different robot physics"),
    ({"follow_frame": "heading"}, "follow frame"),
    ({"dt": .01}, "sample period"),
    ({"arm_initial_position": [0.] * 6}, "initial arm position"),
    ({"levels": [1]}, "level 0"),
    ({"complete": False}, "complete WF"),
])
def test_explicit_incompatible_library_does_not_fall_back(context, tmp_path, metadata, message):
    _, _, checkpoint = context
    make_library(checkpoint.parent.parent / "compatible" / "trajectories")
    bad = tmp_path / "incompatible"
    # levels is also a data-generation argument; changing it here tests both
    # the manifest and the trajectory records for the unsupported level.
    make_library(bad, **metadata)

    with pytest.raises(ValueError, match=message):
        load(context, explicit=bad)


def test_legacy_checkpoint_discovers_newest_compatible_complete_library(context):
    cfg, _, checkpoint = context
    root = checkpoint.parent.parent
    make_library(root / "refs_20260101" / "trajectories", marker=1.)
    newest = root / "refs_20260102" / "trajectories"
    make_library(newest, marker=2.)
    make_library(root / "refs_20260103" / "trajectories", provenance={"physics_hash": "other"})
    make_library(root / "refs_20260104" / "trajectories", complete=False)
    corrupt = root / "refs_20260105" / "trajectories"
    make_library(corrupt)
    (corrupt / "trajectories.npz").write_bytes(b"corrupted bytes")

    library, schedule = load(context)

    assert schedule.library == str(newest.resolve())
    assert float(library.poses[0, 0, 0]) == 2.
    assert cfg.trajectory.library is None


def test_legacy_checkpoint_can_discover_repo_outputs(context):
    directory = playback.REPO_ROOT / "outputs" / "refs_20260102" / "trajectories"
    make_library(directory)

    _, schedule = load(context)

    assert schedule.library == str(directory)


def test_no_compatible_library_gives_actionable_error(context):
    with pytest.raises(FileNotFoundError, match="--trajectory-library"):
        load(context)


def test_missing_explicit_path_does_not_fall_back(context):
    _, _, checkpoint = context
    make_library(checkpoint.parent.parent / "compatible" / "trajectories")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load(context, explicit="missing")


def test_missing_saved_path_without_fingerprint_does_not_guess(context):
    cfg, _, checkpoint = context
    cfg.trajectory.library = "/removed/trajectory/library"
    make_library(checkpoint.parent.parent / "compatible" / "trajectories")

    with pytest.raises(FileNotFoundError, match="checkpoint trajectory library does not exist"):
        load(context)


def test_missing_saved_path_recovers_same_fingerprint(context):
    cfg, _, checkpoint = context
    cfg.trajectory.library = "/removed/trajectory/library"
    root = checkpoint.parent.parent
    moved = root / "refs_20260101" / "trajectories"
    fingerprint = make_library(moved, marker=1.)
    make_library(root / "refs_20260102" / "trajectories", marker=2.)

    library, schedule = load(context, expected_fingerprint=fingerprint)

    assert library.fingerprint == fingerprint
    assert schedule.library == str(moved)


def test_missing_saved_path_never_substitutes_different_fingerprint(context):
    cfg, _, checkpoint = context
    cfg.trajectory.library = "/removed/trajectory/library"
    make_library(checkpoint.parent.parent / "compatible" / "trajectories")

    with pytest.raises(FileNotFoundError, match="matching the checkpoint fingerprint"):
        load(context, expected_fingerprint="missing_fingerprint")


def test_existing_changed_saved_library_does_not_fall_back(context):
    cfg, _, checkpoint = context
    root = checkpoint.parent.parent
    saved, original = root / "saved" / "trajectories", root / "original" / "trajectories"
    fingerprint = make_library(original, marker=1.)
    make_library(saved, marker=2.)
    cfg.trajectory.library = str(saved)

    with pytest.raises(ValueError, match="recorded fingerprint"):
        load(context, expected_fingerprint=fingerprint)


def test_saved_relative_path_resolves_from_checkpoint_project(context):
    cfg, _, checkpoint = context
    saved = checkpoint.parent.parent / "refs" / "trajectories"
    make_library(saved)
    cfg.trajectory.library = "outputs/refs/trajectories"

    _, schedule = load(context)

    assert schedule.library == str(saved)


def test_nondefault_arm_reset_pose_is_checked_numerically(context, tmp_path):
    cfg, _, _ = context
    cfg.sim.arm_initial_position = (0., .1, .2, .3, .4, .5)
    directory = tmp_path / "arm_library"
    make_library(directory, arm_initial_position=list(cfg.sim.arm_initial_position))
    load(context, explicit=directory)
    cfg.sim.arm_initial_position = (0., .1, .2, .3, .4, .6)

    with pytest.raises(ValueError, match="initial arm position"):
        load(context, explicit=directory)


def test_recorded_level_and_segment_length_are_validated(context, tmp_path):
    cfg, _, _ = context
    directory = tmp_path / "library"
    make_library(directory, levels=(0,))

    with pytest.raises(ValueError, match="level 1"):
        load(context, explicit=directory, level=1)
    cfg.trajectory.segment_steps = 8
    with pytest.raises(ValueError, match="segment_steps"):
        load(context, explicit=directory)


@pytest.mark.parametrize("levels, lower_fraction", [((0, 2), .3), ((2,), 0.)])
def test_sparse_lower_levels_and_requested_only_libraries_can_be_sampled(
        context, tmp_path, levels, lower_fraction):
    from supermimic.wf.goal_trajectory import TrajectorySchedule

    cfg, _, _ = context
    cfg.trajectory.lower_level_fraction = lower_fraction
    directory = tmp_path / "sparse_library"
    make_library(directory, levels=levels)
    library, options = load(context, explicit=directory, level=2)
    schedule = TrajectorySchedule(library, options, "body", 1, "cpu",
                                  generator=torch.Generator().manual_seed(7))
    state = torch.zeros(1, 57)
    state[:, 3] = 1.
    goal = torch.zeros(1, 11)
    goal[:, 3] = 1.
    tcp = torch.tensor([[0., 0., .5, 1., 0., 0., 0.]])

    sampled = schedule.start(torch.arange(1), state, goal, 2, tcp)

    assert bool(schedule.active[0])
    assert int(library.levels[schedule.traj[0]]) in levels
    assert bool(torch.isfinite(sampled).all())


def test_level_must_also_exist_in_actual_trajectory_array(context, tmp_path):
    directory = tmp_path / "wrong_level_array"
    make_library(directory, levels=(0,))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["levels"] = [0, 1]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="no valid trajectories for level 1"):
        load(context, explicit=directory, level=1)


@pytest.mark.parametrize("corruption", ["manifest_array", "provenance_array", "truncated_npz"])
def test_discovery_skips_malformed_newer_candidate(context, corruption):
    _, _, checkpoint = context
    root = checkpoint.parent.parent
    usable = root / "refs_20260101" / "trajectories"
    make_library(usable)
    broken = root / "refs_20260102" / "trajectories"
    make_library(broken)
    manifest_path = broken / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "manifest_array":
        manifest = []
    elif corruption == "provenance_array":
        manifest["provenance"] = []
    else:
        data = broken / "trajectories.npz"
        data.write_bytes(b"PK\x03\x04not a complete zip archive")
        manifest["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    _, schedule = load(context)

    assert schedule.library == str(usable)
