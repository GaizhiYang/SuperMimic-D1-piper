"""Resolve a compatible TCP trajectory library for policy video playback."""
from __future__ import annotations

from dataclasses import replace
from copy import copy
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch

from .goal_trajectory import FORMAT, TrajectoryLibrary


REPO_ROOT = Path(__file__).resolve().parents[3]


def sequential_trajectory_playback(library, cfg):
    """Play a complete library route from sample zero instead of short windows.

    A window as long as the route leaves only one possible starting sample.
    Pad with one held sample so the final real target is reached before the
    schedule draws its next window. The source library stays unchanged.
    """
    if library.length < 2:
        raise ValueError("full trajectory playback requires at least two samples")
    playback = copy(library)
    playback.poses = torch.cat((library.poses, library.poses[:, -1:]), dim=1)
    playback.length += 1
    return playback, replace(cfg, fraction=1., segment_steps=library.length,
                             update_steps=1, lower_level_fraction=0.)


def _output_roots(checkpoint_path):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    # Pipeline checkpoints live at OUTPUTS/RUN/ckpt_latest.pt. Also support a
    # copied run in an arbitrarily named output folder without recursive scans.
    local = next((p for p in checkpoint.parents if p.name == "outputs"),
                 checkpoint.parent.parent)
    return tuple(dict.fromkeys((local, REPO_ROOT / "outputs")))


def _resolve_library(directory, checkpoint_path):
    path = Path(directory).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path]
    candidates += [root.parent / path for root in _output_roots(checkpoint_path)]
    candidates += [REPO_ROOT / path, Path(checkpoint_path).resolve().parent / path]
    return next((p.resolve() for p in candidates if p.exists()), candidates[0].resolve())


def _validate_metadata(manifest, cfg, task, physics_hash, level):
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or not manifest.get("complete")):
        raise ValueError("expected a complete WF TCP trajectory library")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("physics_hash") != physics_hash:
        raise ValueError("trajectory library was generated for different robot physics")
    if manifest.get("follow_frame") != task.cfg.goal_follow_frame:
        raise ValueError("trajectory follow frame differs from reward.goal_follow_frame")
    if not np.isclose(manifest.get("dt", float("nan")), cfg.sim.dt):
        raise ValueError("trajectory sample period differs from the control period")
    actual_arm, expected_arm = manifest.get("arm_initial_position"), cfg.sim.arm_initial_position
    if (actual_arm is None) != (expected_arm is None):
        raise ValueError("trajectory library uses a different initial arm position")
    if actual_arm is not None:
        actual_arm, expected_arm = np.asarray(actual_arm), np.asarray(expected_arm)
        if (actual_arm.shape != expected_arm.shape
                or not np.allclose(actual_arm, expected_arm, rtol=0., atol=1e-8)):
            raise ValueError("trajectory library uses a different initial arm position")
    if level not in manifest.get("levels", []):
        raise ValueError(f"trajectory library has no trajectories for level {level}")


def _load(directory, cfg, task, physics_hash, level, expected_fingerprint=None):
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    _validate_metadata(manifest, cfg, task, physics_hash, level)
    if (expected_fingerprint is not None
            and hashlib.sha256(raw).hexdigest() != expected_fingerprint):
        raise ValueError("trajectory library differs from the checkpoint's recorded fingerprint")
    library = TrajectoryLibrary(directory, task, physics_hash, cfg.sim.dt, task.device)
    if library.levels.shape != (library.count,) or not bool((library.levels == level).any()):
        raise ValueError(f"trajectory library has no valid trajectories for level {level}")
    if cfg.trajectory.segment_steps + 1 > library.length:
        raise ValueError("trajectory.segment_steps exceeds the library trajectory length")
    schedule = replace(cfg.trajectory, library=str(directory), fraction=1.)
    return library, schedule


def load_playback_trajectories(cfg, task, physics_hash, checkpoint_path, *, explicit=None,
                               expected_fingerprint=None, level=0):
    """Use the requested/saved library, or discover one for a legacy checkpoint.

    A saved library that has moved is recovered only by its exact manifest
    fingerprint. Legacy checkpoints without a library use the newest compatible
    directory name under the checkpoint's output folder or this repo's outputs.
    Explicit paths override checkpoint provenance but still require compatible
    physics, following frame, sample period, arm reset pose and curriculum level.
    """
    if not 0 <= level <= 3:
        raise ValueError("level must be in [0, 3]")
    if explicit is not None:
        directory = _resolve_library(explicit, checkpoint_path)
        if not directory.exists():
            raise FileNotFoundError(f"trajectory library does not exist: {directory}")
        return _load(directory, cfg, task, physics_hash, level)

    stored = cfg.trajectory.library
    if stored:
        directory = _resolve_library(stored, checkpoint_path)
        if directory.exists():
            return _load(directory, cfg, task, physics_hash, level, expected_fingerprint)
        if expected_fingerprint is None:
            raise FileNotFoundError(
                f"checkpoint trajectory library does not exist: {directory}; "
                "specify --trajectory-library /path/to/trajectories")

    manifests = set()
    for root in _output_roots(checkpoint_path):
        manifests.update(root.glob("*/trajectories/manifest.json"))
        manifests.update(root.glob("trajectories/manifest.json"))
    # Pipeline output names carry sortable timestamps. A stable path tie break
    # makes selection independent of filesystem enumeration order.
    for manifest in sorted(manifests, key=lambda p: (p.parent.parent.name, str(p)), reverse=True):
        directory = manifest.parent.resolve()
        try:
            return _load(directory, cfg, task, physics_hash, level, expected_fingerprint)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError, BadZipFile):
            continue
    qualifier = "matching the checkpoint fingerprint" if expected_fingerprint else "compatible"
    raise FileNotFoundError(
        f"no {qualifier} TCP trajectory library found; "
        "specify --trajectory-library /path/to/trajectories")
