"""Reference visualization reads recorded motion without a policy or assets."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from supermimic.wf.reference_view_data import ReferenceCollection


def write_manifest(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))


def clip_arrays(samples=5):
    states = np.zeros((samples, 57))
    states[:, 0] = np.arange(samples) * .02
    states[:, 3] = 1.
    goal = np.zeros(11)
    goal[3] = goal[10] = 1.
    return dict(states=states, actions=np.zeros((samples - 1, 22)), goal=goal,
                failed=np.zeros(samples - 1, bool), safe=np.ones(samples - 1, bool))


def make_clips(directory, levels=(0, 0, 1, 2), arrays=None, **metadata):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i, level in enumerate(levels):
        path = directory / f"clip_{i:05d}.npz"
        np.savez(path, **(clip_arrays() if arrays is None else arrays))
        records.append(dict(file=path.name, level=level, family="command_drive_reach",
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(format="wf_physical_references_v1", complete=True, dt=.02,
                    provenance={"physics_hash": "physics"}, config={"marker": "physical config"}, clips=records)
    manifest.update(metadata)
    write_manifest(directory, manifest)
    return manifest


def make_trajectories(directory, levels=(0, 1, 2, 3), change=None, **metadata):
    directory.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((len(levels), 8, 7), dtype=np.float32)
    poses[..., 0] = np.arange(8) * .01
    poses[..., 3] = 1.
    arrays = dict(poses=poses, joints=np.zeros((len(levels), 8, 6)),
                  levels=np.asarray(levels), seeds=np.arange(len(levels)) + 900)
    if change:
        change(arrays)
    path = directory / "trajectories.npz"
    np.savez(path, **arrays)
    manifest = dict(format="wf_tcp_trajectories_v1", complete=True, dt=.02,
                    provenance={"physics_hash": "physics"}, follow_frame="body",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest.update(metadata)
    write_manifest(directory, manifest)
    return manifest


def test_pipeline_auto_prefers_recorded_clips_and_explicit_source_selects_commands(tmp_path):
    make_clips(tmp_path / "library")
    make_trajectories(tmp_path / "trajectories")

    recorded = ReferenceCollection(tmp_path)
    commands = ReferenceCollection(tmp_path, source="trajectories")

    assert recorded.kind == "clips" and recorded.count == 4
    assert recorded.directory == tmp_path / "library"
    assert commands.kind == "trajectories" and commands.count == 4
    assert commands.raw_config == recorded.raw_config
    assert ReferenceCollection(tmp_path / "trajectories").kind == "trajectories"
    assert len(recorded.fingerprint) == 64


def test_command_config_is_inherited_only_from_matching_physics(tmp_path):
    make_clips(tmp_path / "library", provenance={"physics_hash": "other"})
    make_trajectories(tmp_path / "trajectories")
    assert ReferenceCollection(tmp_path, source="trajectories").raw_config is None


def test_selection_spans_levels_is_repeatable_and_explicit_indices_are_not_truncated(tmp_path):
    make_clips(tmp_path)
    collection = ReferenceCollection(tmp_path)

    assert collection.select() == [0, 2, 3]
    assert collection.select(count=5) == [0, 2, 3, 1]
    assert collection.select(level=0) == [0, 1]
    assert collection.select(indices=[3, 2, 1, 0], count=1) == [3, 2, 1, 0]
    with pytest.raises(ValueError, match="requested level"):
        collection.select(indices=[0], level=2)
    with pytest.raises(ValueError, match="no sequences"):
        collection.select(level=5)
    for index in [-1, 4, .2, True]:
        with pytest.raises(ValueError, match="reference index"):
            collection.load(index)


def test_clip_samples_use_recorded_initial_state_and_arrays_are_readonly(tmp_path):
    make_clips(tmp_path)
    sequence = ReferenceCollection(tmp_path).load(0)

    assert sequence.samples == 5
    assert sequence.duration_s == pytest.approx(.08)
    np.testing.assert_equal(sequence.states, clip_arrays()["states"])
    assert sequence.metadata["family"] == "command_drive_reach"
    assert sequence.failed.shape == (4,) and sequence.safe.all()
    with pytest.raises(ValueError, match="read-only"):
        sequence.states[0, 0] = 4


def test_trajectory_arrays_loaded_once_and_sequences_are_readonly(tmp_path):
    make_trajectories(tmp_path)
    collection = ReferenceCollection(tmp_path)
    first = collection.load(0)
    (tmp_path / "trajectories.npz").unlink()
    second = collection.load(1)

    assert first.samples == 8 and first.duration_s == pytest.approx(.14)
    assert first.states is None and first.goal is None
    assert first.metadata == dict(follow_frame="body", scope="kinematic TCP command trajectory", seed=900)
    assert second.level == 1 and second.joints.shape == (8, 6)
    with pytest.raises(ValueError, match="read-only"):
        first.poses[0, 0] = 8


@pytest.mark.parametrize("metadata,message", [
    ({"complete": False}, "complete library"),
    ({"dt": 0}, "positive and finite"),
    ({"dt": float("inf")}, "positive and finite"),
    ({"provenance": {}}, "physics provenance"),
])
def test_library_metadata_validation(tmp_path, metadata, message):
    make_clips(tmp_path, **metadata)
    with pytest.raises(ValueError, match=message):
        ReferenceCollection(tmp_path)


def test_only_selected_clip_bytes_are_required_and_verified(tmp_path):
    make_clips(tmp_path)
    collection = ReferenceCollection(tmp_path)
    (tmp_path / "clip_00001.npz").write_bytes(b"broken")

    assert collection.load(0).samples == 5
    with pytest.raises(ValueError, match="checksum mismatch"):
        collection.load(1)


@pytest.mark.parametrize("name", ["../outside.npz", "/tmp/outside.npz"])
def test_clip_paths_cannot_escape_the_library(tmp_path, name):
    manifest = make_clips(tmp_path)
    manifest["clips"][0]["file"] = name
    write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="inside its library"):
        ReferenceCollection(tmp_path).load(0)


def test_clip_symlink_cannot_escape_the_library(tmp_path):
    outside = tmp_path / "outside.npz"
    np.savez(outside, **clip_arrays())
    directory = tmp_path / "library"
    manifest = make_clips(directory)
    linked = directory / "escaped.npz"
    linked.symlink_to(outside)
    manifest["clips"][0]["file"] = linked.name
    write_manifest(directory, manifest)

    with pytest.raises(ValueError, match="inside its library"):
        ReferenceCollection(directory).load(0)


@pytest.mark.parametrize("change,message", [
    (lambda a: a.update(states=np.zeros((5, 56))), "states must have shape"),
    (lambda a: a.update(actions=np.zeros((5, 22))), "actions must have shape"),
    (lambda a: a["states"].__setitem__((0, 3), 0.), "nonzero quaternion"),
    (lambda a: a["states"].__setitem__((0, 0), np.nan), "finite numeric"),
    (lambda a: a.update(safe=np.ones(5, bool)), "one boolean per transition"),
])
def test_invalid_recorded_motion_rejected(tmp_path, change, message):
    arrays = clip_arrays()
    change(arrays)
    make_clips(tmp_path, arrays=arrays)

    with pytest.raises(ValueError, match=message):
        ReferenceCollection(tmp_path).load(0)


@pytest.mark.parametrize("change,message", [
    (lambda a: a.update(joints=np.zeros((4, 7, 6))), "joints must have shape"),
    (lambda a: a["poses"].__setitem__((0, 0, 3), 0.), "nonzero quaternion"),
    (lambda a: a["joints"].__setitem__((0, 0, 0), np.inf), "finite numeric"),
    (lambda a: a.update(levels=np.array([0, 1, 2, 3.1])), "nonnegative integers"),
    (lambda a: a.update(levels=np.array([0, 1, 2])), "one per sequence"),
])
def test_invalid_command_motion_rejected(tmp_path, change, message):
    make_trajectories(tmp_path, change=change)
    with pytest.raises(ValueError, match=message):
        ReferenceCollection(tmp_path)


def test_command_checksum_verified_before_loading_arrays(tmp_path):
    make_trajectories(tmp_path)
    (tmp_path / "trajectories.npz").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        ReferenceCollection(tmp_path)
