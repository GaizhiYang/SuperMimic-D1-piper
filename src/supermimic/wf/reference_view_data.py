"""Read generated reference motion without loading a policy or running physics.

Physical reference clips contain recorded whole-body states. TCP command
libraries contain arm joint paths and poses in the nominal following frame;
their playback is a kinematic preview rather than a dynamic policy rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PHYSICAL_FORMAT = "wf_physical_references_v1"
TRAJECTORY_FORMAT = "wf_tcp_trajectories_v1"
_KINDS = {PHYSICAL_FORMAT: "clips", TRAJECTORY_FORMAT: "trajectories"}


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array.setflags(write=False)
    return array


def _numeric(array: np.ndarray, name: str) -> np.ndarray:
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _quaternions(array: np.ndarray, name: str) -> None:
    if np.any(np.linalg.norm(array, axis=-1) < 1e-10):
        raise ValueError(f"{name} must have nonzero quaternion norm")


def _levels(array: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(array)
    _numeric(array, "reference levels")
    if (array.shape != (count,) or np.any(array < 0)
            or np.any(array != np.floor(array))):
        raise ValueError("reference levels must be nonnegative integers, one per sequence")
    return _readonly(array.astype(np.int64))


@dataclass(frozen=True)
class ReferenceSequence:
    index: int
    label: str
    level: int
    dt: float
    states: np.ndarray | None = None
    joints: np.ndarray | None = None
    poses: np.ndarray | None = None
    goal: np.ndarray | None = None
    failed: np.ndarray | None = None
    safe: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        for array in (self.states, self.poses, self.joints):
            if array is not None:
                return len(array)
        return 0

    @property
    def duration_s(self) -> float:
        return max(0, self.samples - 1) * self.dt


class ReferenceCollection:
    """Validated physical clips or kinematic TCP paths from one library.

    A pipeline output directory may contain both ``library/`` and
    ``trajectories/``. ``source='auto'`` selects the physical clips there;
    request ``source='trajectories'`` to inspect the moving TCP commands.
    """

    def __init__(self, directory: str | Path, source: str = "auto"):
        if source not in ("auto", "clips", "trajectories"):
            raise ValueError("source must be auto, clips, or trajectories")
        root = Path(directory).expanduser().resolve()
        candidates = [root]
        if source in ("auto", "clips"):
            candidates.append(root / "library")
        if source in ("auto", "trajectories"):
            candidates.append(root / "trajectories")
        chosen = None
        for candidate in candidates:
            path = candidate / "manifest.json"
            if not path.is_file():
                continue
            raw = path.read_bytes()
            manifest = json.loads(raw)
            kind = _KINDS.get(manifest.get("format"))
            if kind is not None and (source == "auto" or source == kind):
                chosen = candidate, raw, manifest, kind
                break
        if chosen is None:
            raise FileNotFoundError(f"no {source} reference library found under {root}")
        self.directory, raw, self.manifest, self.kind = chosen
        if self.manifest.get("complete") is not True:
            raise ValueError("reference visualization requires a complete library")
        provenance = self.manifest.get("provenance", {})
        if not isinstance(provenance, dict) or not isinstance(provenance.get("physics_hash"), str) or not provenance["physics_hash"]:
            raise ValueError("reference library requires physics provenance")
        try:
            self.dt = float(self.manifest["dt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("reference dt must be positive and finite") from exc
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("reference dt must be positive and finite")
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.raw_config = self.manifest.get("config")
        if self.raw_config is not None and not isinstance(self.raw_config, dict):
            raise ValueError("reference config must be a JSON object")
        self._arrays: dict[str, np.ndarray] | None = None
        if self.kind == "clips":
            records = self.manifest.get("clips")
            if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
                raise ValueError("physical reference library must contain clip records")
            self.count = len(records)
            self.levels = _levels(np.asarray([r.get("level", self.manifest.get("level", 0)) for r in records]), self.count)
        else:
            self._load_trajectories()
            if self.raw_config is None:
                sibling = self.directory.parent / "library" / "manifest.json"
                if sibling.is_file():
                    m = json.loads(sibling.read_bytes())
                    if (m.get("format") == PHYSICAL_FORMAT and m.get("complete") is True
                            and m.get("provenance", {}).get("physics_hash") == provenance["physics_hash"]
                            and isinstance(m.get("config"), dict)):
                        self.raw_config = m["config"]

    def _file(self, name: str) -> Path:
        if not isinstance(name, str) or not name:
            raise ValueError("reference file name must be a nonempty relative path")
        path = (self.directory / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(self.directory):
            raise ValueError("reference file must stay inside its library directory")
        if not path.is_file():
            raise FileNotFoundError(f"reference data does not exist: {path}")
        return path

    def _checked_file(self, name: str, expected: str) -> Path:
        path = self._file(name)
        if not isinstance(expected, str) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"reference checksum mismatch: {path.name}")
        return path

    def _load_trajectories(self) -> None:
        path = self._checked_file("trajectories.npz", self.manifest.get("sha256"))
        with np.load(path, allow_pickle=False) as source:
            try:
                arrays = {name: _numeric(source[name], f"trajectory {name}")
                          for name in ("poses", "joints", "levels")}
            except KeyError as exc:
                raise ValueError("trajectory library requires poses, joints, and levels") from exc
            if "seeds" in source:
                arrays["seeds"] = _numeric(source["seeds"], "trajectory seeds")
        poses, joints = arrays["poses"], arrays["joints"]
        if poses.ndim != 3 or poses.shape[0] < 1 or poses.shape[1] < 2 or poses.shape[-1] != 7:
            raise ValueError("trajectory poses must have shape (N, samples>=2, 7)")
        if joints.shape != (*poses.shape[:2], 6):
            raise ValueError("trajectory joints must have shape (N, samples, 6)")
        _quaternions(poses[..., 3:7], "trajectory poses")
        self.count = len(poses)
        self.levels = _levels(arrays["levels"], self.count)
        if "seeds" in arrays and arrays["seeds"].shape != (self.count,):
            raise ValueError("trajectory seeds must have one value per sequence")
        self._arrays = {name: _readonly(value) for name, value in arrays.items()}

    def _index(self, index: int) -> int:
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < self.count:
            raise ValueError(f"reference index must be an integer in [0, {self.count - 1}]")
        return int(index)

    def select(self, indices=None, level=None, count=3) -> list[int]:
        """Select explicit indices or a deterministic sample spanning levels."""
        if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)) or count < 1:
            raise ValueError("reference selection count must be a positive integer")
        if level is not None and (not isinstance(level, (int, np.integer)) or isinstance(level, (bool, np.bool_)) or level < 0):
            raise ValueError("reference level must be a nonnegative integer")
        if indices is not None:
            selected = [self._index(index) for index in indices]
            if not selected:
                raise ValueError("at least one reference index is required")
            if level is not None and any(self.levels[index] != level for index in selected):
                raise ValueError("explicit reference index does not belong to the requested level")
            return selected
        pool = np.flatnonzero(np.ones(self.count, dtype=bool) if level is None else self.levels == level).tolist()
        if not pool:
            raise ValueError(f"reference library contains no sequences for level {level}")
        representatives = [next(index for index in pool if self.levels[index] == value)
                           for value in sorted(set(self.levels[pool].tolist()))]
        return (representatives + [index for index in pool if index not in representatives])[:count]

    def load(self, index: int) -> ReferenceSequence:
        index = self._index(index)
        if self.kind == "trajectories":
            assert self._arrays is not None
            metadata = {"follow_frame": self.manifest.get("follow_frame", "body"),
                        "scope": self.manifest.get("scope", "kinematic TCP command trajectory")}
            if "seeds" in self._arrays:
                metadata["seed"] = int(self._arrays["seeds"][index])
            return ReferenceSequence(index=index, label=f"trajectory_{index:04d}", level=int(self.levels[index]),
                                     dt=self.dt, joints=self._arrays["joints"][index],
                                     poses=self._arrays["poses"][index], metadata=metadata)
        record = self.manifest["clips"][index]
        path = self._checked_file(record.get("file"), record.get("sha256"))
        with np.load(path, allow_pickle=False) as source:
            try:
                arrays = {name: _numeric(source[name], f"clip {name}") for name in ("states", "actions", "goal")}
            except KeyError as exc:
                raise ValueError("physical clip requires states, actions, and goal") from exc
            for name in ("failed", "safe"):
                if name in source:
                    arrays[name] = _numeric(source[name], f"clip {name}")
        states, actions, goal = arrays["states"], arrays["actions"], arrays["goal"]
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] != 57:
            raise ValueError("physical clip states must have shape (samples>=2, 57)")
        if actions.shape != (len(states) - 1, 22):
            raise ValueError("physical clip actions must have shape (samples-1, 22)")
        if goal.ndim != 1 or len(goal) < 7:
            raise ValueError("physical clip goal must contain at least xyz and quaternion wxyz")
        _quaternions(states[:, 3:7], "clip states")
        _quaternions(goal[3:7], "clip goal")
        for name in ("failed", "safe"):
            if name in arrays and (arrays[name].shape != (len(states) - 1,)
                                   or not np.isin(arrays[name], [0, 1]).all()):
                raise ValueError(f"physical clip {name} must contain one boolean per transition")
        arrays = {name: _readonly(value) for name, value in arrays.items()}
        return ReferenceSequence(index=index, label=path.stem, level=int(self.levels[index]), dt=self.dt,
                                 states=arrays["states"], goal=arrays["goal"], failed=arrays.get("failed"),
                                 safe=arrays.get("safe"), metadata=dict(record))
