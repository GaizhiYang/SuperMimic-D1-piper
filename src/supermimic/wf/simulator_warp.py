"""Massively parallel MuJoCo Warp backend with the BatchedSimulator interface.

The same compiled MjModel (identical physics hash) is uploaded to the GPU with
``mujoco_warp``; every world advances in one batched call. Actuation is the
same PD/velocity servo law as the CPU backend: torques are recomputed from the
current joint state at EVERY physics substep, clipped by the same velocity and
effort limits, and written to the motor controls. Contact/safety metrics are
derived from the actual solver contacts and constraint forces.

The CPU ``BatchedSimulator`` remains the correctness/evaluation backend. This
module is a collection backend for training; it does not implement ablation
welds (use the CPU evaluator for interventions).
"""

from __future__ import annotations

import copy
from typing import Any

import mujoco
import numpy as np
import torch

from .robot import RobotSpec, build_model

try:  # Optional GPU stack: pip install -e '.[gpu]'
    import warp as wp
    import mujoco_warp as mjw
except ImportError:  # pragma: no cover - exercised only without the extra
    wp = None
    mjw = None


def warp_available() -> bool:
    return wp is not None and mjw is not None


class WarpBatchedSimulator:
    """Drop-in collection backend: numpy in/out, torch/warp on the device."""

    backend = "warp"

    def __init__(self, model: mujoco.MjModel | None = None,
                 spec: RobotSpec | None = None, num_envs: int = 1, seed: int = 0,
                 dt: float = .02, friction_range: tuple[float, float] = (.6, 1.2),
                 friction: float | None = None, timestep: float = .005,
                 asset_dir: str | None = None, leg_action_scale: float = .35,
                 arm_action_scale: float | None = .5, wheel_action_scale: float = 25.,
                 arm_initial_position=None, device: str = "cuda:0",
                 nconmax: int = 64, njmax: int = 320, cuda_graph: bool = True):
        if not warp_available():
            raise ImportError("sim.backend='warp' requires warp-lang and mujoco-warp: pip install -e '.[gpu]'")
        if spec is None:
            if model is not None:
                raise ValueError("provide spec with an externally constructed model")
            model, spec = build_model(asset_dir, timestep=timestep,
                leg_action_scale=leg_action_scale, arm_action_scale=arm_action_scale,
                wheel_action_scale=wheel_action_scale)
        elif model is None:
            model = spec.model
        if model is not spec.model:
            raise ValueError("model and RobotSpec must refer to the same model")
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        substeps = round(dt / model.opt.timestep)
        if substeps < 1 or not np.isclose(substeps * model.opt.timestep, dt, atol=1e-10):
            raise ValueError("dt must be an integer multiple of model.opt.timestep")
        if friction is not None:
            friction_range = (friction, friction)
        if not (0 < friction_range[0] <= friction_range[1]):
            raise ValueError("friction_range must be positive and ordered")
        self.model, self.spec = model, spec
        self.num_envs, self.dt, self.substeps = int(num_envs), float(dt), substeps
        self.friction_range = tuple(float(v) for v in friction_range)
        self.rng = np.random.default_rng(seed)
        self.ablation = "none"
        self.initial_q = spec.q0.copy()
        if arm_initial_position is not None:
            pose = np.asarray(arm_initial_position, dtype=float)
            if pose.shape != (6,) or not np.isfinite(pose).all() or np.any(pose < spec.low[16:]) or np.any(pose > spec.high[16:]):
                raise ValueError('initial arm pose violates source joint bounds')
            self.initial_q[16:] = pose
        initial = (self.initial_q - spec.q0) / spec.action_scale
        initial[spec.wheel_indices] = 0.
        if np.abs(initial).max() > 1.000001:
            raise ValueError('initial arm pose cannot be commanded with this action scale; use the full source range')
        self.initial_action = np.broadcast_to(initial, (self.num_envs, 22)).copy()
        self.ablation_action = self.initial_action.copy()

        self.wp_device = wp.get_device(device)
        self.device = torch.device("cuda", self.wp_device.ordinal) if self.wp_device.is_cuda else torch.device("cpu")
        n = self.num_envs
        with wp.ScopedDevice(self.wp_device):
            self.mjw_model = mjw.put_model(model, batch_sizes={"geom_friction": n})
            self.mjw_data = mjw.make_data(model, nworld=n, nconmax=int(nconmax), njmax=int(njmax))
            self._warp_stream = wp.Stream(self.wp_device) if self.wp_device.is_cuda else None
            if self.wp_device.is_cuda:
                wp.synchronize_device(self.wp_device)
        self.naconmax = int(self.mjw_data.naconmax)
        self.njmax = int(self.mjw_data.njmax)
        t = lambda a: wp.to_torch(a)
        d, m = self.mjw_data, self.mjw_model
        # Zero-copy device views. Warp writes are visible to torch and back.
        self._qpos, self._qvel, self._ctrl = t(d.qpos), t(d.qvel), t(d.ctrl)
        self._qacc_warmstart, self._time = t(d.qacc_warmstart), t(d.time)
        self._act = t(d.act) if model.na else None
        self._xpos, self._xmat = t(d.xpos), t(d.xmat)
        self._cvel, self._subtree_com = t(d.cvel), t(d.subtree_com)
        self._friction = t(m.geom_friction)
        self._nacon = t(d.nacon)
        c = d.contact
        self._c_dist, self._c_pos, self._c_geom = t(c.dist), t(c.pos), t(c.geom)
        self._c_dim, self._c_world, self._c_efc = t(c.dim), t(c.worldid), t(c.efc_address)
        self._efc_force = t(d.efc.force)
        if self._friction.shape[:2] != (n, model.ngeom):
            raise RuntimeError("mujoco_warp did not batch geom_friction per world")

        dev, f32 = self.device, torch.float32
        tt = lambda x, dtype=f32: torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)
        s = spec
        self._qadr = tt(s.qpos_indices, torch.long)
        self._dadr = tt(s.dof_indices, torch.long)
        self._act_ids = tt(s.actuator_ids, torch.long)
        self._bq, self._bv = int(s.base_qpos_index), int(s.base_dof_index)
        self._kp, self._kd = tt(s.kp), tt(s.kd)
        self._effort, self._velocity = tt(s.effort), tt(s.velocity)
        self._q0, self._scale = tt(s.q0), tt(s.action_scale)
        self._low, self._high = tt(s.low), tt(s.high)
        self._wheel = tt(s.wheel_indices, torch.long)
        position = np.ones(22, dtype=bool); position[s.wheel_indices] = False
        self._position = tt(position, torch.bool)
        self._initial_q = tt(self.initial_q)
        self._qpos0 = tt(model.qpos0)
        lookup = np.full(model.ngeom, -1, dtype=np.int64)
        lookup[np.asarray(s.wheel_geom_ids)] = np.arange(4)
        self._wheel_lookup = tt(lookup, torch.long)
        self._wheel_body = tt(s.wheel_body_ids, torch.long)
        self._wheel_root = tt(model.body_rootid[np.asarray(s.wheel_body_ids)], torch.long)
        self._ground = int(s.ground_geom)
        self._base_body = int(s.base_body)

        self.xi = np.ones((n, 1), dtype=np.float64)
        self.applied_action = np.zeros((n, 22), dtype=np.float64)
        self.torque = np.zeros((n, 22), dtype=np.float64)
        self.torque_peak = np.zeros_like(self.torque)
        self._metrics: dict[str, np.ndarray] = {}
        self._overflow_warned = False
        self._graphs: dict[str, Any] = {}
        self._use_graph = bool(cuda_graph) and self.wp_device.is_cuda
        self.reset()

    # ------------------------------------------------------------------ warp
    def _launch(self, name: str) -> None:
        fn = mjw.step if name == "step" else mjw.forward
        if not self.wp_device.is_cuda:
            with wp.ScopedDevice(self.wp_device):
                fn(self.mjw_model, self.mjw_data)
            return
        # Warp runs on its own (capturable) stream. Events order it after the
        # torch writes that precede it, and torch reads after it completes.
        # torch's default stream is the legacy NULL stream, which cannot be
        # captured into a CUDA graph, hence the separate Warp stream.
        torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        warp_stream = self._warp_stream
        warp_stream.wait_stream(torch_stream)
        with wp.ScopedDevice(self.wp_device), wp.ScopedStream(warp_stream, sync_enter=False):
            graph = self._graphs.get(name)
            if graph is not None:
                wp.capture_launch(graph)
            else:
                fn(self.mjw_model, self.mjw_data)  # eager call also loads every kernel
                if self._use_graph and name not in self._graphs:
                    try:
                        with wp.ScopedCapture(stream=warp_stream) as capture:
                            fn(self.mjw_model, self.mjw_data)
                        # Capture only records the launches; state is unchanged.
                        self._graphs[name] = capture.graph
                    except Exception as error:  # pragma: no cover - device dependent
                        import warnings
                        warnings.warn(f"CUDA graph capture of mujoco_warp.{name} failed ({error!r}); running eagerly")
                        self._graphs[name] = None
        torch_stream.wait_stream(warp_stream)

    def _forward(self) -> None:
        self._launch("forward")

    def _step_physics(self) -> None:
        self._launch("step")

    # --------------------------------------------------------------- helpers
    def _ids(self, ids: Any = None) -> np.ndarray:
        result = np.arange(self.num_envs) if ids is None else np.asarray(ids, dtype=int).reshape(-1)
        if np.any(result < 0) or np.any(result >= self.num_envs):
            raise IndexError("environment index outside batch")
        return result

    def _tid(self, ids: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _apply_friction(self, ids: np.ndarray) -> None:
        # Same coefficient on all geoms, as in the CPU backend.
        values = torch.as_tensor(self.xi[ids, 0], dtype=self._friction.dtype, device=self.device)
        self._friction[self._tid(ids), :, 0] = values[:, None]

    def set_friction(self, ids: Any, values: np.ndarray) -> None:
        ids = self._ids(ids)
        values = np.asarray(values, dtype=np.float64).reshape(len(ids))
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("friction must be finite and positive")
        self.xi[ids, 0] = values
        self._apply_friction(ids)

    # ------------------------------------------------------------- interface
    def reset(self, ids: Any = None, state: np.ndarray | None = None) -> np.ndarray:
        ids = self._ids(ids)
        self.xi[ids, 0] = self.rng.uniform(*self.friction_range, size=len(ids))
        self._apply_friction(ids)
        t = self._tid(ids)
        qpos = self._qpos0.to(self._qpos.dtype).expand(len(ids), -1).clone()
        qpos[:, self._qadr] = self._initial_q.to(qpos.dtype)
        self._qpos[t] = qpos
        self._qvel[t] = 0
        self._qacc_warmstart[t] = 0
        self._ctrl[t] = 0
        self._time[t] = 0
        if self._act is not None:
            self._act[t] = 0
        self._forward()
        self.applied_action[ids] = self.initial_action[ids]
        self.ablation_action[ids] = self.initial_action[ids]
        self.torque[ids] = 0
        self.torque_peak[ids] = 0
        if state is not None:
            self.write_state(state, ids=ids)
        self._update_metrics()
        return self.read_state()

    def read_state_torch(self) -> torch.Tensor:
        """[p3, quat_wxyz4, jointq22, v_world3, omega_body3, jointqd22] on device."""
        bq, bv = self._bq, self._bv
        return torch.cat((self._qpos[:, bq:bq+7], self._qpos[:, self._qadr],
                          self._qvel[:, bv:bv+6], self._qvel[:, self._dadr]), dim=1)

    def read_state(self) -> np.ndarray:
        return self.read_state_torch().double().cpu().numpy()

    def write_state(self, state: np.ndarray, ids: Any = None) -> None:
        ids = self._ids(ids)
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (len(ids), 57) or not np.isfinite(state).all():
            raise ValueError(f"state must be finite with shape ({len(ids)}, 57)")
        quat_norm = np.linalg.norm(state[:, 3:7], axis=1)
        if np.any(quat_norm < 1e-10):
            raise ValueError("base quaternion must have nonzero norm")
        state = state.copy()
        state[:, 3:7] /= quat_norm[:, None]
        s = torch.as_tensor(state, dtype=self._qpos.dtype, device=self.device)
        t = self._tid(ids)
        bq, bv = self._bq, self._bv
        qpos, qvel = self._qpos[t], self._qvel[t]
        qpos[:, bq:bq+7] = s[:, :7]
        qpos[:, self._qadr] = s[:, 7:29]
        qvel[:, bv:bv+6] = s[:, 29:35]
        qvel[:, self._dadr] = s[:, 35:57]
        self._qpos[t] = qpos
        self._qvel[t] = qvel
        self._qacc_warmstart[t] = 0
        self._apply_friction(ids)
        self._forward()
        self._update_metrics()

    def _save_worlds(self, keep: torch.Tensor) -> dict[str, torch.Tensor]:
        saved = dict(qpos=self._qpos[keep].clone(), qvel=self._qvel[keep].clone(),
                     qacc_warmstart=self._qacc_warmstart[keep].clone(), time=self._time[keep].clone(),
                     ctrl=self._ctrl[keep].clone())
        if self._act is not None:
            saved["act"] = self._act[keep].clone()
        return saved

    def _restore_worlds(self, keep: torch.Tensor, saved: dict[str, torch.Tensor]) -> None:
        self._qpos[keep] = saved["qpos"]; self._qvel[keep] = saved["qvel"]
        self._qacc_warmstart[keep] = saved["qacc_warmstart"]; self._time[keep] = saved["time"]
        self._ctrl[keep] = saved["ctrl"]
        if self._act is not None:
            self._act[keep] = saved["act"]

    def step(self, action: np.ndarray, ids: Any = None) -> np.ndarray:
        """Advance selected worlds; unselected worlds are restored bit-exactly.

        Stepping a strict subset still costs a full batched step. Training uses
        all worlds every control step; subsets occur only in rare utilities.
        """
        ids = self._ids(ids)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, 22) or not np.isfinite(action).all():
            raise ValueError(f"action must be finite with shape ({self.num_envs}, 22)")
        if self.ablation != "none":
            raise NotImplementedError("use the CPU BatchedSimulator for ablation interventions")
        full = len(ids) == self.num_envs and np.array_equal(ids, np.arange(self.num_envs))
        dtype = self._qpos.dtype
        command = torch.as_tensor(np.clip(action, -1., 1.), dtype=dtype, device=self.device)
        q0, scale = self._q0.to(dtype), self._scale.to(dtype)
        low, high = self._low.to(dtype), self._high.to(dtype)
        requested = q0 + scale * command
        target = torch.minimum(torch.maximum(requested, low), high)
        wv = self._velocity[self._wheel].to(dtype)
        wheel_requested = scale[self._wheel] * command[:, self._wheel]
        wheel_target = torch.minimum(torch.maximum(wheel_requested, -wv), wv)
        pos = self._position
        executed = command.clone()
        executed[:, pos] = torch.where(target[:, pos] != requested[:, pos],
                                       (target[:, pos] - q0[pos]) / scale[pos], command[:, pos])
        executed[:, self._wheel] = torch.where(wheel_target != wheel_requested,
                                               wheel_target / scale[self._wheel], command[:, self._wheel])
        executed = executed.clamp(-1., 1.)
        keep = None if full else self._tid(np.setdiff1d(np.arange(self.num_envs), ids))
        saved = None if keep is None else self._save_worlds(keep)
        kp, kd = self._kp.to(dtype), self._kd.to(dtype)
        effort, velocity = self._effort.to(dtype), self._velocity.to(dtype)
        peak = torch.zeros((self.num_envs, 22), dtype=dtype, device=self.device)
        torque = peak
        for _ in range(self.substeps):
            q, qd = self._qpos[:, self._qadr], self._qvel[:, self._dadr]
            torque = kp * (target - q) - kd * qd
            torque[:, self._wheel] = kd[self._wheel] * (wheel_target - qd[:, self._wheel])
            torque = torch.where((qd.abs() > velocity) & (torque * qd > 0), torch.zeros_like(torque), torque)
            torque = torch.minimum(torch.maximum(torque, -effort), effort)
            self._ctrl[:, self._act_ids] = torque.to(self._ctrl.dtype)
            self._step_physics()
            peak = torch.maximum(peak, torque.abs())
        if saved is not None:
            self._restore_worlds(keep, saved)
        # Make contacts/kinematics agree with the integrated state.
        self._forward()
        sel = ids
        self.applied_action[sel] = executed.double().cpu().numpy()[sel]
        self.torque[sel] = torque.double().cpu().numpy()[sel]
        self.torque_peak[sel] = peak.double().cpu().numpy()[sel]
        self._update_metrics()
        return self.read_state()

    def settle(self, ids: Any = None, steps: int = 50) -> np.ndarray:
        if isinstance(steps, bool) or int(steps) != steps or steps < 0:
            raise ValueError("settling steps must be a nonnegative integer")
        ids = self._ids(ids)
        command = self.initial_action.copy()
        for _ in range(int(steps)):
            self.step(command, ids=ids)
        return self.read_state()

    # --------------------------------------------------------------- metrics
    def _update_metrics(self) -> None:
        n = self.num_envs
        dev = self.device
        xpos, xmat = self._xpos, self._xmat
        height = xpos[:, self._base_body, 2]
        tilt = torch.arccos(xmat[:, self._base_body, 2, 2].clamp(-1., 1.))
        wheel_contact = torch.zeros((n, 4), dtype=torch.bool, device=dev)
        wheel_force = torch.zeros((n * 4,), dtype=xpos.dtype, device=dev)
        slip = torch.zeros((n * 4,), dtype=xpos.dtype, device=dev)
        forbidden = torch.zeros(n, dtype=torch.bool, device=dev)
        nacon = int(self._nacon[0].item())
        if nacon > self.naconmax and not self._overflow_warned:
            import warnings
            warnings.warn(f"mujoco_warp contact buffer overflow ({nacon}>{self.naconmax}); increase sim.warp_nconmax")
            self._overflow_warned = True
        nacon = min(nacon, self.naconmax)
        if nacon:
            dist = self._c_dist[:nacon]
            active = dist <= 0
            geom = self._c_geom[:nacon].long()
            world = self._c_world[:nacon].long()
            g1, g2 = geom[:, 0], geom[:, 1]
            ground = (g1 == self._ground) | (g2 == self._ground)
            other = torch.where(g1 == self._ground, g2, g1).clamp_min(0)
            k = self._wheel_lookup[other]
            wheel_ground = active & ground & (k >= 0)
            forbidden_c = active & ~wheel_ground
            if forbidden_c.any():
                forbidden[world[forbidden_c]] = True
            adr = self._c_efc[:nacon, 0].long()
            dim = self._c_dim[:nacon].long()
            loaded = wheel_ground & (adr >= 0)
            if loaded.any():
                idx = loaded.nonzero().squeeze(-1)
                w, a, cd = world[idx], adr[idx], dim[idx]
                force = self._efc_force
                rows = torch.arange(2 * 5, device=dev)  # condim <= 6 -> <= 10 pyramid rows
                count = torch.where(cd > 1, 2 * (cd - 1), torch.ones_like(cd))
                address = a[:, None] + rows[None]
                valid = (rows[None] < count[:, None]) & (address < self.njmax)
                values = force[w[:, None].expand_as(address), address.clamp_max(self.njmax - 1)]
                normal = (values * valid).sum(-1).clamp_min(0.)
                ok = normal > 1e-6
                if ok.any():
                    w, kk = w[ok], k[idx][ok]
                    flat = w * 4 + kk
                    wheel_force.index_add_(0, flat, normal[ok].to(wheel_force.dtype))
                    # Contact-point velocity on the wheel body (world XY tangent),
                    # including wheel spin, base motion and articulated legs.
                    body = self._wheel_body[kk]
                    root = self._wheel_root[kk]
                    cvel = self._cvel[w, body]
                    r = self._c_pos[idx][ok] - self._subtree_com[w, root]
                    v = cvel[:, 3:] + torch.cross(cvel[:, :3], r, dim=-1)
                    speed = v[:, :2].norm(dim=-1).to(slip.dtype)
                    slip.scatter_reduce_(0, flat, speed, reduce="amax", include_self=True)
                    wheel_contact.view(-1)[flat] = True
        state = self.read_state_torch()
        finite = torch.isfinite(state).all(dim=1)
        q = state[:, 7:29]
        low, high = self._low.to(q.dtype), self._high.to(q.dtype)
        excess = torch.maximum(low - q, q - high)
        limit = excess.clamp_min(0.).max(dim=1).values
        safe = finite & (height > .20) & (tilt < 1.0) & (limit < .05) & ~forbidden
        cpu = lambda x: x.cpu().numpy()
        self._metrics = dict(wheel_contact=cpu(wheel_contact), contacts=cpu(wheel_contact),
                             wheel_normal_force=cpu(wheel_force.view(n, 4)).astype(np.float64),
                             slip=cpu(slip.view(n, 4)).astype(np.float64),
                             torque=self.torque.copy(), torque_peak=self.torque_peak.copy(),
                             base_height=cpu(height).astype(np.float64), tilt=cpu(tilt).astype(np.float64),
                             joint_limit_violation=cpu(limit).astype(np.float64),
                             forbidden_contact=cpu(forbidden), finite=cpu(finite), safe=cpu(safe))

    def metrics(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._metrics.items()}

    def set_ablation(self, mode: str = "none") -> None:
        if mode != "none":
            raise NotImplementedError("ablation interventions are evaluated with the CPU BatchedSimulator")
        self.ablation = mode

    @property
    def contacts(self) -> np.ndarray:
        return self._metrics["wheel_contact"].copy()

    @property
    def slip(self) -> np.ndarray:
        return self._metrics["slip"].copy()

    # ------------------------------------------------------- checkpointing
    def snapshot(self) -> dict[str, Any]:
        state = dict(qpos=self._qpos.cpu().numpy().copy(), qvel=self._qvel.cpu().numpy().copy(),
                     qacc_warmstart=self._qacc_warmstart.cpu().numpy().copy(),
                     time=self._time.cpu().numpy().copy(), ctrl=self._ctrl.cpu().numpy().copy())
        if self._act is not None:
            state["act"] = self._act.cpu().numpy().copy()
        return dict(backend="warp", warp_state=state, xi=self.xi.copy(),
                    applied_action=self.applied_action.copy(), torque=self.torque.copy(),
                    torque_peak=self.torque_peak.copy(),
                    rng=copy.deepcopy(self.rng.bit_generator.state), ablation=self.ablation,
                    ablation_action=self.ablation_action.copy(), num_envs=self.num_envs)

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot["num_envs"] != self.num_envs:
            raise ValueError("snapshot batch size differs")
        if snapshot.get("backend") != "warp":
            raise ValueError("snapshot was recorded by a different simulator backend")
        self.xi[:] = snapshot["xi"]
        self.applied_action[:] = snapshot["applied_action"]
        self.torque[:] = snapshot["torque"]
        self.torque_peak[:] = snapshot.get("torque_peak", np.abs(self.torque))
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.ablation = snapshot.get("ablation", "none")
        self.ablation_action[:] = snapshot.get("ablation_action", self.initial_action)
        state = snapshot["warp_state"]
        for name, view in (("qpos", self._qpos), ("qvel", self._qvel), ("qacc_warmstart", self._qacc_warmstart),
                           ("time", self._time), ("ctrl", self._ctrl), ("act", self._act)):
            if view is not None and name in state:
                view.copy_(torch.as_tensor(state[name], dtype=view.dtype, device=self.device))
        self._apply_friction(np.arange(self.num_envs))
        self._forward()
        self._update_metrics()

    def close(self) -> None:
        self._graphs.clear()


def main():
    """Server check: CUDA-vs-CPU parity on a few worlds, then batched throughput."""
    import argparse
    import json
    import time
    from .simulator import BatchedSimulator
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--asset-dir")
    p.add_argument("--arm-action-scale", type=float, default=None, help="default: full source range (as in configs)")
    p.add_argument("--no-graph", action="store_true")
    a = p.parse_args()
    kw = dict(seed=0, asset_dir=a.asset_dir, arm_action_scale=a.arm_action_scale)
    cpu = BatchedSimulator(num_envs=4, friction=1., **kw)
    gpu = WarpBatchedSimulator(model=cpu.model, spec=cpu.spec, num_envs=4, friction=1., device=a.device,
                               cuda_graph=not a.no_graph, arm_initial_position=None, seed=0)
    friction = np.array([.6, .8, 1.0, 1.2])
    cpu.set_friction(None, friction); gpu.set_friction(None, friction)
    cpu.settle(steps=50); gpu.settle(steps=50)
    rng = np.random.default_rng(0)
    action = np.clip(cpu.initial_action + rng.normal(0, .12, (4, 22)), -1, 1)
    for _ in range(50):
        cpu.step(action); gpu.step(action)
    sc, sg = cpu.read_state(), gpu.read_state()
    parity = dict(max_state_error=float(np.abs(sc - sg).max()),
                  max_position_error_m=float(np.abs(sc[:, :3] - sg[:, :3]).max()),
                  wheel_contact_equal=bool((cpu.metrics()["wheel_contact"] == gpu.metrics()["wheel_contact"]).all()),
                  safe_equal=bool((cpu.metrics()["safe"] == gpu.metrics()["safe"]).all()))
    print(json.dumps(dict(parity=parity)), flush=True)
    gpu.close()
    sim = WarpBatchedSimulator(model=cpu.model, spec=cpu.spec, num_envs=a.num_envs, device=a.device,
                               cuda_graph=not a.no_graph, seed=0)
    sim.settle(steps=5)
    command = np.clip(sim.initial_action + rng.normal(0, .12, sim.initial_action.shape), -1, 1)
    started = time.monotonic()
    for _ in range(a.steps):
        sim.step(command)
    elapsed = time.monotonic() - started
    m = sim.metrics()
    print(json.dumps(dict(num_envs=a.num_envs, control_steps=a.steps, seconds=round(elapsed, 3),
        env_steps_per_s=round(a.num_envs * a.steps / elapsed), cuda_graph=any(v is not None for v in sim._graphs.values()),
        safe_fraction=float(m["safe"].mean()), wheel_contact_fraction=float(m["wheel_contact"].mean()),
        contacts_in_use=int(sim._nacon[0].item()), contact_capacity=sim.naconmax)), flush=True)
    if parity["max_position_error_m"] > 1e-3 or not parity["wheel_contact_equal"]:
        raise SystemExit("Warp/CPU parity check failed; do not train with sim.backend='warp' on this setup")


if __name__ == "__main__":
    main()
