"""Record wide, continuous TCP pose commands executed by a saved WF policy.

python -m supermimic.wf.play_trajectories --checkpoint outputs/<training_run>
The checkpoint's verified trajectory library supplies commands; only the Actor
controls the robot. Complete paths play without random segment switches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from .config import CurriculumConfig, PolicyConfig, SimConfig, TrajectoryConfig
from .evaluate import evaluate_policy
from .goal_trajectory import TrajectoryLibrary
from .math import following_frame
from .networks import Actor
from .play import VideoRecorder, frame_count
from .preview_trajectory import prepare_preview, rank_trajectories
from .robot import build_model
from .simulator import BatchedSimulator
from .task import WholeBodyRewardCfg, WholeBodyTask


SCENARIOS = {"stationary": (0., 0., 0.), "forward": (.1, 0., 0.), "turn": (.1, 0., .03)}


def load_preview_checkpoint(path):
    """Load rollout settings, independent of training-only model/guard options."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "ckpt_latest.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "wf_goal_bptt_v1":
        raise ValueError("expected a WF Actor or full policy checkpoint, not a TorchScript export")
    raw = ck["config"]
    # Some runs contain extra exploration/guard training settings. Retain all
    # simulator settings, while policy architecture and reward parsing stay strict.
    cfg = SimpleNamespace(sim=SimpleNamespace(**(asdict(SimConfig()) | raw.get("sim", {}))),
        policy=PolicyConfig(**raw.get("policy", {})),
        curriculum=CurriculumConfig(**raw.get("curriculum", {})),
        trajectory=TrajectoryConfig(**raw.get("trajectory", {})), asset_dir=raw.get("asset_dir"))
    return path, ck, cfg


def resolve_library(directory, checkpoint):
    """Resolve saved relative paths even when launched outside the repository."""
    path = Path(directory).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path, checkpoint.parent / path,
                  *(parent / path for parent in checkpoint.parents)]
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"trajectory library not found: {directory}; use --library PATH")


class TrajectoryRecorder(VideoRecorder):
    """Observe physical rollout, annotate it and collect a control-rate trace."""

    def __init__(self, *args, preview, scenario, frame, follow_frame, initial_state,
                 initial_tcp, initial_action, initial_goal, command, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview, self.scenario = preview, scenario
        # Leave room above the arm for the complete path and the text overlay.
        self.camera_offset = np.array([0., 0., .5])
        self.frame, self.follow_frame, self.command = frame, follow_frame, command
        self.anchor_pos, self.anchor_quat = self._frame_pose(initial_state)
        stride = max(1, math.ceil((preview.length - 1) / 160))
        indices = np.unique(np.r_[np.arange(0, preview.length - 1, stride), preview.length - 2])
        self.path_local = preview.poses[0, indices, :3].cpu().numpy()
        self.state = np.asarray(initial_state).copy()
        self.times, self.states, self.actions = [0.], [self.state.copy()], [initial_action.copy()]
        self.tcp = [np.asarray(initial_tcp).copy()]
        # The prepared path begins exactly at the settled TCP pose.
        self.goals = [initial_goal.copy()]
        self.position_errors, self.orientation_errors, self.failures = [0.], [0.], [False]
        self.trail_goals, self.trail_tcp = [], []
        self.preview_frame = None

    def _frame_pose(self, state):
        p, q = following_frame(torch.as_tensor(state, dtype=torch.float64), self.follow_frame)
        return p.cpu().numpy(), q.cpu().numpy()

    def _polyline(self, points, color, width=2.):
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(end - start) < 1e-6:
                continue
            geom = self._geom(mujoco.mjtGeom.mjGEOM_LINE, [0, 0, 0], start, color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)

    def _markers(self, data, goal):
        super()._markers(data, goal)
        p, q = self._frame_pose(self.state) if self.frame == "follow" else (self.anchor_pos, self.anchor_quat)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q)
        self._polyline(self.path_local @ rotation.reshape(3, 3).T + p, [.1, .75, 1., .65])
        self._polyline(np.asarray(self.trail_goals[-240:]), [.1, 1., .25, .95], 3.)
        self._polyline(np.asarray(self.trail_tcp[-240:]), [1., .55, .05, 1.], 3.)

    def annotation_lines(self, goal, step, metrics):
        phase = "APPROACH" if step <= self.preview.lead_in_steps else "TRAJECTORY"
        hold_start = self.steps - round(self.preview.metadata["hold_s"] / self.dt)
        if step > hold_start:
            phase = "FINAL HOLD"
        status = "FAILED" if bool(metrics["failed"][0]) else phase
        span = self.preview.metadata["xyz_span_m"]
        return [f"Policy {self.iteration} | {self.scenario} | level {self.level} | "
                f"trajectory {self.preview.metadata['trajectory_id']} | {status}",
                f"t={step * self.dt:.2f}/{self.steps * self.dt:.2f}s | "
                f"path speed {self.preview.metadata['speed']:.2f}x | {self.frame} frame",
                f"TCP error: {float(metrics['position_error_m'][0]) * 100:.1f} cm | "
                f"orientation: {math.degrees(float(metrics['orientation_error_rad'][0])):.1f} deg",
                f"Base command: vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} m/s "
                f"wz={self.command[2]:+.3f} rad/s | path span xyz={span[0]:.2f}/{span[1]:.2f}/{span[2]:.2f} m",
                "Cyan: full target path | green: command trail | orange: actual TCP | RGB: target orientation"]

    def __call__(self, sim, goals, step, metrics):
        self.state = sim.read_state()[0].copy()
        data = sim.data[0]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[self.spec.tcp_site])
        tcp = np.r_[data.site_xpos[self.spec.tcp_site], quaternion]
        goal = goals[0].detach().cpu().numpy().copy()
        self.times.append(step * self.dt)
        self.states.append(self.state.copy())
        self.actions.append(sim.applied_action[0].copy())
        self.tcp.append(tcp)
        self.goals.append(goal)
        self.position_errors.append(float(metrics["position_error_m"][0]))
        self.orientation_errors.append(float(metrics["orientation_error_rad"][0]))
        self.failures.append(bool(metrics["failed"][0]))
        if frame_count(step, self.dt, self.fps) > self.frames or self.failures[-1]:
            self.trail_goals.append(goal[:3])
            self.trail_tcp.append(tcp[:3])
        super().__call__(sim, goals, step, metrics)
        if self.last_frame is not None and (self.preview_frame is None or step == self.steps // 2):
            self.preview_frame = self.last_frame.copy()
        if self.failures[-1]:
            self.preview_frame = self.last_frame.copy()

    def save_trace(self, path):
        np.savez_compressed(path, time_s=np.asarray(self.times), state=np.asarray(self.states),
            applied_action=np.asarray(self.actions), world_goal=np.asarray(self.goals),
            tcp_world_pose=np.asarray(self.tcp), position_error_m=np.asarray(self.position_errors),
            orientation_error_rad=np.asarray(self.orientation_errors), failed=np.asarray(self.failures),
            reference_pose_follow_frame=self.preview.poses[0, :-1].cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="training directory or WF checkpoint file")
    parser.add_argument("--library", help="override checkpoint's trajectory library directory")
    parser.add_argument("--output-dir", help="default: CHECKPOINT_DIR/videos/trajectories_TIMESTAMP")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--clips", type=int, default=1, help="widest distinct paths per scenario")
    parser.add_argument("--trajectory-id", type=int, help="explicit library ID; requires --clips 1")
    parser.add_argument("--level", type=int, choices=range(4), default=3)
    parser.add_argument("--speed", type=float, default=.75, help="path playback speed in (0, 1]")
    parser.add_argument("--lead-in-s", type=float, default=4., help="smooth approach to path start")
    parser.add_argument("--hold-s", type=float, default=2., help="hold final pose after complete path")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="override base commands (m/s, m/s, rad/s)")
    parser.add_argument("--frame", choices=["follow", "world"], default="follow",
                        help="follow: path moves with base, world: fixed initial frame")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--friction", type=float, default=1.)
    parser.add_argument("--device", default="cpu", help="inference device; physics runs on CPU")
    parser.add_argument("--asset-dir")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.)
    parser.add_argument("--camera-elevation", type=float, default=-22.)
    parser.add_argument("--appearance", choices=("visual", "collision"), default="visual",
                        help="URDF surface meshes (default), or collision geometry")
    args = parser.parse_args()
    if min(args.clips, args.fps, args.width, args.height) < 1:
        parser.error("clips, fps and frame dimensions must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("MP4 dimensions must be even")
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be finite and in (0, 1]")
    if not math.isfinite(args.lead_in_s) or args.lead_in_s <= 0:
        parser.error("lead-in-s must be finite and positive")
    if not math.isfinite(args.hold_s) or args.hold_s < 0:
        parser.error("hold-s must be finite and nonnegative")
    if any(not math.isfinite(x) or x <= 0 for x in [args.friction, args.camera_distance]):
        parser.error("friction and camera-distance must be finite and positive")
    if not all(math.isfinite(x) for x in [args.camera_azimuth, args.camera_elevation, *(args.command or [])]):
        parser.error("camera angles and velocity commands must be finite")
    if args.trajectory_id is not None and args.clips != 1:
        parser.error("--trajectory-id requires --clips 1")
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        parser.error("recording needs imageio, imageio-ffmpeg and Pillow in the WF Python environment")

    torch.set_num_threads(4)
    checkpoint, ck, cfg = load_preview_checkpoint(args.checkpoint)
    directory = args.library or cfg.trajectory.library
    if not directory:
        parser.error("checkpoint has no trajectory library; supply --library PATH")
    directory = resolve_library(directory, checkpoint)
    model, spec = build_model(args.asset_dir or cfg.asset_dir, cfg.sim.timestep,
        leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
        wheel_action_scale=cfg.sim.wheel_action_scale)
    from .train import model_manifest
    if model_manifest(spec)["physics_hash"] != ck["manifest"]["physics_hash"]:
        raise ValueError("checkpoint physics differs; use its original assets/configuration")
    task = WholeBodyTask(model, spec, args.device, WholeBodyRewardCfg(**ck["reward_config"]))
    if args.frame == "follow" and not task.cfg.base_command_conditioning:
        parser.error("follow frame requires a base-command-conditioned checkpoint; use --frame world")
    library = TrajectoryLibrary(directory, task, ck["manifest"]["physics_hash"], cfg.sim.dt)
    ids = ([args.trajectory_id] if args.trajectory_id is not None
           else rank_trajectories(library, args.level)[:args.clips])
    if len(ids) != args.clips:
        parser.error("not enough distinct trajectories at the requested level")
    if any(i < 0 or i >= library.count for i in ids):
        parser.error("trajectory ID is outside the library")
    actor = Actor(cfg.policy).to(args.device)
    actor.load_state_dict(ck["actor"], strict=True)
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    if args.frame == "world" and args.command is not None and any(args.command):
        parser.error("a world-fixed path cannot encode a nonzero base command; use --frame follow")
    commands = {name: tuple(args.command) if args.command is not None else SCENARIOS[name] for name in scenarios}
    if args.frame == "world":
        if any(any(command) for command in commands.values()):
            parser.error("use --scenario stationary with --frame world, or --command 0 0 0")
    if not task.cfg.base_command_conditioning and any(any(command) for command in commands.values()):
        parser.error("checkpoint does not support base velocity commands")
    out = (Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint.parent /
           "videos" / f"trajectories_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.mkdir(parents=True, exist_ok=True)
    from .visuals import create_visual_scene
    visual_scene = create_visual_scene(model, spec)
    summaries = []
    for clip, trajectory_id in enumerate(ids):
        seed = args.seed + clip
        fixture = BatchedSimulator(model, spec, num_envs=1, seed=seed, dt=cfg.sim.dt,
            friction=args.friction, arm_initial_position=cfg.sim.arm_initial_position)
        try:
            fixture.reset()
            fixture.settle(steps=cfg.sim.reset_settle_steps)
            initial_state, initial_action = fixture.read_state()[0].copy(), fixture.applied_action[0].copy()
            initial = torch.as_tensor(initial_state[None], dtype=torch.float32, device=task.device)
            initial_tcp = task.tcp_pose(initial)[0].cpu().numpy()
            preview = prepare_preview(library, directory, trajectory_id, fixture, speed=args.speed,
                                      lead_in_s=args.lead_in_s, hold_s=args.hold_s)
        finally:
            fixture.close()
        schedule_cfg = TrajectoryConfig(library=str(directory), fraction=1., segment_steps=preview.steps + 1,
            update_steps=1, continuity_candidates=1, lower_level_fraction=0.,
            tracking_grace_steps=preview.lead_in_steps)
        level = int(preview.levels[0])
        for scenario in scenarios:
            command = commands[scenario]
            video = out / f"{scenario}_level{level}_traj{trajectory_id:04d}.mp4"
            for path in (video, video.with_suffix('.json'), video.with_suffix('.npz'), video.with_suffix('.png')):
                if path.exists():
                    raise FileExistsError(f"preview exists; select a new --output-dir: {path}")
            goals = task.initialize_goal(initial, task.tcp_pose(initial))
            if task.cfg.base_command_conditioning:
                goals[:, 7:10] = goals.new_tensor(command)
                goals[:, 10] = float(args.frame == "follow")
            print(json.dumps(dict(recording=str(video), trajectory=preview.metadata,
                                  command=command), ensure_ascii=False), flush=True)
            with imageio.get_writer(str(video), fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error") as writer:
                recorder = TrajectoryRecorder(model, spec, writer, width=args.width, height=args.height,
                    fps=args.fps, dt=cfg.sim.dt, steps=preview.steps, iteration=ck["iteration"], level=level,
                    episodes=1, distance=args.camera_distance, preview=preview, scenario=scenario,
                    frame=args.frame, follow_frame=task.cfg.goal_follow_frame, initial_state=initial_state,
                    initial_tcp=initial_tcp, initial_action=initial_action,
                    initial_goal=goals[0].cpu().numpy(), command=command,
                    appearance=args.appearance, visual_scene=visual_scene)
                try:
                    recorder.camera.azimuth, recorder.camera.elevation = args.camera_azimuth, args.camera_elevation
                    recorder.begin_episode(1)
                    result = evaluate_policy(actor, task, model, spec, episodes=1, level=level, seed=seed,
                        steps=preview.steps, hold_steps=min(cfg.curriculum.hold_steps, preview.steps),
                        dt=cfg.sim.dt, friction=args.friction, goals=goals,
                        settle_steps=cfg.sim.reset_settle_steps, goal_seed_device=cfg.curriculum.goal_seed_device,
                        arm_initial_position=cfg.sim.arm_initial_position, frame_callback=recorder,
                        trajectories=preview, trajectory_cfg=schedule_cfg)
                    recorder.end_episode()
                    recorder.save_trace(video.with_suffix(".npz"))
                    if recorder.preview_frame is not None:
                        recorder.Image.fromarray(recorder.preview_frame).save(video.with_suffix(".png"))
                finally:
                    recorder.close()
            report = dict(checkpoint=str(checkpoint), checkpoint_iteration=ck["iteration"],
                physics_hash=ck["manifest"]["physics_hash"], video=str(video), trace=str(video.with_suffix(".npz")),
                scenario=scenario, frame=args.frame, command=command, seed=seed, friction=args.friction,
                appearance=args.appearance, floor="white_light_gray_grid",
                fps=args.fps, frames=recorder.total_frames, duration_s=recorder.total_frames / args.fps,
                episode_end_pause_s=1., trajectory=preview.metadata, evaluation=result,
                note="Full library path at control rate, with smooth approach and final hold; "
                     "tracking statistics exclude approach, include final hold, and count unexecuted failure steps as misses.")
            video.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            record = result["records"][0]
            item = dict(video=str(video), report=str(video.with_suffix('.json')),
                trajectory_id=trajectory_id, scenario=scenario, failed=record["failed"],
                executed_s=record["steps"] * cfg.sim.dt, tracking_success_fraction=record["tracking_success_fraction"],
                tracking_rms_position_error_m=record["tracking_rms_position_error_m"],
                tracking_rms_orientation_error_rad=record["tracking_rms_orientation_error_rad"])
            summaries.append(item)
            (out / "summary.json").write_text(json.dumps(dict(checkpoint=str(checkpoint),
                checkpoint_iteration=ck["iteration"], clips=summaries), indent=2, allow_nan=False) + "\n")
            print(json.dumps(item), flush=True)
    print(json.dumps(dict(output_dir=str(out), videos=len(summaries), summary=str(out / 'summary.json'))), flush=True)


if __name__ == "__main__":
    main()
