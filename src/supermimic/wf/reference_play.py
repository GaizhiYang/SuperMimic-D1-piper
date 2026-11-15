"""Visualize saved WF physical references or TCP command trajectories.

Example: python -m supermimic.wf.reference_play --reference-dir outputs/wf_cmd_refs --presentation
Recorded physical states are replayed directly. Command trajectories animate
their saved arm joints at the nominal stance; neither mode runs a policy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from .evaluate import config_from_dict
from .math import following_frame, quat_angle, quat_apply, quat_apply_inverse, quat_error, quat_mul
from .play import VideoRecorder, frame_count
from .reference_plots import plot_reference_paths
from .reference_view_data import ReferenceCollection
from .robot import build_model
from .simulator import BatchedSimulator
from .task import WholeBodyTask


def write_reference_state(model, spec, data, state, time_s):
    """Restore a recorded pose and velocity, without advancing any dynamics."""
    state = np.asarray(state, dtype=float)
    if state.shape != (57,) or not np.isfinite(state).all():
        raise ValueError("reference state must be finite with shape (57,)")
    norm = np.linalg.norm(state[3:7])
    if norm < 1e-10:
        raise ValueError("reference base quaternion must have nonzero norm")
    bq, bv = spec.base_qpos_index, spec.base_dof_index
    data.qpos[bq:bq + 7] = state[:7]
    data.qpos[bq + 3:bq + 7] /= norm
    data.qpos[spec.qpos_indices] = state[7:29]
    data.qvel[bv:bv + 6] = state[29:35]
    data.qvel[spec.dof_indices] = state[35:57]
    data.time = time_s
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


@torch.no_grad()
def prepare_reference(sequence, task, *, nominal=None, plot_frame="body", follow_frame="body"):
    """Return original/reconstructed states, correctly transported goals and TCPs."""
    if sequence.states is not None:
        canonical = np.array(sequence.states, dtype=float, copy=True)
        states = torch.tensor(canonical, dtype=torch.float32)
        initial_goal = torch.tensor(sequence.goal, dtype=torch.float32)[None]
        goals = task.transport_goal(states[:1], states,
                                    initial_goal.expand(len(states), -1).clone())
    else:
        if nominal is None:
            raise ValueError("TCP command visualization requires its settled nominal stance")
        canonical = np.broadcast_to(nominal, (sequence.samples, 57)).copy()
        canonical[:, 23:29] = sequence.joints
        canonical[:, 29:] = 0.
        states = torch.tensor(canonical, dtype=torch.float32)
        poses = torch.tensor(sequence.poses, dtype=torch.float32)
        fp, fq = following_frame(states, follow_frame)
        goals = torch.cat((fp + quat_apply(fq, poses[:, :3]),
                           quat_mul(fq, poses[:, 3:7]), torch.zeros(len(states), 4)), dim=-1)
        goals[:, 10] = 1.
    tcp = task.tcp_pose(states)
    position_error = (tcp[:, :3] - goals[:, :3]).norm(dim=-1)
    orientation_error = quat_angle(quat_error(tcp[:, 3:7], goals[:, 3:7]))
    positions, targets = tcp[:, :3], goals[:, :3]
    if plot_frame == "body":
        fp, fq = following_frame(states, "body")
        positions = quat_apply_inverse(fq, positions - fp)
        targets = quat_apply_inverse(fq, targets - fp)
    elif plot_frame != "world":
        raise ValueError("plot_frame must be body or world")
    route = dict(label=sequence.label, level=sequence.level, dt=sequence.dt,
                 positions=positions.numpy().copy())
    if sequence.states is not None:
        route["target_positions"] = targets.numpy().copy()
    return canonical, goals, position_error, orientation_error, route


class ReferenceRecorder(VideoRecorder):
    """Keep command-only animation visibly distinct from recorded physics."""

    def _annotate(self, picture, goal, step, metrics, failed):
        if self.goal_source != "kinematic_command" or self.hud == "none":
            return super()._annotate(picture, goal, step, metrics, failed)
        overlay = self.Image.new("RGBA", picture.size, (0, 0, 0, 0))
        draw = self.ImageDraw.Draw(overlay)
        pad, line = max(12, picture.height // 55), max(24, picture.height // 30)
        detail = (f"{step * self.dt:.1f} / {self.steps * self.dt:.1f} s | "
                  "Saved arm joints / kinematic replay")
        width = min(picture.width - 2 * pad, max(
            draw.textbbox((0, 0), self.title, font=self.font)[2],
            draw.textbbox((0, 0), detail, font=self.caption_font)[2]) + 2 * pad)
        draw.rounded_rectangle((pad, pad, pad + width, pad * 3 + line * 2),
                               radius=pad // 2, fill=(20, 28, 40, 215))
        draw.text((2 * pad, 2 * pad), self.title, font=self.font, fill=(245, 248, 252, 255))
        draw.text((2 * pad, 2 * pad + line), detail, font=self.caption_font, fill=(213, 225, 238, 255))
        return self.Image.alpha_composite(picture.convert("RGBA"), overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, help="pipeline reference root, library/, or trajectories/")
    parser.add_argument("--source", choices=("auto", "clips", "trajectories"), default="auto",
                        help="auto selects recorded physical clips when both kinds are available")
    parser.add_argument("--indices", type=int, nargs="+", help="zero-based clip/trajectory ids to replay in order")
    parser.add_argument("--level", type=int, choices=range(4))
    parser.add_argument("--count", type=int, default=3, help="representative sequences when indices omitted (default: 3)")
    parser.add_argument("--list", action="store_true", help="list collection statistics and selected references, then exit")
    parser.add_argument("--config", help="training config JSON when a standalone TCP library has no sibling library/config")
    parser.add_argument("--asset-dir")
    parser.add_argument("--video", help="output MP4 (default: REFERENCE_ROOT/visualizations/references_SOURCE.mp4)")
    parser.add_argument("--plot", help="overview PNG (default: same basename as MP4)")
    parser.add_argument("--plot-only", action="store_true", help="write curves without rendering a video")
    parser.add_argument("--no-plot", action="store_true", help="only render video, without matplotlib")
    parser.add_argument("--plot-frame", choices=("body", "world"), default="body",
                        help="curve coordinates; body isolates arm motion from base travel")
    parser.add_argument("--presentation", action="store_true", help="studio lights, shadows, 1080p/50 FPS defaults")
    parser.add_argument("--hud", choices=("compact", "none"), default="compact")
    parser.add_argument("--reflections", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.plot_only and args.no_plot:
        parser.error("--plot-only and --no-plot cannot be used together")
    args.width = args.width if args.width is not None else 1920 if args.presentation else 1280
    args.height = args.height if args.height is not None else 1080 if args.presentation else 720
    args.fps = args.fps if args.fps is not None else 50 if args.presentation else 30
    if min(args.width, args.height, args.fps) < 1 or args.width % 2 or args.height % 2:
        parser.error("frame dimensions must be positive/even and FPS must be positive")
    if not math.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("camera distance must be finite and positive")
    collection = ReferenceCollection(args.reference_dir, source=args.source)
    selected = collection.select(indices=args.indices, level=args.level, count=args.count)
    sequences = [collection.load(index) for index in selected]
    counts = {int(level): int((collection.levels == level).sum()) for level in np.unique(collection.levels)}
    summary = dict(source=str(collection.directory), kind=collection.kind, count=collection.count,
        levels=counts, selected=[dict(index=s.index, label=s.label, level=s.level,
                                    samples=s.samples, duration_s=s.duration_s) for s in sequences])
    print(json.dumps(summary), flush=True)
    if args.list:
        return
    raw = json.loads(Path(args.config).expanduser().read_text()) if args.config else collection.raw_config
    if raw is None:
        parser.error("no generating config found; specify --config /path/to/pipeline_config.json")
    cfg = config_from_dict(raw)
    if args.asset_dir:
        cfg.asset_dir = args.asset_dir
    if not np.isclose(cfg.sim.dt, collection.dt):
        raise ValueError("reference sample period differs from the generating config")
    if (collection.kind == "trajectories" and
            not np.array_equal(collection.manifest.get("arm_initial_position"), cfg.sim.arm_initial_position)):
        raise ValueError("reference initial arm pose differs from the generating config")
    torch.set_num_threads(4)
    model, spec = build_model(cfg.asset_dir, cfg.sim.timestep, leg_action_scale=cfg.sim.leg_action_scale,
        arm_action_scale=cfg.sim.arm_action_scale, wheel_action_scale=cfg.sim.wheel_action_scale)
    from .train import model_manifest
    if model_manifest(spec)["physics_hash"] != collection.manifest["provenance"]["physics_hash"]:
        raise ValueError("reference physics differs; use the original assets/config")
    task = WholeBodyTask(model, spec, "cpu", cfg.reward)
    nominal = None
    if collection.kind == "trajectories":
        sim = BatchedSimulator(model, spec, num_envs=1, seed=0, friction=1., dt=cfg.sim.dt,
                               arm_initial_position=cfg.sim.arm_initial_position)
        try:
            sim.reset(); sim.settle(steps=cfg.sim.reset_settle_steps)
            nominal = sim.read_state()[0].copy()
        finally:
            sim.close()
        if collection.manifest["follow_frame"] != cfg.reward.goal_follow_frame:
            raise ValueError("reference following frame differs from the generating config")
    prepared = [prepare_reference(s, task, nominal=nominal, plot_frame=args.plot_frame,
                                   follow_frame=cfg.reward.goal_follow_frame) for s in sequences]
    root = collection.directory.parent if collection.directory.name in ("library", "trajectories") else collection.directory
    name = "references_" + collection.kind + (f"_level{args.level}" if args.level is not None else "")
    video = Path(args.video).expanduser().resolve() if args.video else root / "visualizations" / f"{name}.mp4"
    plot = Path(args.plot).expanduser().resolve() if args.plot else video.with_suffix(".png")
    if video.suffix.lower() != ".mp4" or plot.suffix.lower() != ".png":
        parser.error("--video must end in .mp4 and --plot in .png")
    if not args.no_plot:
        plot_reference_paths([p[-1] for p in prepared], plot,
            title="Recorded physical reference paths" if collection.kind == "clips" else "TCP command trajectories (kinematic)",
            frame=args.plot_frame)
        print(json.dumps(dict(plot=str(plot))), flush=True)
    records = []
    if not args.plot_only:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            parser.error("video dependencies missing; install imageio, imageio-ffmpeg and pillow")
        video.parent.mkdir(parents=True, exist_ok=True)
        data = mujoco.MjData(model)
        scene = SimpleNamespace(data=[data])
        source = "recorded_reference" if collection.kind == "clips" else "kinematic_command"
        with imageio.get_writer(str(video), fps=args.fps, codec="libx264", quality=9 if args.presentation else 8,
                                macro_block_size=1, ffmpeg_log_level="error") as writer:
            recorder = ReferenceRecorder(model, spec, writer, width=args.width, height=args.height,
                fps=args.fps, dt=collection.dt, steps=sequences[0].samples - 1, iteration=0,
                level=sequences[0].level, episodes=len(sequences), distance=args.camera_distance,
                goal_source=source, presentation=args.presentation, hud=args.hud,
                shadows=args.presentation, reflections=args.reflections, elevation=-18. if args.presentation else -22.)
            try:
                for episode, (sequence, prepared_sequence) in enumerate(zip(sequences, prepared), 1):
                    states, goals, pe, re, _ = prepared_sequence
                    recorder.level, recorder.steps = sequence.level, sequence.samples - 1
                    recorder.title = ("Recorded reference" if collection.kind == "clips" else "Kinematic TCP command") + f" | {sequence.label} | level {sequence.level}"
                    recorder.begin_episode(episode)
                    start_frame = recorder.total_frames
                    for index in range(1, sequence.samples):
                        if frame_count(index, collection.dt, args.fps) <= recorder.frames:
                            continue
                        write_reference_state(model, spec, data, states[index], index * collection.dt)
                        failed = bool(sequence.failed[index - 1]) if sequence.failed is not None else False
                        recorder(scene, goals[index:index + 1], index, dict(
                            position_error_m=pe[index:index + 1], orientation_error_rad=re[index:index + 1],
                            failed=torch.tensor([failed]), held_success=torch.tensor([False])))
                    recorder.end_episode()
                    records.append(dict(index=sequence.index, label=sequence.label, level=sequence.level,
                        duration_s=sequence.duration_s, frames=recorder.total_frames - start_frame,
                        goal=sequence.goal.tolist() if sequence.goal is not None else None,
                        final_position_error_m=float(pe[-1]), final_orientation_error_rad=float(re[-1]),
                        metadata=sequence.metadata))
                    print(json.dumps(dict(completed=episode, total=len(sequences), index=sequence.index)), flush=True)
            finally:
                recorder.close()
        summary.update(video=str(video), fps=args.fps, frames=recorder.total_frames,
            duration_s=recorder.total_frames / args.fps, episode_end_pause_s=1.)
    summary.update(plot=str(plot) if not args.no_plot else None,
        manifest_fingerprint=collection.fingerprint, physics_hash=collection.manifest["provenance"]["physics_hash"],
        replay_mode="saved physical states" if collection.kind == "clips" else "saved arm joints at nominal stance (kinematic)",
        policy_inference=False, plot_frame=args.plot_frame, records=records)
    report = (plot if args.plot_only else video).with_suffix(".json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(report=str(report), video=summary.get("video"), plot=summary["plot"])), flush=True)


if __name__ == "__main__":
    main()
