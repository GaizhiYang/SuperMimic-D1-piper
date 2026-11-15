"""Headless, shareable plots of measured and commanded TCP reference paths."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _routes_for_plot(routes):
    """Validate before creating an output file or importing the plotting backend."""
    validated = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise ValueError(f"route {index} must be a dictionary")
        label = str(route.get("label", f"Route {index + 1}"))
        try:
            dt = float(route["dt"])
            positions = np.asarray(route["positions"], dtype=float)
            level = int(route["level"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"route {index} requires level, dt and numeric positions") from error
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError(f"route {index} dt must be finite and positive")
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError(f"route {index} positions must have shape (N, 3), with N >= 1")
        if not np.isfinite(positions).all():
            raise ValueError(f"route {index} positions contain nonfinite values")
        targets = route.get("target_positions")
        if targets is not None:
            try:
                targets = np.asarray(targets, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"route {index} target_positions must be numeric") from error
            if targets.shape != positions.shape or not np.isfinite(targets).all():
                raise ValueError(f"route {index} target_positions must be finite and match positions")
        validated.append((label, level, dt, positions, targets))
    if not validated:
        raise ValueError("at least one reference route is required")
    return validated


def plot_reference_paths(routes, output, *, title="Reference trajectories", frame="world"):
    """Save a TCP path and three coordinate histories to a standalone image.

    Each route contains ``label``, ``level``, ``dt`` (seconds) and ``positions``
    of shape ``(N, 3)`` in metres. Optional ``target_positions`` must have the
    same shape and are drawn dashed. Coordinates are already expressed in
    ``frame`` by the caller; this function does not transform robot poses.
    Matplotlib is imported only when plotting is requested and uses Agg without
    opening a graphical window. The returned path is the written file.
    """
    routes = _routes_for_plot(routes)
    try:
        from matplotlib import colormaps, rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError("Reference plots require matplotlib; install it with "
                           "`python -m pip install matplotlib`.") from error

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.titleweight": "normal", "axes.edgecolor": "#c7cdd4",
        "axes.labelcolor": "#374151", "text.color": "#1f2937", "xtick.color": "#586474",
        "ytick.color": "#586474", "grid.color": "#e6eaf0", "grid.linewidth": .7,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    }
    with rc_context(style):
        figure = Figure(figsize=(13, 9), dpi=180)
        FigureCanvasAgg(figure)
        path_axis = figure.add_subplot(2, 2, 1, projection="3d")
        axes = [figure.add_subplot(2, 2, panel) for panel in (2, 3, 4)]
        color_map = colormaps["tab10" if len(routes) <= 10 else "tab20"]
        colors = [color_map(index % color_map.N) for index in range(len(routes))]
        handles = []
        has_targets = False
        bounds = []
        for (label, level, dt, positions, targets), color in zip(routes, colors):
            time = np.arange(len(positions)) * dt
            legend_label = f"{label} | level {level}"
            handles.append(Line2D([], [], color=color, lw=2., label=legend_label))
            path_axis.plot(*positions.T, color=color, lw=1.8, alpha=.95)
            path_axis.scatter(*positions[0], color=color, s=25, depthshade=False,
                              edgecolor="white", linewidth=.6)
            path_axis.scatter(*positions[-1], color=color, s=35, marker="X", depthshade=False,
                              edgecolor="white", linewidth=.5)
            bounds.append(positions)
            for coordinate, axis in enumerate(axes):
                axis.plot(time, positions[:, coordinate], color=color, lw=1.7, alpha=.95)
            if targets is not None:
                has_targets = True
                bounds.append(targets)
                path_axis.plot(*targets.T, color=color, lw=1.2, ls="--", alpha=.65)
                for coordinate, axis in enumerate(axes):
                    axis.plot(time, targets[:, coordinate], color=color, lw=1.15,
                              ls="--", alpha=.65)

        # A common metre scale prevents the 3-D view from exaggerating small motion.
        all_positions = np.concatenate(bounds)
        lower, upper = all_positions.min(axis=0), all_positions.max(axis=0)
        center = (lower + upper) / 2.
        half_span = max(float(np.max(upper - lower)) * .57, .025)
        path_axis.set_xlim(center[0] - half_span, center[0] + half_span)
        path_axis.set_ylim(center[1] - half_span, center[1] + half_span)
        path_axis.set_zlim(center[2] - half_span, center[2] + half_span)
        path_axis.set_box_aspect((1., 1., 1.))
        path_axis.set_proj_type("ortho")
        path_axis.view_init(elev=25, azim=-58)
        path_axis.set_title(f"TCP path ({frame} frame)", pad=12)
        for coordinate, label in zip((path_axis.xaxis, path_axis.yaxis, path_axis.zaxis),
                                     ("x (m)", "y (m)", "z (m)")):
            coordinate.set_label_text(label)
            coordinate.labelpad = 7
            coordinate.set_pane_color((1., 1., 1., 1.))
        path_axis.tick_params(labelsize=8, pad=1)
        for coordinate, axis in enumerate(axes):
            name = "xyz"[coordinate]
            axis.set_title(f"TCP {name} coordinate", loc="left", pad=10)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(f"{name} (m)")
            axis.grid(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=.015, y=.10)

        # A figure-level legend keeps all data panels clear even for many routes.
        if has_targets:
            handles.append(Line2D([], [], color="#626d7c", lw=1.5, ls="--", label="Target"))
        handles.extend((
            Line2D([], [], color="#626d7c", lw=0, marker="o", markersize=5, label="Start"),
            Line2D([], [], color="#626d7c", lw=0, marker="X", markersize=6, label="End"),
        ))
        columns = min(4, len(handles))
        legend_rows = math.ceil(len(handles) / columns)
        legend_height = min(.30, .026 * legend_rows + .035)
        figure.suptitle(title, fontsize=18, weight="bold", x=.065, y=.975, ha="left")
        figure.text(.065, .935, f"{len(routes)} route{'s' if len(routes) != 1 else ''}  |  "
                    f"Coordinates in metres  |  {frame} frame", color="#667384", fontsize=10)
        figure.subplots_adjust(left=.065, right=.97, top=.875, bottom=.08 + legend_height,
                               hspace=.43, wspace=.24)
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .022),
                      ncol=columns, frameon=False, fontsize=9, columnspacing=1.8,
                      handlelength=2.4)
        figure.savefig(output, dpi=180, facecolor="white")
        figure.clear()
    return output
