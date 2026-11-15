"""Standalone reference figures work without a desktop and reject bad archives."""
import numpy as np
import pytest

from supermimic.wf.reference_plots import plot_reference_paths


def test_reference_plot_supports_different_route_lengths_and_targets(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    positions = np.array([[.1, .2, .3], [.2, .4, .6], [.3, .1, .5]])
    routes = [
        dict(label="Measured clip 12", level=2, dt=.02, positions=positions,
             target_positions=positions + .01),
        dict(label="Command route 4", level=0, dt=.1, positions=np.ones((1, 3))),
    ]
    output = tmp_path / "nested" / "paths.png"
    assert plot_reference_paths(routes, output, frame="body") == output
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (2340, 1620)
        pixels = np.asarray(image.convert("RGB"))
    assert (pixels < 230).any()
    assert (pixels[:10] == 255).all()


@pytest.mark.parametrize("route", [
    dict(level=0, dt=0., positions=np.zeros((2, 3))),
    dict(level=0, dt=np.inf, positions=np.zeros((2, 3))),
    dict(level=0, dt=.02, positions=np.zeros((0, 3))),
    dict(level=0, dt=.02, positions=np.zeros((2, 4))),
    dict(level=0, dt=.02, positions=np.full((2, 3), np.nan)),
    dict(level=0, dt=.02, positions=np.zeros((2, 3)), target_positions=np.zeros((3, 3))),
])
def test_invalid_plot_data_does_not_write_output(tmp_path, route):
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        plot_reference_paths([route], output)
    assert not output.exists()


def test_empty_reference_selection_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        plot_reference_paths([], tmp_path / "empty.png")
