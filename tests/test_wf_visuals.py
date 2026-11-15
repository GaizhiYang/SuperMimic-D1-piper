"""Video presentation follows physical poses and never changes trained physics."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def visual_robot():
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf.robot import build_model
    from supermimic.wf.train import model_manifest
    from supermimic.wf.visuals import RobotVisuals

    model, spec = build_model(arm_action_scale=None)
    before = model_manifest(spec)
    appearance = dict(rgba=model.geom_rgba.copy(), groups=model.geom_group.copy(),
                      materials=model.geom_matid.copy())
    visuals = RobotVisuals(model, spec)
    assert model_manifest(spec) == before
    np.testing.assert_array_equal(model.geom_rgba, appearance["rgba"])
    np.testing.assert_array_equal(model.geom_group, appearance["groups"])
    np.testing.assert_array_equal(model.geom_matid, appearance["materials"])
    return model, spec, visuals


@pytest.mark.private_assets
def test_outer_meshes_replace_collision_shapes_only_in_video(visual_robot):
    model, spec, visuals = visual_robot
    urdf = ET.parse(spec.asset_dir / "urdf/wf-arm-tbot.urdf").getroot()
    mesh_bodies = [link.get("name") for link in urdf.findall("link")
                   if link.find("visual/geometry/mesh") is not None]
    assert len(mesh_bodies) == 33
    assert visuals.model.nmesh > model.nmesh
    for body in mesh_bodies:
        geom = visuals.model.geom(f"wf_outer_{body}_0")
        assert geom.type == mujoco.mjtGeom.mjGEOM_MESH
        assert geom.contype == geom.conaffinity == 0
        assert visuals.option.geomgroup[geom.group.item()]
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name != "wf_ground":
            assert not visuals.option.geomgroup[visuals.model.geom(name).group.item()]
    assert visuals.option.geomgroup[visuals.model.geom("wf_ground").group.item()]


@pytest.mark.private_assets
def test_checker_floor_is_white_gray_and_render_only(visual_robot):
    model, spec, visuals = visual_robot
    assert model.geom_matid[spec.ground_geom] == -1
    floor = visuals.model.geom("wf_ground")
    material = floor.matid.item()
    texture = int(visuals.model.mat_texid[material, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    assert texture >= 0
    pixels = visuals.model.tex_data.reshape(-1, 3)
    np.testing.assert_array_equal(pixels[:, 0], pixels[:, 1])
    np.testing.assert_array_equal(pixels[:, 0], pixels[:, 2])
    assert pixels.max() == 255
    assert 225 <= pixels.min() <= 235
    assert len(np.unique(pixels[:, 0])) == 2


@pytest.mark.private_assets
def test_render_sync_preserves_physics_and_matches_all_link_poses(visual_robot):
    model, spec, visuals = visual_robot
    from supermimic.wf.train import model_manifest

    data = mujoco.MjData(model)
    data.qpos[spec.qpos_indices] = spec.q0
    data.qpos[spec.base_qpos_index:spec.base_qpos_index + 3] = [.2, -.1, .6]
    data.qpos[spec.base_qpos_index + 3:spec.base_qpos_index + 7] = [np.cos(.3), 0, 0, np.sin(.3)]
    data.qpos[spec.qpos_indices[16:22]] += [.2, -.2, .15, -.1, .1, .1]
    mujoco.mj_forward(model, data)
    data.time = 1.2
    before_state = data.__getstate__()
    before_hash = model_manifest(spec)["physics_hash"]
    rendered = visuals.sync(data)
    assert rendered.time == data.time
    assert data.__getstate__() == before_state
    assert model_manifest(spec)["physics_hash"] == before_hash
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        target = mujoco.mj_name2id(visuals.model, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(rendered.xpos[target], data.xpos[body], rtol=0, atol=1e-12)
        np.testing.assert_allclose(rendered.xquat[target], data.xquat[body], rtol=0, atol=1e-12)
    target_site = mujoco.mj_name2id(visuals.model, mujoco.mjtObj.mjOBJ_SITE, "wf_tcp")
    np.testing.assert_allclose(rendered.site_xpos[target_site], data.site_xpos[spec.tcp_site], rtol=0, atol=1e-12)


@pytest.mark.private_assets
@pytest.mark.parametrize("presentation", [False, True])
def test_studio_lights_and_quality_leave_physics_unchanged(visual_robot, presentation):
    model, spec, default = visual_robot
    from supermimic.wf.train import model_manifest
    from supermimic.wf.visuals import RobotVisuals

    physical_hash = model_manifest(spec)["physics_hash"]
    physical_lights, physical_samples = model.nlight, model.vis.quality.offsamples
    before_rgba = model.geom_rgba.copy()
    studio = RobotVisuals(model, spec, presentation=presentation, shadows=True,
                          reflections=True, msaa=4)
    assert studio.model.nlight == default.model.nlight + 2
    assert studio.model.light("wf_video_key").castshadow
    assert not studio.model.light("wf_video_fill").castshadow
    assert studio.model.vis.quality.offsamples == 4
    assert studio.model.vis.quality.shadowsize == 4096
    assert studio.model.vis.map.shadowclip == 2.5
    assert any(studio.model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX) == presentation
    floor_material = studio.model.geom("wf_ground").matid.item()
    assert studio.model.mat_reflectance[floor_material] == pytest.approx(.12)
    assert studio.model.mat_emission[floor_material] < default.model.mat_emission[
        default.model.geom("wf_ground").matid.item()]

    data = mujoco.MjData(model)
    data.qpos[spec.qpos_indices] = spec.q0
    data.qpos[spec.base_qpos_index:spec.base_qpos_index + 3] = [3., -2., .6]
    mujoco.mj_forward(model, data)
    before_state = data.__getstate__()
    studio.sync(data)
    np.testing.assert_allclose(studio.model.stat.center, data.xpos[spec.base_body] + [0, 0, .25])
    assert data.__getstate__() == before_state
    assert model_manifest(spec)["physics_hash"] == physical_hash
    assert (model.nlight, model.vis.quality.offsamples) == (physical_lights, physical_samples)
    np.testing.assert_array_equal(model.geom_rgba, before_rgba)
