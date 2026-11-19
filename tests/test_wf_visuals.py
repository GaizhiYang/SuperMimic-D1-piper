"""URDF appearance follows physical FK without changing policy rollouts."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest


@pytest.fixture
def physical_sim():
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf.simulator import BatchedSimulator

    sim = BatchedSimulator(asset_dir=os.environ["WFARM_ASSET_DIR"], num_envs=1,
                           arm_action_scale=None, dt=.02, friction=1., seed=17)
    try:
        sim.settle(steps=100)
        yield sim
    finally:
        sim.close()


def assert_snapshot_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key


@pytest.mark.private_assets
def test_visual_meshes_follow_all_physical_links_and_fixed_fingers(physical_sim):
    import mujoco

    from supermimic.wf.visuals import create_visual_scene

    sim = physical_sim
    scene = create_visual_scene(sim.model, sim.spec)
    try:
        assert scene.model is not sim.model and scene.data is not sim.data[0]
        for moved in (False, True):
            if moved:
                state = sim.read_state()
                state[0, :3] += [.3, -.2, .1]
                mujoco.mju_axisAngle2Quat(state[0, 3:7], np.array([0., 0., 1.]), .47)
                state[0, 7:29] += .03*np.sin(np.arange(22))
                state[0, 29:] = np.linspace(-.2, .3, 28)
                sim.write_state(state)
                sim.data[0].time = 3.25
            scene.update(sim.data[0])
            np.testing.assert_array_equal(scene.data.qpos, sim.data[0].qpos)
            np.testing.assert_array_equal(scene.data.qvel, sim.data[0].qvel)
            assert scene.data.time == sim.data[0].time
            # Name-based comparisons include every fixed URDF link, especially
            # the welded gripper fingers, and do not assume body-ID ordering.
            for body in range(1, sim.model.nbody):
                name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, body)
                visual_body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
                assert visual_body >= 0, name
                np.testing.assert_allclose(scene.data.xpos[visual_body], sim.data[0].xpos[body],
                                           atol=1e-12, rtol=0., err_msg=name)
                np.testing.assert_allclose(scene.data.xmat[visual_body], sim.data[0].xmat[body],
                                           atol=1e-12, rtol=0., err_msg=name)
            np.testing.assert_allclose(scene.data.site_xpos[:sim.model.nsite], sim.data[0].site_xpos,
                                       atol=1e-12, rtol=0.)

        visible = scene.model.geom_group == 2
        assert visible.any() and bool(scene.option.geomgroup[2])
        assert np.all(scene.model.geom_contype[visible] == 0)
        assert np.all(scene.model.geom_conaffinity[visible] == 0)
        root = ET.parse(sim.spec.asset_dir / "urdf/wf-arm-tbot.urdf").getroot()
        for link in root.findall("link"):
            if not link.findall("visual/geometry/mesh"):
                continue
            body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, link.get("name"))
            mesh = visible & (scene.model.geom_bodyid == body)
            mesh &= scene.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
            assert mesh.any(), f"missing original visual mesh for {link.get('name')}"
        for name in ("piper_gripper_link1", "piper_gripper_link2"):
            assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        # Collision geometry remains available in the independent model but
        # is excluded from the rendered appearance.
        for geom in range(sim.model.ngeom):
            if geom != sim.spec.ground_geom:
                name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                visual_geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                assert visual_geom >= 0, name
                assert scene.model.geom_group[visual_geom] == 3, name
        assert not bool(scene.option.geomgroup[3])
    finally:
        scene.close()


@pytest.mark.private_assets
def test_visual_floor_is_a_visible_white_and_light_gray_grid(physical_sim):
    import mujoco

    from supermimic.wf.visuals import create_visual_scene

    sim = physical_sim
    scene = create_visual_scene(sim.model, sim.spec)
    try:
        ground = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "wf_ground")
        assert ground >= 0
        assert scene.model.geom_group[ground] == 0 and bool(scene.option.geomgroup[0])
        assert np.min(scene.model.geom_rgba[ground, :3]) >= .8
        material = int(scene.model.geom_matid[ground])
        assert material >= 0
        # The RGB role is what standard floor rendering samples; assigning
        # only a user texture would leave the rendered floor untextured.
        texture = int(scene.model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        assert texture >= 0
        start = int(scene.model.tex_adr[texture])
        size = int(scene.model.tex_width[texture]*scene.model.tex_height[texture])
        channels = int(scene.model.tex_nchannel[texture])
        pixels = scene.model.tex_data[start:start+size*channels].reshape(size, channels)[:, :3]
        assert len(np.unique(pixels, axis=0)) >= 2
        assert pixels.min() >= 180 and pixels.max() >= 245
        assert np.max(np.ptp(pixels.astype(float), axis=1)) <= 20
    finally:
        scene.close()


@pytest.mark.private_assets
def test_visual_scene_preserves_physics_hash_states_actions_and_contacts(physical_sim):
    from supermimic.wf.train import model_manifest
    from supermimic.wf.visuals import create_visual_scene

    sim = physical_sim
    initial = sim.snapshot()
    physics_hash = model_manifest(sim.spec)["physics_hash"]
    fields = ("body_mass", "body_inertia", "geom_type", "geom_size", "geom_pos", "geom_quat",
              "geom_contype", "geom_conaffinity", "geom_friction", "geom_rgba", "geom_group",
              "geom_margin", "geom_gap", "jnt_range", "actuator_ctrlrange")
    original = {field: getattr(sim.model, field).copy() for field in fields}
    scene = create_visual_scene(sim.model, sim.spec)
    try:
        for _ in range(2):
            scene.update(sim.data[0])
        assert_snapshot_equal(sim.snapshot(), initial)
        assert model_manifest(sim.spec)["physics_hash"] == physics_hash
        for field, value in original.items():
            np.testing.assert_array_equal(getattr(sim.model, field), value, err_msg=field)

        commands = []
        for step in range(8):
            action = sim.initial_action.copy()
            action[0] += .01*np.sin(np.arange(22)+step)
            commands.append(action)
        expected = []
        for action in commands:
            sim.step(action)
            expected.append((sim.snapshot(), sim.metrics()))
        sim.restore(initial)
        for action, (snapshot, metrics) in zip(commands, expected):
            scene.update(sim.data[0])
            sim.step(action)
            scene.update(sim.data[0])
            assert_snapshot_equal(sim.snapshot(), snapshot)
            for key, value in metrics.items():
                np.testing.assert_array_equal(sim.metrics()[key], value, err_msg=key)
        assert model_manifest(sim.spec)["physics_hash"] == physics_hash
        for field, value in original.items():
            np.testing.assert_array_equal(getattr(sim.model, field), value, err_msg=field)
    finally:
        scene.close()
