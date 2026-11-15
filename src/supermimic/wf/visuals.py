"""Presentation model for policy videos, independent of the physics model.

Training deliberately uses simplified collision geometry. This module attaches
the URDF's outer meshes to a separate copy of the kinematic tree and follows
the evaluated simulation by joint name. Never step this presentation model.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .robot import RobotSpec, _rpy_matrix, build_model


def _rgba(material, global_materials):
    if material is None:
        return None
    color = material.find("color")
    if color is None and material.get("name") in global_materials:
        color = global_materials[material.get("name")].find("color")
    return np.fromstring(color.get("rgba"), sep=" ") if color is not None else None


def _mesh_file(path: Path) -> tuple[Path, np.ndarray | None]:
    """Use the supplied STL counterpart of a Collada visual without new deps.

    MuJoCo cannot read DAE. The portable assets include matching STL surfaces;
    keep the most common Collada diffuse color when URDF gives no override.
    """
    if path.suffix.lower() != ".dae":
        return path, None
    collada = ET.parse(path).getroot()
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    colors = Counter(color.text.strip() for color in
                     collada.findall(".//c:diffuse/c:color", namespace) if color.text)
    color = np.fromstring(colors.most_common(1)[0][0], sep=" ") if colors else None
    candidates = (path.with_suffix(".stl"), path.parent.parent / f"{path.stem}.stl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate, color
    raise FileNotFoundError(f"MuJoCo needs an STL counterpart for visual mesh: {path}")


def _add_visuals(mj, source):
    urdf = ET.parse(source).getroot()
    materials = {material.get("name"): material for material in urdf.findall("material")}
    meshes = {}
    for link in urdf.findall("link"):
        for index, visual in enumerate(link.findall("visual")):
            geometry = visual.find("geometry")
            if geometry is None or not len(geometry):
                continue
            shape = geometry[0]
            origin = visual.find("origin")
            pos = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, _rpy_matrix(rpy).reshape(-1))
            color = _rgba(visual.find("material"), materials)
            options = {}
            if shape.tag == "mesh":
                path = (source.parent / shape.get("filename")).resolve()
                scale = tuple(np.fromstring(shape.get("scale", "1 1 1"), sep=" "))
                key = path, scale
                if key not in meshes:
                    mesh_path, mesh_color = _mesh_file(path)
                    name = f"wf_outer_mesh_{len(meshes)}"
                    mj.add_mesh(name=name, file=str(mesh_path), scale=scale)
                    meshes[key] = name, mesh_color
                name, mesh_color = meshes[key]
                options = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
                if color is None:
                    color = mesh_color
            elif shape.tag == "box":
                options = dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                               size=np.fromstring(shape.get("size"), sep=" ") / 2)
            elif shape.tag == "sphere":
                options = dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(shape.get("radius")), 0, 0])
            elif shape.tag == "cylinder":
                options = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                               size=[float(shape.get("radius")), float(shape.get("length")) / 2, 0])
            else:
                raise ValueError(f"unsupported URDF visual geometry: {shape.tag}")
            mj.body(link.get("name")).add_geom(
                name=f"wf_outer_{link.get('name')}_{index}", pos=pos, quat=quat,
                rgba=color if color is not None else [.65, .65, .65, 1.],
                contype=0, conaffinity=0, mass=0, group=1, **options)


class RobotVisuals:
    """URDF outer surfaces and a white/gray checkerboard, for rendering only.

    Create with the *physics* model/spec, render ``self.model`` with
    ``scene_option=self.option``, and call ``sync(physics_data)`` before each
    frame. The input model, spec, and data are never modified.

    ``presentation`` adds studio lighting and a pale backdrop. Shadows and
    reflections also require the corresponding ``MjvScene.flags`` when
    rendering. ``msaa=None`` keeps MuJoCo's default antialiasing quality.
    """

    def __init__(self, physics_model: mujoco.MjModel, spec: RobotSpec, *,
                 presentation: bool = False, shadows: bool = False,
                 reflections: bool = False, msaa: int | None = None):
        if msaa is not None and msaa not in (0, 2, 4, 8):
            raise ValueError("MSAA samples must be one of 0, 2, 4, 8")
        _, _, mj = build_model(spec.asset_dir, timestep=physics_model.opt.timestep,
                               with_mj_spec=True)
        mj.compiler.discardvisual = False
        # Keep collision shapes available for debugging, but hide their group.
        for geom in mj.geoms:
            geom.group = 3
        ground = mj.geom("wf_ground")
        ground.group = 0
        ground.rgba = [1, 1, 1, 1]
        mj.add_texture(name="wf_video_checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                       builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       rgb1=[1, 1, 1], rgb2=[.90, .90, .90], width=256, height=256)
        textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
        textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "wf_video_checker"
        mj.add_material(name="wf_video_floor", textures=textures, texuniform=True,
                        texrepeat=[1, 1], specular=.10 if presentation else .05,
                        reflectance=.12 if reflections else 0,
                        # Brighten this surface without flattening robot shading.
                        emission=.18 if presentation or shadows else .36, rgba=[1, 1, 1, 1])
        ground.material = "wf_video_floor"
        _add_visuals(mj, spec.asset_dir / "urdf/wf-arm-tbot.urdf")
        if msaa is not None:
            mj.visual.quality.offsamples = msaa
        if shadows:
            mj.visual.quality.shadowsize = 4096
            # Keep the shadow-map boundary outside the main camera view; the
            # default tight volume can cause speckling on an infinite plane.
            mj.visual.map.shadowclip = 2.5
        if presentation:
            # A bright studio backdrop also prevents a black horizon when a
            # higher camera angle leaves some sky visible. These lights and
            # materials exist only in this rendering model.
            mj.add_texture(name="wf_video_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           rgb1=[.93, .95, .98], rgb2=[1, 1, 1],
                           width=512, height=3072)
        if presentation or shadows:
            mj.visual.headlight.ambient = [.22, .22, .22]
            mj.visual.headlight.diffuse = [.238, .238, .238]
            mj.visual.headlight.specular = [.12, .12, .12]
            mj.worldbody.add_light(name="wf_video_key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                   pos=[-2, -3, 4], dir=[2, 3, -4],
                                   diffuse=[.5525, .5355, .51], specular=[.30, .30, .30],
                                   ambient=[.04, .04, .04], castshadow=shadows)
            mj.worldbody.add_light(name="wf_video_fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                   pos=[2, 2, 3], dir=[-2, -2, -3],
                                   diffuse=[.2125, .238, .272], specular=[.12, .12, .12],
                                   ambient=[0, 0, 0], castshadow=False)
        self.model = mj.compile()
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[:2] = 1
        self.option.sitegroup[:] = 0
        self._follow_shadow_volume = presentation or shadows
        self._source_base_body = spec.base_body
        source_qpos, target_qpos = [], []
        for joint in range(physics_model.njnt):
            name = mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if target < 0 or self.model.jnt_type[target] != physics_model.jnt_type[joint]:
                raise ValueError(f"render joint differs from physical model: {name}")
            kind = physics_model.jnt_type[joint]
            width = 7 if kind == mujoco.mjtJoint.mjJNT_FREE else 4 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
            source_qpos.extend(range(physics_model.jnt_qposadr[joint], physics_model.jnt_qposadr[joint] + width))
            target_qpos.extend(range(self.model.jnt_qposadr[target], self.model.jnt_qposadr[target] + width))
        self._source_qpos = np.asarray(source_qpos)
        self._target_qpos = np.asarray(target_qpos)

    def sync(self, physics_data: mujoco.MjData) -> mujoco.MjData:
        """Update render transforms from a physical state without advancing it."""
        self.data.qpos[self._target_qpos] = physics_data.qpos[self._source_qpos]
        self.data.time = physics_data.time
        if self._follow_shadow_volume:
            self.model.stat.center[:] = physics_data.xpos[self._source_base_body] + [0, 0, .25]
        # Rendering needs body, geom, site, camera and light transforms, not a
        # dynamics solve or contact/constraint calculation in this second model.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        return self.data
