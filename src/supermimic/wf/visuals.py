"""Original URDF surface geometry in an independent, display-only scene.

The policy simulator retains its collision geometry, contacts and parameters.
Piper COLLADA triangles and diffuse colors are decoded directly into MuJoCo
visual meshes; no asset conversion or extra graphics dependency is needed.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .robot import _rpy_matrix, build_model


def _numbers(text, default=""):
    return np.fromstring(text or default, sep=" ")


def _node_transform(node):
    transform = np.eye(4)
    for child in node:
        value = _numbers(child.text)
        local = np.eye(4)
        if child.tag == "matrix":
            local = value.reshape(4, 4)
        elif child.tag == "translate":
            local[:3, 3] = value
        elif child.tag == "scale":
            local[:3, :3] = np.diag(value)
        elif child.tag == "rotate":
            axis = value[:3] / np.linalg.norm(value[:3])
            x, y, z = axis
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            angle = math.radians(value[3])
            local[:3, :3] = np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)
        else:
            continue
        transform = transform @ local
    return transform


def _dae_meshes(path):
    """Read the triangulated, untextured COLLADA surfaces shipped with Piper."""
    root = ET.parse(path).getroot()
    for item in root.iter():
        item.tag = item.tag.rsplit("}", 1)[-1]
    if root.findall(".//image"):
        raise ValueError(f"textured COLLADA visuals are unsupported: {path}")
    effects = {}
    for effect in root.findall(".//library_effects/effect"):
        color = effect.find(".//diffuse/color")
        effects[effect.get("id")] = tuple(_numbers(color.text) if color is not None else [.7, .7, .7, 1.])
    materials = {item.get("id"): effects[item.find("instance_effect").get("url").lstrip("#")]
                 for item in root.findall(".//library_materials/material")}
    geometries = {item.get("id"): item.find("mesh") for item in root.findall(".//library_geometries/geometry")}
    groups = defaultdict(list)

    def visit(node, parent):
        transform = parent @ _node_transform(node)
        for instance in node.findall("instance_geometry"):
            mesh = geometries[instance.get("url").lstrip("#")]
            if any(child.tag in ("polylist", "polygons", "trifans", "tristrips") for child in mesh):
                raise ValueError(f"COLLADA visual must contain triangles: {path}")
            sources = {}
            for source in mesh.findall("source"):
                accessor = source.find("technique_common/accessor")
                stride = int(accessor.get("stride", "1"))
                sources[source.get("id")] = _numbers(source.find("float_array").text).reshape(-1, stride)
            vertices = {item.get("id"): item.find("input").get("source").lstrip("#")
                        for item in mesh.findall("vertices")}
            bindings = {item.get("symbol"): item.get("target").lstrip("#")
                        for item in instance.findall(".//instance_material")}
            for triangles in mesh.findall("triangles"):
                inputs = triangles.findall("input")
                stride = max(int(item.get("offset", "0")) for item in inputs) + 1
                indices = np.fromstring(triangles.find("p").text, sep=" ", dtype=np.int32).reshape(-1, stride)
                vertex_input = next(item for item in inputs if item.get("semantic") == "VERTEX")
                positions = sources[vertices[vertex_input.get("source").lstrip("#")]][:, :3]
                positions = positions @ transform[:3, :3].T + transform[:3, 3]
                faces = indices[:, int(vertex_input.get("offset", "0"))].reshape(-1, 3)
                normal_input = next((item for item in inputs if item.get("semantic") == "NORMAL"), None)
                normals, normal_faces = None, None
                if normal_input is not None:
                    normals = sources[normal_input.get("source").lstrip("#")][:, :3] @ np.linalg.inv(transform[:3, :3])
                    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-12)
                    normal_faces = indices[:, int(normal_input.get("offset", "0"))].reshape(-1, 3)
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, ::-1]
                    if normal_faces is not None:
                        normal_faces = normal_faces[:, ::-1]
                material = bindings.get(triangles.get("material"), triangles.get("material"))
                groups[materials.get(material, (.7, .7, .7, 1.))].append((positions, faces, normals, normal_faces))
        for child in node.findall("node"):
            visit(child, transform)

    scene_id = root.find("scene/instance_visual_scene").get("url").lstrip("#")
    scene = next(item for item in root.findall("library_visual_scenes/visual_scene") if item.get("id") == scene_id)
    unit = root.find("asset/unit")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.
    up_axis = root.findtext("asset/up_axis", "Z_UP").strip()
    if up_axis != "Z_UP":
        raise ValueError(f"expected Z_UP COLLADA robot visual: {path}")
    transform = np.diag([scale, scale, scale, 1.])
    for node in scene.findall("node"):
        visit(node, transform)
    result = []
    for color, parts in groups.items():
        positions, faces, normals, normal_faces = [], [], [], []
        vertex_offset, normal_offset = 0, 0
        use_normals = all(part[2] is not None for part in parts)
        for vertex, face, normal, normal_face in parts:
            positions.append(vertex)
            faces.append(face + vertex_offset)
            vertex_offset += len(vertex)
            if use_normals:
                normals.append(normal)
                normal_faces.append(normal_face + normal_offset)
                normal_offset += len(normal)
        result.append(dict(rgba=color, uservert=np.concatenate(positions).ravel().tolist(),
            userface=np.concatenate(faces).ravel().tolist(),
            **(dict(usernormal=np.concatenate(normals).ravel().tolist(),
                    userfacenormal=np.concatenate(normal_faces).ravel().tolist()) if use_normals else {})))
    if not result:
        raise ValueError(f"COLLADA visual has no triangle surfaces: {path}")
    return result


def _add_urdf_visuals(render_spec, asset_dir):
    source = Path(asset_dir) / "urdf/wf-arm-tbot.urdf"
    root = ET.parse(source).getroot()
    named_materials = {item.get("name"): _numbers(item.find("color").get("rgba"))
                       for item in root.findall("material") if item.find("color") is not None}
    mesh_cache = {}
    count = 0
    for link in root.findall("link"):
        body = render_spec.body(link.get("name"))
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            pos = _numbers(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
            rpy = _numbers(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, _rpy_matrix(rpy).ravel())
            material = visual.find("material")
            color = [.7, .7, .7, 1.]
            if material is not None:
                color = named_materials.get(material.get("name"), color)
                if material.find("color") is not None:
                    color = _numbers(material.find("color").get("rgba"))
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            pieces = []
            if mesh is not None:
                path = (source.parent / mesh.get("filename")).resolve()
                scale = tuple(_numbers(mesh.get("scale", "1 1 1")))
                cache_key = (str(path), scale)
                if cache_key not in mesh_cache:
                    definitions = (_dae_meshes(path) if path.suffix.lower() == ".dae" else [dict(file=str(path))])
                    loaded = []
                    for definition in definitions:
                        rgba = definition.pop("rgba", None)
                        name = f"wf_surface_mesh_{len(render_spec.meshes)}"
                        render_spec.add_mesh(name=name, scale=scale,
                            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL, **definition)
                        loaded.append((name, rgba))
                    mesh_cache[cache_key] = loaded
                pieces = [dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                               rgba=rgba if rgba is not None else color) for name, rgba in mesh_cache[cache_key]]
            elif geometry.find("box") is not None:
                pieces = [dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                               size=_numbers(geometry.find("box").get("size")) / 2, rgba=color)]
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                pieces = [dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                               size=[float(cylinder.get("radius")), float(cylinder.get("length"))/2, 0], rgba=color)]
            elif geometry.find("sphere") is not None:
                pieces = [dict(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                               size=[float(geometry.find("sphere").get("radius")), 0, 0], rgba=color)]
            else:
                raise ValueError(f"unsupported visual geometry for {link.get('name')}")
            for piece in pieces:
                body.add_geom(name=f"wf_surface_{count}", pos=pos, quat=quaternion,
                    contype=0, conaffinity=0, density=0, group=2, **piece)
                count += 1
    return count


class VisualScene:
    """A render model/data pair synchronized by copying physical joint poses."""

    def __init__(self, physical_model, spec):
        _, _, render_spec = build_model(spec.asset_dir, physical_model.opt.timestep,
            leg_action_scale=float(spec.action_scale[0]), arm_action_scale=None,
            wheel_action_scale=float(spec.action_scale[12]), with_mj_spec=True)
        render_spec.compiler.discardvisual = False
        for geom in render_spec.geoms:
            geom.group = 3
        self.visual_geom_count = _add_urdf_visuals(render_spec, spec.asset_dir)
        render_spec.add_texture(name="wf_white_background", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1., 1., 1.],
            rgb2=[1., 1., 1.], width=256, height=256)
        render_spec.add_texture(name="wf_bright_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[1., 1., 1.], rgb2=[.9, .9, .9],
            width=512, height=512)
        render_spec.add_material(name="wf_bright_floor", textures=["", "wf_bright_floor"],
            texrepeat=[1., 1.], texuniform=True, rgba=[1., 1., 1., 1.],
            specular=.1, shininess=.1, reflectance=0., emission=.4)
        floor = render_spec.geom("wf_ground")
        floor.group, floor.material, floor.rgba = 0, "wf_bright_floor", [1., 1., 1., 1.]
        self.model = render_spec.compile()
        if (self.model.nq, self.model.nv, self.model.nbody) != (physical_model.nq, physical_model.nv, physical_model.nbody):
            raise ValueError("visual scene must retain the physical model's bodies and joint coordinates")
        for joint_id in range(physical_model.njnt):
            if (self.model.joint(joint_id).name != physical_model.joint(joint_id).name
                    or self.model.jnt_qposadr[joint_id] != physical_model.jnt_qposadr[joint_id]):
                raise ValueError("visual scene joint order differs from the physical simulator")
        for body_id in range(physical_model.nbody):
            if self.model.body(body_id).name != physical_model.body(body_id).name:
                raise ValueError("visual scene body order differs from the physical simulator")
        self.model.vis.headlight.ambient[:] = [.4, .4, .4]
        self.model.vis.headlight.diffuse[:] = [.5, .5, .5]
        self.model.vis.headlight.specular[:] = [.15, .15, .15]
        self.model.vis.rgba.fog[:] = [1., 1., 1., 1.]
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.option.sitegroup[:] = 0

    def update(self, physical_data):
        """Update only render state; never call dynamics on the physical scene."""
        self.data.qpos[:] = physical_data.qpos
        self.data.qvel[:] = physical_data.qvel
        self.data.time = physical_data.time
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def close(self):
        pass


def create_visual_scene(model, spec):
    return VisualScene(model, spec)
