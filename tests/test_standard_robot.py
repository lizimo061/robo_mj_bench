"""A benchmark task must preserve the pinned standard robot's physical model."""

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robo_mj_bench.protocol import JOINT_LIMITS
from robo_mj_bench.robot_assets import MODEL_DIR, MODEL_PATH, PROVENANCE_PATH
from robo_mj_bench.simulation import ButtonSimulation


PINNED_COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
MODEL_SHA256 = "beb1ccea8eefc3e181a562459fb55626e4beb7ef63eb0ceac55bdf8f24c63434"
LICENSE_SHA256 = "dc46c3a7dc48221aa783e634d15db0f816ba83ba5cdb77bbaa91525c61c7552f"


def test_standard_robot_inventory_has_pinned_revision_license_and_verified_bytes():
    manifest = json.loads(PROVENANCE_PATH.read_text())
    assert manifest["repository"] == "https://github.com/google-deepmind/mujoco_menagerie"
    assert manifest["commit"] == PINNED_COMMIT
    assert manifest["model"] == "yam_linear_4310.xml"
    assert manifest["license"] == "MIT"
    assert manifest["modifications"] == []
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256
    assert hashlib.sha256((MODEL_DIR / "LICENSE").read_bytes()).hexdigest() == LICENSE_SHA256
    names = [row["path"] for row in manifest["files"]]
    assert len(names) == len(set(names))
    for row in manifest["files"]:
        path = MODEL_DIR / row["path"]
        assert path.resolve().is_relative_to(MODEL_DIR.resolve())
        content = path.read_bytes()
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
        assert hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == row["git_blob_sha1"]
    root = ET.parse(MODEL_PATH).getroot()
    meshdir = root.find("compiler").get("meshdir")
    meshes = {f"{meshdir}/{mesh.get('file')}" for mesh in root.findall("./asset/mesh")}
    assert meshes <= set(names)
    assert all(name in {"LICENSE", "README.md", "CHANGELOG.md", "yam_linear_4310.xml"}
               or name.startswith("assets/") for name in names)


def test_task_preserves_entire_upstream_robot_subtree_defaults_and_actuators():
    upstream = ET.parse(MODEL_PATH).getroot()
    sim = ButtonSimulation(seed=19)
    try:
        task = ET.fromstring(sim.xml)
        for selector in ("./worldbody/body[@name='arm']", "default", "asset", "actuator", "equality"):
            assert ET.tostring(task.find(selector)) == ET.tostring(upstream.find(selector))
        for exclude in upstream.findall("./contact/exclude"):
            assert exclude.attrib in [item.attrib for item in task.findall("./contact/exclude")]
    finally:
        sim.close()


def test_compiled_robot_retains_six_dof_limits_actuators_and_inertias():
    upstream = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sim = ButtonSimulation(seed=19)
    try:
        task = sim.model
        assert len(JOINT_LIMITS) == 6
        assert upstream.nu == task.nu == 7
        for index in range(6):
            name = f"joint{index + 1}"
            source_joint, task_joint = upstream.joint(name).id, task.joint(name).id
            source_dof, task_dof = upstream.jnt_dofadr[source_joint], task.jnt_dofadr[task_joint]
            np.testing.assert_array_equal(JOINT_LIMITS[index], upstream.jnt_range[source_joint])
            for field in ("jnt_range", "jnt_axis", "jnt_type", "jnt_limited"):
                np.testing.assert_array_equal(getattr(task, field)[task_joint], getattr(upstream, field)[source_joint])
            for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
                np.testing.assert_array_equal(getattr(task, field)[task_dof], getattr(upstream, field)[source_dof])
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"):
            source_id, task_id = upstream.actuator(name).id, task.actuator(name).id
            for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm",
                          "actuator_ctrllimited", "actuator_forcelimited", "actuator_trntype"):
                np.testing.assert_array_equal(getattr(task, field)[task_id], getattr(upstream, field)[source_id])
            source_joint = upstream.joint(int(upstream.actuator_trnid[source_id, 0])).name
            task_joint = task.joint(int(task.actuator_trnid[task_id, 0])).name
            assert source_joint == task_joint
        for source_id in range(1, upstream.nbody):
            name = upstream.body(source_id).name
            task_id = task.body(name).id
            for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_gravcomp"):
                np.testing.assert_array_equal(getattr(task, field)[task_id], getattr(upstream, field)[source_id])
            source_geoms = np.flatnonzero(upstream.geom_bodyid == source_id)
            task_geoms = np.flatnonzero(task.geom_bodyid == task_id)
            assert len(source_geoms) == len(task_geoms)
            for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
                          "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity", "geom_group"):
                np.testing.assert_array_equal(getattr(task, field)[task_geoms], getattr(upstream, field)[source_geoms])
    finally:
        sim.close()
