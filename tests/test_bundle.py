"""The handoff archive must remain a narrow public-client package."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import mujoco

from robo_mj_bench.robot_assets import MODEL_DIR


def test_agent_handoff_contains_only_public_client_and_instructions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "agent").mkdir()
    (tmp_path / "src").mkdir()
    shutil.copy(root / "scripts" / "build_agent_bundle.py", tmp_path / "scripts")
    public_agent = ("robot_client.py", "robot_kinematics.py", "INSTRUCTIONS.md")
    for name in public_agent:
        shutil.copy(root / "agent" / name, tmp_path / "agent")
    # Neither accidental files in agent/ nor trusted siblings may enter the zip.
    for name in ("private-scene.xml", "vision_demo.py", "result.json", "agent-token"):
        (tmp_path / "agent" / name).write_text("private episode ground truth")
    (tmp_path / "src" / "simulation.py").write_text("private seed and evaluator")
    package = tmp_path / "src" / "robo_mj_bench"
    package.mkdir()
    shutil.copy(root / "src" / "robo_mj_bench" / "__init__.py", package)
    shutil.copy(root / "src" / "robo_mj_bench" / "robot_assets.py", package)
    model = package / "models" / "i2rt_yam"
    shutil.copytree(MODEL_DIR, model)
    (model / "private-scene.xml").write_text("private seed and task layout")
    (model / "assets" / "extra-private.stl").write_text("not in immutable manifest")
    subprocess.run([sys.executable, str(tmp_path / "scripts" / "build_agent_bundle.py")],
                   check=True, capture_output=True, text=True, timeout=10,
                   env={**os.environ, "PYTHONPATH": str(tmp_path / "src")})
    manifest = json.loads((MODEL_DIR / "PROVENANCE.json").read_text())
    expected_robot = {"robot/" + row["path"] for row in manifest["files"]} | {"robot/PROVENANCE.json"}
    with zipfile.ZipFile(tmp_path / "artifacts" / "agent-bundle.zip") as archive:
        assert set(archive.namelist()) == set(public_agent) | expected_robot
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name in archive.namelist():
            source = MODEL_DIR / name.removeprefix("robot/") if name.startswith("robot/") else root / "agent" / name
            assert archive.read(name) == source.read_bytes()
        source = archive.read("robot_client.py").decode()
        exported = tmp_path / "exported"
        archive.extractall(exported)
    robot = mujoco.MjModel.from_xml_path(str(exported / "robot" / "yam_linear_4310.xml"))
    assert robot.nq == 8
    assert robot.nu == 7
    assert all(robot.joint(index).name != "button_travel" for index in range(robot.njnt))
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.add(node.module.split(".")[0])
    assert imports <= sys.stdlib_module_names
    assert not imports.intersection({"robo_mj_bench", "mujoco", "numpy"})
