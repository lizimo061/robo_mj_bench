"""Pinned public robot data, separate from all private episode geometry."""
import hashlib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models" / "i2rt_yam"
MODEL_PATH = MODEL_DIR / "yam_linear_4310.xml"
PROVENANCE_PATH = MODEL_DIR / "PROVENANCE.json"
ROBOT_ID = "i2rt_yam_linear_4310"


def provenance():
    return json.loads(PROVENANCE_PATH.read_text())


def model_xml():
    row = next(row for row in provenance()["files"] if row["path"] == MODEL_PATH.name)
    content = MODEL_PATH.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise RuntimeError("Robot model checksum mismatch")
    return content.decode("utf-8")


def model_assets():
    assets = {}
    for row in provenance()["files"]:
        if row["path"].startswith("assets/"):
            content = (MODEL_DIR / row["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise RuntimeError("Robot mesh checksum mismatch")
            assets[row["path"]] = content
    return assets


def public_robot_files():
    """Only immutable robot-model inventory, never any composed task scene."""
    for row in provenance()["files"]:
        name = row["path"]
        if not (name in ("yam_linear_4310.xml", "LICENSE", "README.md", "CHANGELOG.md")
                or (name.startswith("assets/") and name.endswith(".stl"))):
            raise RuntimeError("Unexpected public robot file")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Invalid public robot path")
        source = MODEL_DIR / path
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Public robot checksum mismatch")
        yield "robot/" + name, source
    yield "robot/PROVENANCE.json", PROVENANCE_PATH
