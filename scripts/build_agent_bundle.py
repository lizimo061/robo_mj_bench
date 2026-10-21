"""Create an allowlisted agent handoff. Never recursively archive the repository."""

from pathlib import Path
import zipfile
from robo_mj_bench.robot_assets import public_robot_files

root = Path(__file__).resolve().parents[1]
files = ("robot_client.py", "robot_kinematics.py", "INSTRUCTIONS.md")
output = root / "artifacts" / "agent-bundle.zip"
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(root / "agent" / name, arcname=name)
    for name, path in public_robot_files():
        archive.write(path, arcname=name)
print(output)
