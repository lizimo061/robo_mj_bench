"""Fetch a pinned, byte-verified Menagerie YAM robot and referenced public meshes."""

from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/robo_mj_bench/models/i2rt_yam"
REPO = "google-deepmind/mujoco_menagerie"
PREFIX = "i2rt_yam/"
REVISION = "cda5236f98383f6fbaa4820c6b18dba892cd087f"


def fetch(url):
    with tempfile.NamedTemporaryFile() as stream:
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--retry", "6", "--retry-all-errors", "--max-time", "120",
                        url, "--output", stream.name], check=True)
        return Path(stream.name).read_bytes()


def main():
    commit = REVISION
    print(f"Source revision: {commit}", flush=True)
    listing = json.loads(fetch(f"https://api.github.com/repos/{REPO}/contents/i2rt_yam?ref={commit}"))
    entries = {item["name"]: item for item in listing if item["type"] == "file"}
    asset_sha = next(item["sha"] for item in listing if item["name"] == "assets")
    assets = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{asset_sha}?recursive=1"))
    entries.update({"assets/" + item["path"]: item for item in assets["tree"] if item["type"] == "blob"})

    def download(name):
        item = entries[name]
        blob = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/blobs/{item['sha']}"))
        content = base64.b64decode(blob["content"])
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest != item["sha"]:
            raise RuntimeError(f"Upstream blob verification failed: {name}")
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f"Verified {name} ({len(content)} bytes)", flush=True)
        return {"path": name, "bytes": len(content), "git_blob_sha1": digest,
                "sha256": hashlib.sha256(content).hexdigest()}

    DEST.mkdir(parents=True, exist_ok=True)
    model_name = "yam_linear_4310.xml"
    files = [download(model_name)]
    model = ET.parse(DEST / model_name).getroot()
    meshdir = model.find("compiler").get("meshdir", "")
    names = {f"{meshdir}/{mesh.get('file')}" for mesh in model.findall("./asset/mesh")}
    names.update({"LICENSE", "README.md", "CHANGELOG.md"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        files.extend(pool.map(download, sorted(names)))
    provenance = {"repository": f"https://github.com/{REPO}", "commit": commit,
                  "model": model_name, "license": "MIT", "modifications": [],
                  "files": sorted(files, key=lambda row: row["path"])}
    (DEST / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Vendored {len(files)} unchanged files from {commit}", flush=True)


if __name__ == "__main__":
    main()
