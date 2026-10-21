"""Allowlisted gateway and private audit trail for exactly one episode."""

import base64
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import time
from PIL import Image

from .protocol import BadRequest, SafetyStop, JOINT_LIMITS, SCHEMAS, validate_request


@dataclass(frozen=True)
class Limits:
    max_requests: int = 2000
    max_observations: int = 200
    max_commands: int = 500
    max_sim_seconds: float = 120.0
    max_wall_seconds: float = 1800.0


def _robot_projection(raw):
    """No arbitrary simulator dictionary is ever returned to the agent."""
    result = {}
    for key in ("joint_positions", "joint_velocities"):
        values = raw[key]
        if len(values) != len(JOINT_LIMITS) or not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError("invalid robot telemetry")
        result[key] = [float(v) for v in values]
    for key in ("gripper_opening", "sim_time"):
        result[key] = float(raw[key])
        if not math.isfinite(result[key]):
            raise RuntimeError("invalid robot telemetry")
    return result


class EpisodeService:
    """Call from a single owning thread (also required by the GL context)."""

    def __init__(self, simulation, token, audit_dir, limits=Limits()):
        if not isinstance(token, str) or len(token) < 16 or not token.isascii():
            raise ValueError("episode token must contain at least 16 ASCII characters")
        self.simulation = simulation
        self._expected_auth = ("Bearer " + token).encode("ascii")
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.limits = limits
        self.closed = False
        self._started = time.monotonic()
        self._requests = 0
        self._observations = 0
        self._commands = 0
        self._sim_seconds = 0.0
        # Exclusive creation prevents mixing/replacing an earlier episode's audit.
        self._audit = (self.audit_dir / "events.jsonl").open("x", encoding="utf-8")
        (self.audit_dir / "events.jsonl").chmod(0o600)

    def _record(self, event):
        self._audit.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        self._audit.flush()

    def _save_report(self, reason):
        prepare = getattr(self.simulation, "prepare_finish", None)
        if prepare is not None:
            prepare(reason)
        report = self.simulation.finish()
        if reason == "submitted" and report.get("safety_stopped", False):
            reason = "safety_stop"
        report["valid"] = reason in ("submitted", "safety_stop")
        if reason == "safety_stop":
            report["success"] = False
            report["score"] = 0
        elif reason != "submitted":
            report["success"] = False
            report["score"] = None
        report["termination"] = reason
        report["usage"] = {
            "requests": self._requests, "observations": self._observations,
            "commands": self._commands, "commanded_sim_seconds": self._sim_seconds,
            "wall_seconds": time.monotonic() - self._started,
        }
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(item) for item in value]
            return value

        # Never leave a truncated result that appears to be a completed report.
        encoded = json.dumps(finite_json(report), indent=2, allow_nan=False)
        staging = self.audit_dir / "result.json.tmp"
        with staging.open("w", encoding="utf-8") as stream:
            staging.chmod(0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(self.audit_dir / "result.json")
        return reason

    def authorized(self, authorization):
        if type(authorization) is not str or not authorization.isascii():
            return False
        return hmac.compare_digest(authorization.encode("ascii"), self._expected_auth)

    def handle(self, path, payload, authorization):
        if not self.authorized(authorization):
            return 401, {"error": "unauthorized"}
        if self.closed:
            return 409, {"error": "episode_closed"}
        if path not in SCHEMAS:
            return 404, {"error": "not_found"}
        # Submission is always available after other budgets have expired.
        self._requests += 1
        if path != "/v1/submit" and (
            self._requests > self.limits.max_requests
            or time.monotonic() - self._started > self.limits.max_wall_seconds
        ):
            return 429, {"error": "budget_exhausted"}
        try:
            request = validate_request(path, payload)
        except BadRequest:
            return 400, {"error": "invalid_request"}
        is_motion = path in ("/v1/move_joints", "/v1/set_gripper", "/v1/wait")
        if path == "/v1/observe" and self._observations >= self.limits.max_observations:
            return 429, {"error": "budget_exhausted"}
        if is_motion and (self._commands >= self.limits.max_commands
                          or self._sim_seconds + request["duration"] > self.limits.max_sim_seconds + 1e-9):
            return 429, {"error": "budget_exhausted"}
        try:
            self._record({"event": "request", "path": path, "payload": request})
            if path == "/v1/submit":
                if "answer" in request:
                    setter = getattr(self.simulation, "set_answer", None)
                    if setter is None:
                        raise BadRequest("Task does not accept an answer")
                    setter(request["answer"])
                self.closed = True
                termination = self._save_report("submitted")
                response = {"status": termination}
            elif path == "/v1/task":
                raw = self.simulation.public_spec()
                keys = ("task_id", "instruction", "approximation")
                if any(type(raw[key]) is not str or len(raw[key]) > 20_000 for key in keys):
                    raise RuntimeError("Invalid public task description")
                response = {key: raw[key] for key in keys}
                reference = raw.get("reference_image_png")
                if reference is not None:
                    if type(reference) is not bytes or len(reference) > 2_000_000 or not reference.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise RuntimeError("Invalid public task reference")
                    with Image.open(io.BytesIO(reference)) as source:
                        if source.width > 1024 or source.height > 1024:
                            raise RuntimeError("Public reference is too large")
                        clean = Image.new("RGB", source.size)
                        clean.paste(source.convert("RGB"))
                        encoded = io.BytesIO()
                        clean.save(encoded, format="PNG")
                    response["reference_image_base64"] = base64.b64encode(encoded.getvalue()).decode("ascii")
            elif path == "/v1/robot_state":
                response = _robot_projection(self.simulation.robot_state())
            elif path == "/v1/observe":
                self._observations += 1
                png = self.simulation.observe(request["camera_id"])
                digest = hashlib.sha256(png).hexdigest()
                frame = self.audit_dir / f"frame-{self._observations:04d}.png"
                frame.write_bytes(png)
                response = {"camera_id": request["camera_id"], "mime_type": "image/png",
                            "image_base64": base64.b64encode(png).decode("ascii")}
                self._record({"event": "frame", "file": frame.name, "sha256": digest})
            else:
                # Charge before execution, including commands that fail partway.
                self._commands += 1
                self._sim_seconds += request["duration"]
                if path == "/v1/move_joints":
                    self.simulation.move_joints(request["targets"], request["duration"])
                elif path == "/v1/set_gripper":
                    self.simulation.set_gripper(request["opening"], request["duration"])
                else:
                    self.simulation.wait(request["duration"])
                response = {"status": "ok", "sim_time": _robot_projection(self.simulation.robot_state())["sim_time"]}
            self._record({"event": "response", "path": path, "status": 200})
            return 200, response
        except SafetyStop:
            self.closed = True
            try:
                self._save_report("safety_stop")
                self._record({"event": "response", "path": path, "status": 200, "terminal": "safety_stop"})
                return 200, {"status": "safety_stop"}
            except Exception:
                return 500, {"error": "internal_error"}
        except BadRequest:
            self._record({"event": "response", "path": path, "status": 400})
            return 400, {"error": "invalid_request"}
        except Exception as error:
            self.closed = True
            # Exception text remains operator-only; never include it in HTTP errors.
            try:
                self._record({"event": "internal_error", "type": type(error).__name__, "detail": str(error)})
                if not (self.audit_dir / "result.json").exists():
                    self._save_report("internal_error")
            except Exception:
                pass  # Fail closed even when the private audit storage fails.
            return 500, {"error": "internal_error"}

    def close(self):
        try:
            if not self.closed:
                self.closed = True
                self._save_report("operator_stopped")
        finally:
            try:
                self._audit.close()
            finally:
                self.simulation.close()
