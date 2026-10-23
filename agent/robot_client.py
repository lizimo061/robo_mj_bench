"""Agent-side client. Standard library only; no simulator or private dependencies."""

import base64
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class RobotError(RuntimeError):
    pass


class RobotClient:
    def __init__(self, url=None, token=None):
        self.url = (url or os.environ["ROBOT_URL"]).rstrip("/")
        self._token = token or os.environ["ROBOT_TOKEN"]
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("invalid gateway URL")
        self._http = build_opener(ProxyHandler({}), _NoRedirects())

    def _call(self, action, payload):
        request = Request(self.url + "/v1/" + action, json.dumps(payload, allow_nan=False).encode(),
                          {"Authorization": "Bearer " + self._token, "Content-Type": "application/json"}, method="POST")
        try:
            with self._http.open(request, timeout=30) as response:
                result = json.load(response)
                if result.get("status") == "safety_stop":
                    raise RobotError("Robot safety stop ended the episode")
                return result
        except HTTPError as error:
            raise RobotError(f"Robot gateway returned HTTP {error.code}") from None

    def robot_state(self):
        return self._call("robot_state", {})

    def task(self):
        return self._call("task", {})

    def observe(self, camera_id="front"):
        return base64.b64decode(self._call("observe", {"camera_id": camera_id})["image_base64"], validate=True)

    def move_joints(self, targets, duration):
        return self._call("move_joints", {"targets": list(targets), "duration": duration})

    def set_gripper(self, opening, duration):
        return self._call("set_gripper", {"opening": opening, "duration": duration})

    def move_bimanual(self, left, right, duration):
        """Each arm command contains six targets and one gripper opening in meters."""
        return self._call("move_bimanual", {"left": left, "right": right, "duration": duration})

    def wait(self, duration):
        return self._call("wait", {"duration": duration})

    def submit(self, answer=None):
        return self._call("submit", {} if answer is None else {"answer": answer})
