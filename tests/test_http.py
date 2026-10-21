"""Transport parsing is part of the agent/trusted-service boundary."""

import http.client
from http.server import HTTPServer
import json
import socket
import threading

import pytest

from robo_mj_bench.server import MAX_BODY_BYTES, handler_for
from robo_mj_bench.service import EpisodeService
from test_service import AUTH, TOKEN, FakeSimulation


@pytest.fixture
def endpoint(tmp_path):
    service = EpisodeService(FakeSimulation(), TOKEN, tmp_path)
    server = HTTPServer(("127.0.0.1", 0), handler_for(service))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server.server_port
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    service.close()


def request(port, body=b"{}", *, path="/v1/robot_state", method="POST", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(method, path, body=body, headers=headers if headers is not None else {
            "Authorization": AUTH, "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_http_robot_state_has_no_cache_or_framework_disclosure(endpoint):
    status, headers, body = request(endpoint)
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert headers["Content-Type"] == "application/json"
    assert "Python" not in headers["Server"]
    assert set(json.loads(body)) == {"joint_positions", "joint_velocities", "gripper_opening", "sim_time"}


@pytest.mark.parametrize("body", [
    b'{"duration":NaN}', b'{"duration":Infinity}', b'{"duration":-Infinity}',
    b'{"duration":0.1,"duration":0.2}', b'{', b'\xff', b'[]',
])
def test_malformed_or_ambiguous_json_is_rejected(endpoint, body):
    status, _, response = request(endpoint, body, path="/v1/wait")
    assert status == 400
    assert b"Traceback" not in response


def test_oversize_body_rejected(endpoint):
    assert request(endpoint, b" " * (MAX_BODY_BYTES + 1))[0] == 413


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "OPTIONS", "PATCH"])
def test_non_post_methods_rejected(endpoint, method):
    status, _, _ = request(endpoint, method=method)
    assert status in (405, 501)


def test_authentication_required_before_parsing(endpoint):
    status, _, body = request(endpoint, b"not-json", headers={"Content-Type": "application/json"})
    assert status == 401
    assert json.loads(body) == {"error": "unauthorized"}


@pytest.mark.parametrize("extra_header", [
    "Authorization: Bearer duplicate\r\n",
    "Content-Length: 2\r\n",
    "Transfer-Encoding: chunked\r\n",
])
def test_duplicate_or_ambiguous_http_headers_rejected(endpoint, extra_header):
    wire = (f"POST /v1/robot_state HTTP/1.0\r\nAuthorization: {AUTH}\r\n"
            f"Content-Type: application/json\r\nContent-Length: 2\r\n{extra_header}\r\n{{}}")
    with socket.create_connection(("127.0.0.1", endpoint), timeout=2) as connection:
        connection.sendall(wire.encode("ascii"))
        response = connection.recv(4096)
    assert response.startswith((b"HTTP/1.0 400 ", b"HTTP/1.0 401 "))


def test_query_parameter_cannot_expand_api(endpoint):
    assert request(endpoint, path="/v1/robot_state?include_objects=true")[0] == 404


def test_agent_client_treats_safety_receipt_as_terminal(endpoint, monkeypatch):
    from robo_mj_bench.protocol import SafetyStop
    from agent.robot_client import RobotClient, RobotError

    def stop(self, duration):
        raise SafetyStop()

    monkeypatch.setattr(FakeSimulation, "wait", stop)
    client = RobotClient(url=f"http://127.0.0.1:{endpoint}", token=TOKEN)
    with pytest.raises(RobotError, match="safety"):
        client.wait(0.1)
    with pytest.raises(RobotError, match="409"):
        client.robot_state()
