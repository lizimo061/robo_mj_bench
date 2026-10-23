#!/usr/bin/env python3
"""Check stored ChatGPT CLI inference without opening any credential files.

This is a transport preflight, never a robotics attempt or a policy runner.
Only an allowlisted summary is written; raw CLI output is not persisted.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def probe(model, timeout):
    env = os.environ.copy()
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL",
                 "CODEX_ACCESS_TOKEN", "OPENAI_IDENTITY_TOKEN_FILE"):
        env.pop(name, None)
    result = {"requested_model": model, "verified_model": None,
              "reasoning_effort": "low", "chatgpt_login": False,
              "timeout_seconds": timeout,
              "robotics_attempt": False, "usage": None}
    try:
        login = subprocess.run(["codex", "login", "status"], env=env,
                               capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return dict(result, status="chatgpt_login_unavailable")
    subscription = login.returncode == 0 and "Logged in using ChatGPT" in (
        login.stdout + login.stderr)
    result["chatgpt_login"] = subscription
    if not subscription:
        return dict(result, status="chatgpt_login_unavailable")
    args = ["codex", "exec", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--ephemeral", "--json",
            "-s", "read-only", "-m", model]
    config = {"approval_policy": "never", "forced_login_method": "chatgpt",
              "web_search": "disabled", "model_reasoning_effort": "low"}
    for feature in ("shell_tool", "apps", "plugins", "memories", "multi_agent",
                    "browser_use", "computer_use", "image_generation", "hooks",
                    "view_image", "shell_snapshot", "skill_search"):
        config[f"features.{feature}"] = False
    for key, value in config.items():
        args += ["-c", key + "=" + json.dumps(value)]
    args += ["Return exactly READY. Do not use tools."]
    started = time.monotonic()
    # Keep input empty and cwd unrelated to the benchmark. Authentication is
    # consumed by the CLI itself; this program never reads/copies its secrets.
    with tempfile.TemporaryDirectory(prefix="robot-inference-preflight-") as cwd:
        process = subprocess.Popen(args, cwd=cwd, env=env,
                                   stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
    result.update(elapsed_seconds=round(time.monotonic() - started, 3),
                  process_exit_code=process.returncode, deadline_expired=timed_out)
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    result["event_types"] = [e.get("type") for e in events]
    messages = [e.get("item", {}).get("text", "") for e in events
                if e.get("type") == "item.completed"
                and e.get("item", {}).get("type") == "agent_message"]
    result["ready_received"] = any(m.strip() == "READY" for m in messages)
    result["turn_completed"] = any(e.get("type") == "turn.completed" for e in events)
    result["tool_item_seen"] = any(
        e.get("item", {}).get("type") in
        ("command_execution", "mcp_tool_call", "web_search", "file_change")
        for e in events)
    for event in events:
        if event.get("type") == "turn.completed":
            result["usage"] = event.get("usage")
    combined = (stdout + stderr).lower()
    # Only categorical errors are exported; arbitrary error strings may include
    # private filesystem locations or server details.
    result["error_categories"] = [label for needle, label in (
        ("request timed out", "inference_request_timeout"),
        ("operation not permitted", "local_permission_error"),
        ("unauthorized", "authentication_error"),
        ("error loading config.toml", "configuration_error"),
        ("not supported", "unsupported_configuration_or_model"),
        ("reconnecting", "transport_retry")) if needle in combined]
    result["status"] = ("passed" if result["ready_received"] and
                        result["turn_completed"] and not timed_out and
                        process.returncode == 0 and not result["tool_item_seen"]
                        else "infrastructure_blocked")
    # A successful READY does not attest model identity or tool removal. Those
    # require separate session metadata before starting a robotics benchmark.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = probe(args.model, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
