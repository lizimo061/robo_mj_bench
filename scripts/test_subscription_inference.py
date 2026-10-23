"""Transport preflight tests; does not import benchmark/evaluator code."""

import json
import os
import subprocess
import unittest
from unittest.mock import Mock, patch

from check_subscription_inference import probe


def events(*items):
    return "\n".join(json.dumps(item) for item in items)


READY = {"type": "item.completed", "item": {"type": "agent_message", "text": "READY"}}
COMPLETED = {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 1}}


class PreflightTests(unittest.TestCase):
    def run_probe(self, output, code=0, timeout=False):
        login = Mock(returncode=0, stdout="Logged in using ChatGPT", stderr="")
        process = Mock(returncode=code, pid=12345)
        process.communicate.side_effect = (
            [subprocess.TimeoutExpired("codex", 1), (output, "request timed out")]
            if timeout else [(output, "")])
        with patch("check_subscription_inference.subprocess.run", return_value=login), \
                patch("check_subscription_inference.subprocess.Popen", return_value=process) as popen, \
                patch("check_subscription_inference.os.killpg"):
            result = probe("gpt-6.1-sol", 1)
        return result, popen

    def test_ready_requires_completed_turn(self):
        result, _ = self.run_probe(events(READY))
        self.assertEqual(result["status"], "infrastructure_blocked")

    def test_completed_ready_passes_without_attesting_model(self):
        result, _ = self.run_probe(events(READY, COMPLETED))
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(result["verified_model"])
        self.assertEqual(result["usage"]["output_tokens"], 1)

    def test_deadline_with_zero_exit_is_still_failure(self):
        result, _ = self.run_probe(events(READY, COMPLETED), timeout=True)
        self.assertEqual(result["process_exit_code"], 0)
        self.assertTrue(result["deadline_expired"])
        self.assertEqual(result["status"], "infrastructure_blocked")

    def test_tool_execution_rejects_preflight(self):
        tool = {"type": "item.completed", "item": {"type": "command_execution"}}
        result, _ = self.run_probe(events(tool, READY, COMPLETED))
        self.assertEqual(result["status"], "infrastructure_blocked")

    def test_api_login_never_starts_inference(self):
        login = Mock(returncode=0, stdout="Logged in using an API key", stderr="")
        with patch("check_subscription_inference.subprocess.run", return_value=login), \
                patch("check_subscription_inference.subprocess.Popen") as popen:
            result = probe("gpt-6.1-sol", 1)
        popen.assert_not_called()
        self.assertEqual(result["status"], "chatgpt_login_unavailable")

    def test_login_timeout_is_reported(self):
        with patch("check_subscription_inference.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("codex", 15)):
            result = probe("gpt-6.1-sol", 1)
        self.assertEqual(result["status"], "chatgpt_login_unavailable")

    def test_paid_credentials_removed_and_subscription_forced(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret",
                                     "CODEX_API_KEY": "test-secret"}):
            _, popen = self.run_probe(events(READY, COMPLETED))
        kwargs = popen.call_args.kwargs
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("CODEX_API_KEY", kwargs["env"])
        args = popen.call_args.args[0]
        self.assertIn('forced_login_method="chatgpt"', args)
        self.assertIn('features.shell_tool=false', args)
        self.assertIn('--ignore-user-config', args)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
