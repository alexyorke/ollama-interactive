from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.error import URLError

from scripts import ollama_perf_probe as probe


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class OllamaPerfProbeTests(unittest.TestCase):
    def test_chat_tolerates_non_dict_message_payload(self) -> None:
        payload = {
            "prompt_eval_count": 12,
            "eval_count": 6,
            "prompt_eval_duration": 1_000_000_000,
            "eval_duration": 2_000_000_000,
            "load_duration": 500_000_000,
            "total_duration": 3_500_000_000,
            "done_reason": "stop",
            "message": "not-a-dict",
        }
        with patch("scripts.ollama_perf_probe.urllib.request.urlopen", return_value=_FakeResponse(payload)):
            row = probe._chat("http://127.0.0.1:11434", "demo-model", "hello", options={"num_predict": 8})

        self.assertEqual(row["prompt_tokens"], 12)
        self.assertEqual(row["output_tokens"], 6)
        self.assertEqual(row["done_reason"], "stop")
        self.assertEqual(row["preview"], "")

    def test_main_writes_error_payload_when_chat_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "perf.json"
            with patch.object(probe, "_chat", side_effect=URLError("connection refused")), patch.object(
                probe, "_ollama_ps", return_value=""
            ), patch.object(probe, "_stop_model"):
                exit_code = probe.main(["--models", "demo-model", "--output", str(output_path)])

            self.assertEqual(exit_code, 1)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["host"], "http://127.0.0.1:11434")
        self.assertEqual(len(payload["results"]), 3)
        for row in payload["results"]:
            self.assertEqual(row["model"], "demo-model")
            self.assertFalse(row["ok"])
            self.assertIn("connection refused", row["error"])

    def test_main_writes_error_payload_when_ollama_cli_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "perf.json"
            with patch("subprocess.run", side_effect=FileNotFoundError("ollama not found")), patch.object(
                probe, "_chat", side_effect=URLError("connection refused")
            ):
                exit_code = probe.main(["--models", "demo-model", "--output", str(output_path)])

            self.assertEqual(exit_code, 1)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["results"]), 3)
        for row in payload["results"]:
            self.assertFalse(row["ok"])
            self.assertIn("connection refused", row["error"])
            self.assertIn("ollama not found", row["ollama_ps_after"])


if __name__ == "__main__":
    unittest.main()
