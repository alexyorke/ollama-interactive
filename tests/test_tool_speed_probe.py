from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import tool_speed_probe as probe


class ToolSpeedProbeTests(unittest.TestCase):
    def test_measure_preserves_explicit_zero_count(self) -> None:
        row = probe._measure(
            "file_search",
            lambda: {
                "ok": True,
                "count": 0,
                "output": "no matches found\n",
            },
        )

        self.assertEqual(row["count"], 0)
        self.assertTrue(row["ok"])

    def test_main_accepts_argv_and_writes_output(self) -> None:
        payload = {"generated_files": 3, "rows": [{"name": "file_search", "elapsed_ms": 1.23, "ok": True, "count": 1}]}

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "tool-speed.json"
            with patch.object(probe, "run_probe", return_value=payload) as run_probe:
                exit_code = probe.main(["--generated-files", "3", "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            run_probe.assert_called_once_with(3)
            self.assertTrue(output_path.exists())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(written, payload)


if __name__ == "__main__":
    unittest.main()
