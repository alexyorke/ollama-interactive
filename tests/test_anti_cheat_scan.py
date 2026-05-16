from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import anti_cheat_scan


class AntiCheatScanTests(unittest.TestCase):
    def test_current_runtime_and_prompts_have_no_benchmark_cheats(self) -> None:
        result = anti_cheat_scan.scan()

        self.assertEqual(result["findings"], [])

    def test_runtime_scan_flags_public_task_and_marker_literals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "ollama_code"
            package.mkdir()
            (package / "agent.py").write_text("if task == 'scale-generator': return 'BENCH_EXACT'\n", encoding="utf-8")

            findings = anti_cheat_scan.scan_runtime(root)

        self.assertEqual(len(findings), 1)
        self.assertIn("hard-coded public smoke task", findings[0]["findings"])
        self.assertIn("synthetic marker token", findings[0]["findings"])

    def test_runtime_scan_flags_local_benchmark_case_switches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "ollama_code"
            package.mkdir()
            (package / "agent.py").write_text("case = 'issue_fix_hidden_tests'\n", encoding="utf-8")

            findings = anti_cheat_scan.scan_runtime(root)

        self.assertEqual(len(findings), 1)
        self.assertIn("local benchmark case switch", findings[0]["findings"])

    def test_benchmark_prompt_scan_flags_public_task_leaks(self) -> None:
        with patch.object(anti_cheat_scan.public_bench, "public_task_prompt", return_value="Solve list-ops with foldr."):
            findings = anti_cheat_scan.scan_benchmark_prompts()

        public_findings = [item for item in findings if item.get("scope") == "public_prompt"]
        self.assertEqual(len(public_findings), 1)
        self.assertIn("public smoke task", public_findings[0]["findings"])
        self.assertIn("task-specific solution hint", public_findings[0]["findings"])

    def test_benchmark_prompt_scan_flags_coding_accuracy_leaks(self) -> None:
        case = anti_cheat_scan.coding_bench.BenchmarkCase(
            name="leaky",
            suite="local-small",
            turns=("Use read_file and reply with BENCH_123.",),
            validate=lambda ctx: "pass",
        )
        with patch.object(anti_cheat_scan.coding_bench, "selected_cases", return_value=[case]):
            findings = anti_cheat_scan.scan_benchmark_prompts()

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["scope"], "coding_prompt")
        self.assertIn("synthetic marker token", findings[0]["findings"])

    def test_benchmark_prompt_scan_flags_expanded_task_names(self) -> None:
        with patch.object(
            anti_cheat_scan.public_bench,
            "public_task_set",
            return_value=("zebra-puzzle",),
        ):
            case = anti_cheat_scan.coding_bench.BenchmarkCase(
                name="leaky",
                suite="local-full",
                turns=("Implement the zebra-puzzle exercise from this repo and return 123.",),
                benchmark_kind="coding_accuracy",
                validate=lambda ctx: "pass",
            )
            with patch.object(anti_cheat_scan.coding_bench, "selected_cases", return_value=[case]):
                findings = anti_cheat_scan.scan_benchmark_prompts()

        self.assertEqual(len(findings), 1)
        coding_findings = [item for item in findings if item.get("scope") == "coding_prompt"]
        self.assertEqual(len(coding_findings), 1)
        self.assertIn("public benchmark task slug", coding_findings[0]["findings"])

    def test_runtime_scan_uses_expanded_public_task_set(self) -> None:
        with patch.object(
            anti_cheat_scan.public_bench,
            "public_task_set",
            return_value=("affine-cipher",),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = root / "ollama_code"
                package.mkdir()
                (package / "agent.py").write_text("if mode == 'affine-cipher': return 'ok'\n", encoding="utf-8")

                findings = anti_cheat_scan.scan_runtime(root)

        self.assertEqual(len(findings), 1)
        self.assertIn("hard-coded public smoke task", findings[0]["findings"])


if __name__ == "__main__":
    unittest.main()
