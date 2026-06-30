from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import verification_eval


class VerificationEvalTests(unittest.TestCase):
    def test_resolve_requested_models_dedupes_latest_aliases_while_preserving_order(self) -> None:
        models = verification_eval.resolve_requested_models(
            ["gemma4:e4b", "gemma4:e4b:latest", "qwen3", "qwen3:latest", "gemma4:e4b"],
            {"gemma4:e4b", "gemma4:e4b:latest", "qwen3:latest"},
        )

        self.assertEqual(models, ["gemma4:e4b", "gemma4:e4b:latest", "qwen3:latest"])

    def test_main_returns_nonzero_when_model_discovery_is_unreachable(self) -> None:
        with patch("scripts.e2e_suite.urllib.request.urlopen", side_effect=OSError("unreachable")):
            exit_code = verification_eval.main(["--models", "demo-model"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
