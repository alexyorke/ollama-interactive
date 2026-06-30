from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import live_matrix


class LiveMatrixTests(unittest.TestCase):
    def test_installed_models_returns_empty_list_when_ollama_tags_are_unreachable(self) -> None:
        with patch("scripts.live_matrix.urllib.request.urlopen", side_effect=OSError("unreachable")):
            models = live_matrix.installed_models()

        self.assertEqual(models, [])

    def test_resolve_requested_models_dedupes_latest_aliases_while_preserving_order(self) -> None:
        models = live_matrix.resolve_requested_models(
            ["gemma4:e4b", "gemma4:e4b:latest", "qwen3", "qwen3:latest", "gemma4:e4b"],
            {"gemma4:e4b", "gemma4:e4b:latest", "qwen3:latest"},
        )

        self.assertEqual(models, ["gemma4:e4b", "gemma4:e4b:latest", "qwen3:latest"])


if __name__ == "__main__":
    unittest.main()
