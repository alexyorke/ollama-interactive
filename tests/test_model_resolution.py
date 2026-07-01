from __future__ import annotations

import unittest

from scripts.model_resolution import resolve_requested_model


class ModelResolutionTests(unittest.TestCase):
    def test_prefers_exact_match(self) -> None:
        self.assertEqual(resolve_requested_model("gemma4:e4b", {"gemma4:e4b", "gemma4:e4b:latest"}), "gemma4:e4b")

    def test_falls_back_to_latest_alias(self) -> None:
        self.assertEqual(resolve_requested_model("qwen3", {"qwen3:latest"}), "qwen3:latest")
        self.assertIsNone(resolve_requested_model("missing", {"qwen3:latest"}))
