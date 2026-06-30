from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import e2e_suite


class E2ESuiteTests(unittest.TestCase):
    def test_installed_models_returns_empty_list_when_ollama_tags_are_unreachable(self) -> None:
        with patch("scripts.e2e_suite.urllib.request.urlopen", side_effect=OSError("unreachable")):
            models = e2e_suite.installed_models()

        self.assertEqual(models, [])


if __name__ == "__main__":
    unittest.main()
