from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import token_efficiency_eval


class TokenEfficiencyEvalTests(unittest.TestCase):
    def test_main_returns_nonzero_when_model_discovery_is_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "token-eval.json"
            with patch("scripts.e2e_suite.urllib.request.urlopen", side_effect=OSError("unreachable")):
                exit_code = token_efficiency_eval.main(["--models", "demo-model", "--output", str(output_path)])

        self.assertEqual(exit_code, 1)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
