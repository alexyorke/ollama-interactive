from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import gguf_inspect as inspect_gguf


class GgufInspectTests(unittest.TestCase):
    def test_main_skips_stale_manifest_entries_with_missing_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifests" / "registry.ollama.ai" / "library" / "demo" / "latest"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": "sha256:deadbeef",
                                "size": 123,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "gguf.json"

            exit_code = inspect_gguf.main(["--models-dir", str(root), "--json", "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["results"], [])

    def test_main_handles_empty_models_dir_with_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "gguf.json"

            exit_code = inspect_gguf.main(["--models-dir", str(root), "--json", "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["models_dir"], str(root))
        self.assertEqual(payload["results"], [])

    def test_main_handles_empty_models_dir_without_crashing_table_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            exit_code = inspect_gguf.main(["--models-dir", str(root)])

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
