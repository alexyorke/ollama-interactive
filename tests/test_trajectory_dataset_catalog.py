import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts import trajectory_dataset_catalog as catalog


class TrajectoryDatasetCatalogTests(unittest.TestCase):
    def test_build_catalog_reports_local_public_and_gated_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "local-openhands").mkdir()

            def fake_fetch(url: str, *, timeout: int = 30) -> object:
                self.assertEqual(timeout, 30)
                if url == "https://huggingface.co/api/datasets/acme/local-openhands":
                    return {
                        "gated": False,
                        "downloads": 7,
                        "lastModified": "2026-05-22T00:00:00.000Z",
                        "siblings": [{"rfilename": "data/train-00000-of-00001.parquet"}],
                    }
                if url == "https://datasets-server.huggingface.co/splits?dataset=acme%2Flocal-openhands":
                    return {"splits": [{"config": "default", "split": "train"}]}
                if url == "https://datasets-server.huggingface.co/first-rows?dataset=acme%2Flocal-openhands&config=default&split=train":
                    return {
                        "features": [{"name": "trajectory"}, {"name": "repo"}],
                        "rows": [{"row": {"trajectory": [], "repo": "demo"}}],
                    }
                if url == "https://huggingface.co/api/datasets/acme/gated-set":
                    raise urllib.error.HTTPError(url, 401, "Unauthorized", hdrs=None, fp=None)
                raise AssertionError(f"unexpected url: {url}")

            with patch.object(
                catalog,
                "HF_DATASET_CANDIDATES",
                (
                    {
                        "repo_id": "acme/local-openhands",
                        "slug": "local-openhands",
                        "priority": "high",
                        "kind": "openhands-trajectories",
                        "reason": "Local compatible corpus.",
                        "adapter_hint": "openhands",
                        "path_globs": ("data/*.parquet",),
                    },
                    {
                        "repo_id": "acme/gated-set",
                        "slug": "gated-set",
                        "priority": "high",
                        "kind": "real-user-coding-agent-sessions",
                        "reason": "Gated example.",
                        "adapter_hint": None,
                        "path_globs": (),
                    },
                ),
            ), patch.object(catalog, "EXTERNAL_CANDIDATES", ()), patch.object(catalog, "_fetch_json", side_effect=fake_fetch):
                payload = catalog.build_catalog(data_root, include_preview=True)

        summary = payload["summary"]
        self.assertEqual(summary["entries"], 2)
        self.assertEqual(summary["local_entries"], 1)
        self.assertEqual(summary["analysis_ready_local_entries"], 1)
        self.assertEqual(summary["gated_entries"], 1)
        local_entry = next(item for item in payload["entries"] if item["id"] == "acme/local-openhands")
        self.assertEqual(local_entry["access_status"], "public")
        self.assertTrue(local_entry["analysis_ready"])
        self.assertIn("trajectory", local_entry["remote_preview"]["feature_names"])
        gated_entry = next(item for item in payload["entries"] if item["id"] == "acme/gated-set")
        self.assertEqual(gated_entry["access_status"], "gated")


if __name__ == "__main__":
    unittest.main()
