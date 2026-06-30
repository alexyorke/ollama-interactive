import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts import trajectory_dataset_catalog as catalog


class TrajectoryDatasetCatalogTests(unittest.TestCase):
    def test_summary_dedupes_repeated_repo_ids_in_id_lists(self) -> None:
        summary = catalog._summary(
            [
                {
                    "id": "nvidia/Open-SWE-Traces",
                    "source": "huggingface",
                    "local_dir": "/tmp/openhands",
                    "analysis_ready": True,
                    "access_status": "public",
                    "priority": "high",
                },
                {
                    "id": "nvidia/Open-SWE-Traces",
                    "source": "huggingface",
                    "local_dir": "/tmp/sweagent",
                    "analysis_ready": True,
                    "access_status": "public",
                    "priority": "high",
                },
                {
                    "id": "acme/missing",
                    "source": "huggingface",
                    "local_dir": None,
                    "analysis_ready": False,
                    "access_status": "public",
                    "priority": "high",
                    "reason": "Missing local copy.",
                },
            ]
        )

        self.assertEqual(summary["local_entries"], 2)
        self.assertEqual(summary["analysis_ready_local_entries"], 2)
        self.assertEqual(summary["local_ids"], ["nvidia/Open-SWE-Traces"])
        self.assertEqual(summary["analysis_ready_local_ids"], ["nvidia/Open-SWE-Traces"])
        self.assertEqual(summary["public_missing_ids"], ["acme/missing"])

    def test_summary_dedupes_high_priority_public_missing_repo_ids(self) -> None:
        summary = catalog._summary(
            [
                {
                    "id": "nvidia/Open-SWE-Traces",
                    "source": "huggingface",
                    "local_dir": None,
                    "analysis_ready": False,
                    "access_status": "public",
                    "priority": "high",
                    "reason": "OpenHands variant.",
                },
                {
                    "id": "nvidia/Open-SWE-Traces",
                    "source": "huggingface",
                    "local_dir": None,
                    "analysis_ready": False,
                    "access_status": "public",
                    "priority": "high",
                    "reason": "SWE-agent variant.",
                },
                {
                    "id": "acme/missing",
                    "source": "huggingface",
                    "local_dir": None,
                    "analysis_ready": False,
                    "access_status": "public",
                    "priority": "high",
                    "reason": "Missing local copy.",
                },
            ]
        )

        self.assertEqual(
            summary["high_priority_public_missing"],
            [
                {"id": "nvidia/Open-SWE-Traces", "reason": "OpenHands variant."},
                {"id": "acme/missing", "reason": "Missing local copy."},
            ],
        )

    def test_default_candidates_include_real_user_and_unified_public_corpora(self) -> None:
        entries = {str(item["repo_id"]): item for item in catalog.HF_DATASET_CANDIDATES}

        self.assertIn("trace-commons/agent-traces", entries)
        self.assertEqual(entries["trace-commons/agent-traces"]["slug"], "trace-commons-agent-traces")
        self.assertEqual(entries["trace-commons/agent-traces"]["priority"], "high")
        self.assertEqual(entries["trace-commons/agent-traces"]["kind"], "real-user-coding-agent-sessions")
        self.assertEqual(entries["trace-commons/agent-traces"]["adapter_hint"], "trace_commons")
        self.assertEqual(entries["trace-commons/agent-traces"]["path_globs"], ("data/train-*.parquet",))

        self.assertIn("thoughtworks/agentic-coding-trajectories", entries)
        self.assertEqual(
            entries["thoughtworks/agentic-coding-trajectories"]["slug"],
            "thoughtworks-agentic-coding-trajectories",
        )
        self.assertEqual(entries["thoughtworks/agentic-coding-trajectories"]["priority"], "medium")
        self.assertEqual(entries["thoughtworks/agentic-coding-trajectories"]["kind"], "unified-agentic-coding-corpus")
        self.assertEqual(entries["thoughtworks/agentic-coding-trajectories"]["adapter_hint"], "thoughtworks")
        self.assertEqual(entries["thoughtworks/agentic-coding-trajectories"]["path_globs"], ("sessions.parquet",))

        self.assertIn("davanstrien/agent-race-traces", entries)
        self.assertEqual(entries["davanstrien/agent-race-traces"]["slug"], "agent-race-traces")
        self.assertEqual(entries["davanstrien/agent-race-traces"]["adapter_hint"], "agent_race")

        self.assertIn("yoonholee/terminalbench-trajectories", entries)
        self.assertEqual(entries["yoonholee/terminalbench-trajectories"]["adapter_hint"], "terminalbench")
        self.assertEqual(entries["yoonholee/terminalbench-trajectories"]["path_globs"], ("data/train-*.parquet",))

        self.assertIn("zai-org/CC-Bench-trajectories", entries)
        self.assertEqual(entries["zai-org/CC-Bench-trajectories"]["slug"], "cc-bench-trajectories")
        self.assertEqual(entries["zai-org/CC-Bench-trajectories"]["adapter_hint"], "cc_bench")
        self.assertEqual(entries["zai-org/CC-Bench-trajectories"]["path_globs"], ("train.parquet",))

        self.assertIn("NJU-LINK/CodeTraceBench", entries)
        self.assertEqual(entries["NJU-LINK/CodeTraceBench"]["slug"], "codetracebench")
        self.assertIsNone(entries["NJU-LINK/CodeTraceBench"]["adapter_hint"])
        self.assertEqual(
            entries["NJU-LINK/CodeTraceBench"]["path_globs"],
            ("bench_manifest.parquet", "bench_manifest.verified.parquet"),
        )

        self.assertIn("AlienKevin/SWE-ZERO-12M-trajectories", entries)
        self.assertEqual(
            entries["AlienKevin/SWE-ZERO-12M-trajectories"]["slug"],
            "swe-zero-12m-trajectories",
        )
        self.assertIsNone(entries["AlienKevin/SWE-ZERO-12M-trajectories"]["adapter_hint"])
        self.assertEqual(
            entries["AlienKevin/SWE-ZERO-12M-trajectories"]["path_globs"],
            ("data/train-*.parquet",),
        )

        self.assertIn("badlogicgames/pi-mono", entries)
        self.assertEqual(entries["badlogicgames/pi-mono"]["slug"], "pi-mono")
        self.assertIsNone(entries["badlogicgames/pi-mono"]["adapter_hint"])
        self.assertEqual(entries["badlogicgames/pi-mono"]["path_globs"], ("*.jsonl", "manifest.jsonl"))

        self.assertIn("peteromallet/my-personal-codex-data", entries)
        self.assertEqual(entries["peteromallet/my-personal-codex-data"]["slug"], "personal-codex-dataclaw")
        self.assertIsNone(entries["peteromallet/my-personal-codex-data"]["adapter_hint"])
        self.assertEqual(
            entries["peteromallet/my-personal-codex-data"]["path_globs"],
            ("codex/*.jsonl", ".dataclaw/manifest.json"),
        )

        self.assertIn("misterkerns/my-personal-claude-code-data", entries)
        self.assertEqual(
            entries["misterkerns/my-personal-claude-code-data"]["slug"],
            "personal-claude-code-dataclaw",
        )
        self.assertIsNone(entries["misterkerns/my-personal-claude-code-data"]["adapter_hint"])
        self.assertEqual(
            entries["misterkerns/my-personal-claude-code-data"]["path_globs"],
            ("conversations.jsonl", "metadata.json"),
        )

        self.assertIn("ultralazr/claude-code-traces", entries)
        self.assertEqual(entries["ultralazr/claude-code-traces"]["slug"], "ultralazr-claude-code-traces")
        self.assertIsNone(entries["ultralazr/claude-code-traces"]["adapter_hint"])
        self.assertEqual(
            entries["ultralazr/claude-code-traces"]["path_globs"],
            ("data/*.jsonl", "manifest.jsonl"),
        )

        self.assertIn("Glint-Research/Fable-5-traces", entries)
        self.assertEqual(entries["Glint-Research/Fable-5-traces"]["slug"], "fable-5-traces")
        self.assertIsNone(entries["Glint-Research/Fable-5-traces"]["adapter_hint"])
        self.assertEqual(entries["Glint-Research/Fable-5-traces"]["path_globs"], ("pi-traces/*.jsonl",))

        self.assertIn("nmuendler/share-codex", entries)
        self.assertEqual(entries["nmuendler/share-codex"]["slug"], "share-codex")
        self.assertIsNone(entries["nmuendler/share-codex"]["adapter_hint"])
        self.assertEqual(entries["nmuendler/share-codex"]["path_globs"], ("train.jsonl", "export_manifest.json"))

        self.assertIn("nvidia/SWE-Zero-openhands-trajectories", entries)
        self.assertEqual(
            entries["nvidia/SWE-Zero-openhands-trajectories"]["slug"],
            "nvidia-swe-zero-openhands-trajectories",
        )
        self.assertEqual(entries["nvidia/SWE-Zero-openhands-trajectories"]["adapter_hint"], "openhands")
        self.assertEqual(
            entries["nvidia/SWE-Zero-openhands-trajectories"]["path_globs"],
            ("data/train-*.parquet",),
        )

        open_swe_entries = [item for item in catalog.HF_DATASET_CANDIDATES if item["repo_id"] == "nvidia/Open-SWE-Traces"]
        self.assertEqual(len(open_swe_entries), 2)
        self.assertEqual({item["slug"] for item in open_swe_entries}, {"open-swe-traces-openhands", "open-swe-traces-sweagent"})
        self.assertEqual({item["adapter_hint"] for item in open_swe_entries}, {"openhands"})

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

    def test_external_candidates_keep_current_metadata_only_source(self) -> None:
        entry = next(item for item in catalog.EXTERNAL_CANDIDATES if item["slug"] == "agentlens-bench")
        self.assertEqual(entry["id"], "agentlens/process-quality-paper")
        self.assertEqual(entry["source"], "paper")
        self.assertEqual(entry["url"], "https://arxiv.org/abs/2605.12925")
        self.assertIn("still planned", entry["reason"])


if __name__ == "__main__":
    unittest.main()
