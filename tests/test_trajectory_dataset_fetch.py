import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import trajectory_dataset_fetch as fetch


class TrajectoryDatasetFetchTests(unittest.TestCase):
    def test_fetch_datasets_removes_stale_manifest_after_refresh_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_name = "nebius-swe-agent-trajectories"
            dataset_dir = data_root / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            stale_manifest_path = fetch.dataset_manifest_path(data_root, dataset_name)
            stale_manifest_path.write_text(
                '{"dataset":"nebius-swe-agent-trajectories","repo_id":"nebius/SWE-agent-trajectories","resolved_revision":"old"}',
                encoding="utf-8",
            )

            class FakeApi:
                def dataset_info(self, requested_repo_id: str, revision: str | None = None) -> object:
                    return types.SimpleNamespace(sha="newsha")

            def fake_download(**_: object) -> str:
                raise RuntimeError("download failed")

            payload = fetch.fetch_datasets(
                [dataset_name],
                data_root,
                revision="main",
                api=FakeApi(),
                downloader=fake_download,
            )

            row = payload["datasets"][0]
            manifest = fetch.read_dataset_manifest(data_root, dataset_name)

        self.assertFalse(row["ok"])
        self.assertEqual(row["status"], "error")
        self.assertIn("download failed", row["error"])
        self.assertIsNone(manifest)
        self.assertFalse(stale_manifest_path.exists())

    def test_fetch_datasets_writes_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_name = "nebius-swe-agent-trajectories"
            repo_id = fetch.SUPPORTED_DATASET_SPECS[dataset_name]["repo_id"]

            class FakeApi:
                def dataset_info(self, requested_repo_id: str, revision: str | None = None) -> object:
                    self.assertEqual(requested_repo_id, repo_id)
                    self.assertEqual(revision, "main")
                    return types.SimpleNamespace(sha="abc123")

                def assertEqual(self, left: object, right: object) -> None:
                    if left != right:
                        raise AssertionError(f"{left!r} != {right!r}")

            def fake_download(**kwargs: object) -> str:
                self.assertEqual(kwargs["repo_id"], repo_id)
                self.assertEqual(kwargs["repo_type"], "dataset")
                self.assertEqual(kwargs["revision"], "main")
                self.assertEqual(kwargs["allow_patterns"], ["data/*.parquet"])
                local_dir = Path(str(kwargs["local_dir"]))
                (local_dir / "data").mkdir(parents=True, exist_ok=True)
                (local_dir / "data" / "train-00000.parquet").write_text("stub\n", encoding="utf-8")
                return str(local_dir)

            payload = fetch.fetch_datasets(
                [dataset_name],
                data_root,
                revision="main",
                api=FakeApi(),
                downloader=fake_download,
            )
            row = payload["datasets"][0]
            manifest = fetch.read_dataset_manifest(data_root, dataset_name)

        self.assertTrue(row["ok"])
        self.assertEqual(row["status"], "downloaded")
        self.assertEqual(row["resolved_revision"], "abc123")
        self.assertEqual(row["files"], ["data/train-00000.parquet"])
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest["repo_id"], repo_id)
        self.assertEqual(manifest["resolved_revision"], "abc123")
        self.assertEqual(manifest["file_count"], 1)

    def test_fetch_datasets_fails_cleanly_without_huggingface_hub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            with patch.object(fetch, "HfApi", None), patch.object(fetch, "snapshot_download", None):
                with self.assertRaisesRegex(RuntimeError, "huggingface_hub is required"):
                    fetch.fetch_datasets(["nebius-swe-agent-trajectories"], data_root)

    def test_coderforge_dataset_is_supported_with_trajectory_glob(self) -> None:
        dataset_name = "coderforge-preview-swe-bench-verified-trajectories"

        spec = fetch.SUPPORTED_DATASET_SPECS[dataset_name]

        self.assertEqual(
            spec["repo_id"],
            "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories",
        )
        self.assertEqual(spec["adapter"], "openhands")
        self.assertEqual(spec["path_globs"], ["trajectory/train-*.parquet"])

    def test_agent_race_dataset_is_supported_with_jsonl_glob(self) -> None:
        dataset_name = "agent-race-traces"

        spec = fetch.SUPPORTED_DATASET_SPECS[dataset_name]

        self.assertEqual(spec["repo_id"], "davanstrien/agent-race-traces")
        self.assertEqual(spec["adapter"], "agent_race")
        self.assertEqual(spec["path_globs"], ["*.jsonl"])

    def test_terminalbench_dataset_is_supported_with_step_adapter(self) -> None:
        dataset_name = "terminalbench-trajectories"

        spec = fetch.SUPPORTED_DATASET_SPECS[dataset_name]

        self.assertEqual(spec["repo_id"], "yoonholee/terminalbench-trajectories")
        self.assertEqual(spec["adapter"], "terminalbench")
        self.assertEqual(spec["path_globs"], ["data/train-*.parquet"])

    def test_thoughtworks_dataset_is_supported_with_session_parquet(self) -> None:
        dataset_name = "thoughtworks-agentic-coding-trajectories"

        spec = fetch.SUPPORTED_DATASET_SPECS[dataset_name]

        self.assertEqual(spec["repo_id"], "thoughtworks/agentic-coding-trajectories")
        self.assertEqual(spec["adapter"], "thoughtworks")
        self.assertEqual(spec["path_globs"], ["sessions.parquet"])

    def test_trace_commons_dataset_is_supported_with_data_parquet(self) -> None:
        dataset_name = "trace-commons-agent-traces"

        spec = fetch.SUPPORTED_DATASET_SPECS[dataset_name]

        self.assertEqual(spec["repo_id"], "trace-commons/agent-traces")
        self.assertEqual(spec["adapter"], "trace_commons")
        self.assertEqual(spec["path_globs"], ["data/train-*.parquet"])

    def test_swe_zero_dataset_is_supported_with_openhands_glob(self) -> None:
        dataset_name = "nvidia-swe-zero-openhands-trajectories"

        spec = fetch.SUPPORTED_DATASET_SPECS[dataset_name]

        self.assertEqual(spec["repo_id"], "nvidia/SWE-Zero-openhands-trajectories")
        self.assertEqual(spec["adapter"], "openhands")
        self.assertEqual(spec["path_globs"], ["data/train-*.parquet"])

    def test_open_swe_traces_datasets_are_supported_with_config_specific_globs(self) -> None:
        openhands_spec = fetch.SUPPORTED_DATASET_SPECS["open-swe-traces-openhands"]
        sweagent_spec = fetch.SUPPORTED_DATASET_SPECS["open-swe-traces-sweagent"]

        self.assertEqual(openhands_spec["repo_id"], "nvidia/Open-SWE-Traces")
        self.assertEqual(openhands_spec["adapter"], "openhands")
        self.assertEqual(openhands_spec["path_globs"], ["data/*_openhands_trajectories/train-*.parquet"])

        self.assertEqual(sweagent_spec["repo_id"], "nvidia/Open-SWE-Traces")
        self.assertEqual(sweagent_spec["adapter"], "openhands")
        self.assertEqual(sweagent_spec["path_globs"], ["data/*_sweagent_trajectories/train-*.parquet"])


if __name__ == "__main__":
    unittest.main()
