from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

from scripts import trajectory_transcript_sampler as sampler


@unittest.skipIf(pa is None or pq is None, "pyarrow not installed")
class TrajectoryTranscriptSamplerTests(unittest.TestCase):
    def test_build_parser_accepts_legacy_output_flag(self) -> None:
        args = sampler._build_parser().parse_args(["--output", "scratch/external/datasets/transcript-review.json"])

        self.assertEqual(args.output, Path("scratch/external/datasets/transcript-review.json"))

    def test_resolve_output_paths_from_legacy_output_json_path(self) -> None:
        output_json, output_md = sampler._resolve_output_paths(
            output=Path("scratch/external/datasets/transcript-review.json"),
            output_json=None,
            output_md=None,
        )

        self.assertEqual(output_json, Path("scratch/external/datasets/transcript-review.json"))
        self.assertEqual(output_md, Path("scratch/external/datasets/transcript-review.md"))

    def test_sample_dataset_reads_trace_commons_prompt_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "harness": ["claude-code"],
                    "session_id": ["s1"],
                    "prompt": ["Build a metrics dashboard for sales data."],
                    "messages": [
                        json.dumps(
                            [
                                {"role": "user", "content": "Build a metrics dashboard for sales data."},
                                {
                                    "role": "assistant",
                                    "content": "I will inspect the repo first.",
                                    "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
                                },
                            ]
                        )
                    ],
                    "tools": [""],
                    "metadata": ["{}"],
                    "sent_at": [""],
                    "num_user_messages": [1],
                    "num_tool_calls": [1],
                    "trace": [""],
                    "file_path": ["data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "trace-commons-agent-traces",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(summary["rows_scanned"], 1)
        self.assertEqual(summary["top_tasks"], [])
        self.assertEqual(summary["samples"][0]["prompt_preview"], "Build a metrics dashboard for sales data.")
        self.assertEqual(summary["samples"][0]["assistant_preview"], "I will inspect the repo first.")
        self.assertEqual(summary["samples"][0]["tool_preview"], ["read_file"])
        self.assertEqual(summary["top_models_or_harnesses"][0]["name"], "claude-code")

    def test_task_label_ignores_harness_model_and_source_dataset_fallbacks(self) -> None:
        self.assertEqual(
            sampler._task_label(
                {
                    "harness": "claude-code",
                    "model_name": "demo-model",
                    "source_dataset": "swe-smith-claude-3-7-sonnet",
                    "agent_framework": "openhands",
                }
            ),
            "",
        )

    def test_sample_dataset_reads_trace_commons_trace_fallback_when_messages_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "harness": [None],
                    "session_id": [None],
                    "prompt": [None],
                    "messages": [json.dumps(['{"role":"assistant","content":""}'])],
                    "tools": [json.dumps([])],
                    "metadata": [json.dumps({"trace_type": "hermes"})],
                    "sent_at": [None],
                    "num_user_messages": [0],
                    "num_tool_calls": [0],
                    "trace": [
                        json.dumps(
                            [
                                json.dumps(
                                    {
                                        "role": "user",
                                        "message": {
                                            "content": [
                                                {"type": "text", "text": "Which log should I use to monitor API usage by route in Grafana Loki?"}
                                            ]
                                        },
                                    }
                                ),
                                json.dumps(
                                    {
                                        "role": "assistant",
                                        "message": {
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "I will inspect the logging setup and route definitions first.",
                                                },
                                                {
                                                    "type": "tool_use",
                                                    "name": "Read",
                                                    "input": {"path": "/workspace/app/logging.py"},
                                                },
                                            ]
                                        },
                                    }
                                ),
                            ]
                        )
                    ],
                    "file_path": ["data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "trace-commons-agent-traces",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(
            summary["samples"][0]["prompt_preview"],
            "Which log should I use to monitor API usage by route in Grafana Loki?",
        )
        self.assertEqual(
            summary["samples"][0]["assistant_preview"],
            "I will inspect the logging setup and route definitions first.",
        )
        self.assertEqual(summary["samples"][0]["tool_preview"], ["read"])

    def test_sample_dataset_reads_trace_commons_structured_trace_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "harness": [None],
                    "session_id": [None],
                    "prompt": [None],
                    "messages": [json.dumps([])],
                    "tools": [json.dumps([])],
                    "metadata": [json.dumps({"trace_type": "structured"})],
                    "sent_at": [None],
                    "num_user_messages": [0],
                    "num_tool_calls": [0],
                    "trace": [
                        json.dumps(
                            {
                                "info": {"title": "Kernel anticheat"},
                                "messages": [
                                    {
                                        "info": {"role": "user"},
                                        "parts": [
                                            {
                                                "type": "text",
                                                "text": "Look up whether kernel-level anticheat is lazy development or a real security need.",
                                            }
                                        ],
                                    },
                                    {
                                        "info": {"role": "assistant"},
                                        "parts": [
                                            {
                                                "type": "reasoning",
                                                "text": "I will search for multiple expert perspectives first.",
                                            },
                                            {
                                                "type": "tool",
                                                "tool": "websearch",
                                                "state": {"input": {"query": "kernel anti cheat necessary security"}},
                                            },
                                        ],
                                    },
                                ],
                            }
                        )
                    ],
                    "file_path": ["data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "trace-commons-agent-traces",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(
            summary["samples"][0]["prompt_preview"],
            "Look up whether kernel-level anticheat is lazy development or a real security need.",
        )
        self.assertEqual(
            summary["samples"][0]["assistant_preview"],
            "I will search for multiple expert perspectives first.",
        )
        self.assertEqual(summary["samples"][0]["tool_preview"], ["websearch"])

    def test_sample_dataset_prefers_reasoning_content_when_assistant_content_is_tool_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "harness": ["claude-code"],
                    "session_id": ["s1"],
                    "prompt": [None],
                    "messages": [
                        json.dumps(
                            [
                                {"role": "user", "content": "Fix the failing parser and rerun the tests."},
                                {
                                    "role": "assistant",
                                    "content": [{"type": "tool_use", "name": "read_file", "input": {"path": "src/parser.py"}}],
                                    "reasoning_content": "I will inspect src/parser.py first, then rerun the parser tests.",
                                },
                            ]
                        )
                    ],
                    "tools": [json.dumps([])],
                    "metadata": [json.dumps({})],
                    "sent_at": [""],
                    "num_user_messages": [1],
                    "num_tool_calls": [1],
                    "trace": [""],
                    "file_path": ["data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "trace-commons-agent-traces",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(
            summary["samples"][0]["assistant_preview"],
            "I will inspect src/parser.py first, then rerun the parser tests.",
        )

    def test_sample_dataset_reads_terminalbench_prompt_from_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["csv-cleanup"],
                    "agent": ["claude-code"],
                    "model": ["demo-model"],
                    "reward": [1],
                    "duration_seconds": [12.0],
                    "input_tokens": [100.0],
                    "output_tokens": [50.0],
                    "cache_tokens": [0.0],
                    "cost_cents": [0.0],
                    "trial_name": ["trial"],
                    "trial_id": ["t1"],
                    "started_at": [""],
                    "ended_at": [""],
                    "steps": [
                        json.dumps(
                            [
                                {"src": "user", "msg": "Clean the CSV and print summary stats."},
                                {
                                    "src": "agent",
                                    "msg": "I will start by listing files.",
                                    "tools": [{"fn": "run_shell", "cmd": "ls"}],
                                    "obs": "",
                                },
                            ]
                        )
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(summary["samples"][0]["prompt_preview"], "Clean the CSV and print summary stats.")
        self.assertEqual(summary["samples"][0]["assistant_preview"], "I will start by listing files.")
        self.assertEqual(summary["samples"][0]["tool_preview"], ["run_shell"])
        self.assertEqual(summary["top_models_or_harnesses"][0]["name"], "demo-model")

    def test_terminalbench_sampling_prefers_non_warmup_rows_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["warmup-task", "real-task"],
                    "agent": ["claude-code", "claude-code"],
                    "model": ["demo-model", "demo-model"],
                    "reward": [0, 1],
                    "duration_seconds": [1.0, 2.0],
                    "input_tokens": [1.0, 2.0],
                    "output_tokens": [1.0, 3.0],
                    "cache_tokens": [0.0, 0.0],
                    "cost_cents": [0.0, 0.0],
                    "trial_name": ["trial-a", "trial-b"],
                    "trial_id": ["t1", "t2"],
                    "started_at": ["", ""],
                    "ended_at": ["", ""],
                    "steps": [
                        json.dumps(
                            [
                                {"src": "user", "msg": "Warmup"},
                                {"src": "agent", "msg": "I am ready to help.", "tools": [{"fn": "search"}], "obs": ""},
                            ]
                        ),
                        json.dumps(
                            [
                                {"src": "user", "msg": "Debug the failing parser and rerun the tests."},
                                {
                                    "src": "agent",
                                    "msg": "I will inspect the parser module first.",
                                    "tools": [{"fn": "run_shell", "cmd": "pytest -q"}],
                                    "obs": "",
                                },
                            ]
                        ),
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(summary["samples"][0]["prompt_preview"], "Debug the failing parser and rerun the tests.")
        self.assertEqual(summary["samples"][0]["assistant_preview"], "I will inspect the parser module first.")

    def test_terminalbench_placeholder_prompt_falls_back_to_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["adaptive-rejection-sampler"],
                    "agent": ["claude-code"],
                    "model": ["demo-model"],
                    "reward": [1],
                    "duration_seconds": [2.0],
                    "input_tokens": [2.0],
                    "output_tokens": [3.0],
                    "cache_tokens": [0.0],
                    "cost_cents": [0.0],
                    "trial_name": ["trial-a1"],
                    "trial_id": ["t1"],
                    "started_at": [""],
                    "ended_at": [""],
                    "steps": [
                        json.dumps(
                            [
                                {"src": "user", "msg": "$31"},
                                {"src": "agent", "msg": "I will inspect the algorithm first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        )
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(summary["samples"][0]["prompt_preview"], "adaptive-rejection-sampler")

    def test_terminalbench_environment_context_prompt_falls_back_to_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["adaptive-rejection-sampler"],
                    "agent": ["claude-code"],
                    "model": ["demo-model"],
                    "reward": [1],
                    "duration_seconds": [2.0],
                    "input_tokens": [2.0],
                    "output_tokens": [3.0],
                    "cache_tokens": [0.0],
                    "cost_cents": [0.0],
                    "trial_name": ["trial-a1"],
                    "trial_id": ["t1"],
                    "started_at": [""],
                    "ended_at": [""],
                    "steps": [
                        json.dumps(
                            [
                                {
                                    "src": "user",
                                    "msg": "<environment_context>\n<cwd>/app</cwd>\n<approval_policy>never</approval_policy>\n<sandbox_mode>danger-full-access</sandbox_mode>\n<network_access>enabled</network_access>\n<shell>bash</shell>\n</environment_context>",
                                },
                                {"src": "agent", "msg": "I will inspect the algorithm first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        )
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(summary["samples"][0]["prompt_preview"], "adaptive-rejection-sampler")

    def test_terminalbench_environment_context_prefix_preserves_real_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["adaptive-rejection-sampler"],
                    "agent": ["claude-code"],
                    "model": ["demo-model"],
                    "reward": [1],
                    "duration_seconds": [2.0],
                    "input_tokens": [2.0],
                    "output_tokens": [3.0],
                    "cache_tokens": [0.0],
                    "cost_cents": [0.0],
                    "trial_name": ["trial-a1"],
                    "trial_id": ["t1"],
                    "started_at": [""],
                    "ended_at": [""],
                    "steps": [
                        json.dumps(
                            [
                                {
                                    "src": "user",
                                    "msg": "<environment_context>\n<cwd>/app</cwd>\n<approval_policy>never</approval_policy>\n</environment_context>\nImplement adaptive rejection sampling in sampler.py and run tests.",
                                },
                                {"src": "agent", "msg": "I will inspect the repository first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        )
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(
            summary["samples"][0]["prompt_preview"],
            "Implement adaptive rejection sampling in sampler.py and run tests.",
        )

    def test_iter_dataset_rows_skips_terminalbench_null_rows_before_max_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["skip", "keep-1", "keep-2"],
                    "trial_id": ["skip-1", "keep-1", "keep-2"],
                    "steps": [
                        "null",
                        '[{"src":"user","msg":"Implement the fix."},{"src":"agent","msg":"I will inspect first.","tools":[{"fn":"bash_command","cmd":"pytest -q"}],"obs":"ok"}]',
                        '[{"src":"user","msg":"Implement the other fix."},{"src":"agent","msg":"I will inspect first.","tools":[{"fn":"bash_command","cmd":"pytest -q"}],"obs":"ok"}]',
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            adapter, rows = sampler._iter_dataset_rows(root, "terminalbench-trajectories", 1)
            filtered = list(rows)

        self.assertEqual(adapter, "terminalbench")
        self.assertEqual([row["trial_id"] for row in filtered], ["keep-1"])

    def test_terminalbench_sampling_diversifies_tasks_when_scores_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["task-a", "task-a", "task-b"],
                    "agent": ["claude-code", "claude-code", "claude-code"],
                    "model": ["demo-model", "demo-model", "demo-model"],
                    "reward": [1, 1, 1],
                    "duration_seconds": [2.0, 2.0, 2.0],
                    "input_tokens": [2.0, 2.0, 2.0],
                    "output_tokens": [3.0, 3.0, 3.0],
                    "cache_tokens": [0.0, 0.0, 0.0],
                    "cost_cents": [0.0, 0.0, 0.0],
                    "trial_name": ["trial-a1", "trial-a2", "trial-b1"],
                    "trial_id": ["t1", "t2", "t3"],
                    "started_at": ["", "", ""],
                    "ended_at": ["", "", ""],
                    "steps": [
                        json.dumps(
                            [
                                {"src": "user", "msg": "Fix parser bug in task A."},
                                {"src": "agent", "msg": "I will inspect parser_a.py first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        ),
                        json.dumps(
                            [
                                {"src": "user", "msg": "Fix parser bug in task A."},
                                {"src": "agent", "msg": "I will inspect parser_a.py first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        ),
                        json.dumps(
                            [
                                {"src": "user", "msg": "Repair serializer edge case in task B."},
                                {"src": "agent", "msg": "I will inspect serializer_b.py first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        ),
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=2,
            )

        self.assertEqual(
            [sample["task"] for sample in summary["samples"]],
            ["task-a", "task-b"],
        )

    def test_terminalbench_sampling_avoids_near_duplicate_prompt_shape_with_same_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["task-a", "task-a", "task-a"],
                    "agent": ["claude-code", "claude-code", "claude-code"],
                    "model": ["demo-model", "demo-model", "demo-model"],
                    "reward": [1, 1, 1],
                    "duration_seconds": [2.0, 2.0, 2.0],
                    "input_tokens": [2.0, 2.0, 2.0],
                    "output_tokens": [3.0, 3.0, 3.0],
                    "cache_tokens": [0.0, 0.0, 0.0],
                    "cost_cents": [0.0, 0.0, 0.0],
                    "trial_name": ["trial-a1", "trial-a2", "trial-a3"],
                    "trial_id": ["t1", "t2", "t3"],
                    "started_at": ["", "", ""],
                    "ended_at": ["", "", ""],
                    "steps": [
                        json.dumps(
                            [
                                {"src": "user", "msg": "Fix parser bug in task A."},
                                {"src": "agent", "msg": "I will inspect parser_a.py first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        ),
                        json.dumps(
                            [
                                {"src": "user", "msg": "Fix parser bug in task A."},
                                {"src": "agent", "msg": "I will inspect parser_a.py first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        ),
                        json.dumps(
                            [
                                {"src": "user", "msg": "Repair serializer edge case in task A."},
                                {"src": "agent", "msg": "I will inspect serializer_a.py first.", "tools": [{"fn": "run_shell", "cmd": "pytest -q"}], "obs": ""},
                            ]
                        ),
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "terminalbench-trajectories",
                max_rows=10,
                samples_per_dataset=2,
            )

        self.assertEqual(
            [sample["prompt_preview"] for sample in summary["samples"]],
            ["Fix parser bug in task A.", "Repair serializer edge case in task A."],
        )

    def test_build_report_formats_multiple_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_root = root / "trace-commons-agent-traces" / "data"
            trace_root.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "harness": ["claude-code"],
                        "session_id": ["s1"],
                        "prompt": ["Build a landing page."],
                        "messages": ['[{"role":"user","content":"Build a landing page."}]'],
                        "tools": [""],
                        "metadata": ["{}"],
                        "sent_at": [""],
                        "num_user_messages": [1],
                        "num_tool_calls": [0],
                        "trace": [""],
                        "file_path": ["x"],
                    }
                ),
                trace_root / "train-00000-of-00001.parquet",
            )
            terminal_root = root / "terminalbench-trajectories" / "data"
            terminal_root.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "task_name": ["shell-fix"],
                        "agent": ["claude-code"],
                        "model": ["demo"],
                        "reward": [0],
                        "duration_seconds": [1.0],
                        "input_tokens": [1.0],
                        "output_tokens": [1.0],
                        "cache_tokens": [0.0],
                        "cost_cents": [0.0],
                        "trial_name": ["trial"],
                        "trial_id": ["t2"],
                        "started_at": [""],
                        "ended_at": [""],
                        "steps": ['[{"src":"user","msg":"Fix the shell script."}]'],
                    }
                ),
                terminal_root / "train-00000-of-00001.parquet",
            )

            payload = sampler.build_report(
                data_root=root,
                datasets=["trace-commons-agent-traces", "terminalbench-trajectories"],
                max_rows=10,
                samples_per_dataset=1,
            )
            markdown = sampler._format_markdown(payload)

        self.assertEqual(len(payload["datasets"]), 2)
        self.assertIn("## trace-commons-agent-traces", markdown)
        self.assertIn("## terminalbench-trajectories", markdown)
        self.assertIn("Build a landing page.", markdown)
        self.assertIn("Fix the shell script.", markdown)

    def test_sample_dataset_prefers_duplicate_informative_rows_over_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "harness": ["claude_code", "claude_code", "", ""],
                    "session_id": ["s1", "s2", "s3", "s4"],
                    "prompt": [
                        "Build a metrics dashboard.",
                        "Build a metrics dashboard.",
                        "",
                        "",
                    ],
                    "messages": [
                        json.dumps(
                            [
                                {"role": "user", "content": "Build a metrics dashboard."},
                                {"role": "assistant", "content": "I will inspect the repo first."},
                            ]
                        ),
                        json.dumps(
                            [
                                {"role": "user", "content": "Build a metrics dashboard."},
                                {"role": "assistant", "content": "I will inspect the repo first."},
                            ]
                        ),
                        json.dumps([]),
                        json.dumps([]),
                    ],
                    "tools": ["", "", "", ""],
                    "metadata": ["{}", "{}", "{}", "{}"],
                    "sent_at": ["", "", "", ""],
                    "num_user_messages": [1, 1, 0, 0],
                    "num_tool_calls": [0, 0, 0, 0],
                    "trace": ["", "", "", ""],
                    "file_path": [
                        "data/train-00000-of-00001.parquet",
                        "data/train-00000-of-00001.parquet",
                        "data/train-00000-of-00001.parquet",
                        "data/train-00000-of-00001.parquet",
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = sampler.sample_dataset(
                root,
                "trace-commons-agent-traces",
                max_rows=10,
                samples_per_dataset=2,
            )

        self.assertEqual(len(summary["samples"]), 2)
        self.assertEqual(
            [sample["prompt_preview"] for sample in summary["samples"]],
            ["Build a metrics dashboard.", "Build a metrics dashboard."],
        )
        self.assertEqual(
            [sample["assistant_preview"] for sample in summary["samples"]],
            ["I will inspect the repo first.", "I will inspect the repo first."],
        )

    def test_agent_race_preview_ignores_signature_blob_and_tool_use_wrapper(self) -> None:
        row = {
            "events": [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
                        ]
                    },
                },
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "name": "plan_tool", "input": {"todos": []}},
                        ]
                    },
                },
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "I will inspect the dataset schema first."},
                        ]
                    },
                },
            ]
        }

        preview = sampler._assistant_preview("agent_race", row)

        self.assertEqual(preview, "I will inspect the dataset schema first.")

    def test_agent_race_preview_reads_nested_message_roles_inside_message_events(self) -> None:
        row = {
            "events": [
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Explore the dataset schema and train the model."},
                        ],
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me inspect the available tools first."},
                            {"type": "toolCall", "id": "functions.bash:0", "name": "bash", "arguments": {"command": "hf --help"}},
                        ],
                    },
                },
            ]
        }

        prompt = sampler._prompt_preview("agent_race", row)
        preview = sampler._assistant_preview("agent_race", row)

        self.assertEqual(prompt, "Explore the dataset schema and train the model.")
        self.assertEqual(preview, "Let me inspect the available tools first.")

    def test_agent_race_summary_falls_back_to_source_file_for_harness_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "agent-race-traces"
            dataset_root.mkdir(parents=True)
            (dataset_root / "claude-code.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"type": "user", "message": {"content": "Build a dashboard."}, "sessionId": "s1"}),
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {"content": [{"type": "text", "text": "I will inspect the repo."}]},
                                "sessionId": "s1",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = sampler.sample_dataset(
                root,
                "agent-race-traces",
                max_rows=10,
                samples_per_dataset=1,
            )

        self.assertEqual(summary["rows_scanned"], 1)
        self.assertEqual(summary["top_models_or_harnesses"][0]["name"], "claude-code")


if __name__ == "__main__":
    unittest.main()
