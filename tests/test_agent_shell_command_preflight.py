import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from ollama_code.agent import OllamaCodeAgent
from ollama_code.features import ENV_OLLAMA_CODE_FEATURE_PROFILE
from ollama_code.tools import ToolExecutor
from tests.agent_test_support import AgentTestBase, CountingToolExecutor, FakeClient


class AgentShellCommandPreflightTests(AgentTestBase):
    def test_agent_normalizes_bare_python_test_file_shell_command_to_run_test(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text(
            "def test_ok():\n    assert True\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'bare_test OK\')"')
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "tests/test_sample.py"},
            request_text="Validate the project and summarize failures.",
            exact_shell_command=None,
        )

        self.assertEqual(name, "run_test")
        self.assertEqual(arguments["command"], 'python -c "print(\'bare_test OK\')"')
        self.assertIn("bare Python test-file", reason)

    def test_agent_normalizes_bare_python_test_file_shell_command_to_pytest_without_config(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text(
            "def test_ok():\n    assert True\n",
            encoding="utf-8",
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "tests/test_sample.py"},
            request_text="Validate the project and summarize failures.",
            exact_shell_command=None,
        )

        self.assertEqual(name, "run_test")
        self.assertEqual(arguments["command"], "python -m pytest tests/test_sample.py")
        self.assertIn("pytest", reason)

    def test_agent_preserves_exact_user_requested_bare_python_test_file_shell_command(self) -> None:
        root = self._workspace_scratch()
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text("print('shell only')\n", encoding="utf-8")
        tools = ToolExecutor(root, approval_mode="auto", test_command='python -c "print(\'bare_test OK\')"')
        agent = OllamaCodeAgent(client=FakeClient([]), tools=tools, model="fake-model", debate_enabled=False)

        name, arguments, reason = agent._normalize_shell_test_call(
            "run_shell",
            {"command": "tests/test_sample.py"},
            request_text="Use run_shell to execute exactly: tests/test_sample.py.",
            exact_shell_command="tests/test_sample.py",
        )

        self.assertEqual(name, "run_shell")
        self.assertEqual(arguments, {"command": "tests/test_sample.py"})
        self.assertIsNone(reason)

    def test_shell_recursive_grep_inspection_normalizes_to_search(self) -> None:
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"grep -r \\"THREADPOOL\\" azure_functions_worker"}}',
                '{"type":"final","message":"THREADPOOL is in constants.py"}',
            ]
        )
        root = self._workspace_scratch()
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)
        (root / "azure_functions_worker").mkdir()
        (root / "azure_functions_worker" / "constants.py").write_text("PYTHON_THREADPOOL_THREAD_COUNT = 1\n", encoding="utf-8")

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Inspect azure_functions_worker for THREADPOOL and summarize.")

        self.assertEqual(result.message, "THREADPOOL is in constants.py")
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "THREADPOOL", "path": "azure_functions_worker"})

    def test_shell_recursive_grep_with_unsupported_flags_does_not_normalize(self) -> None:
        root = self._workspace_scratch()
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        normalized_name, normalized_arguments, reason = agent._normalize_shell_inspection_call(
            "run_shell",
            {"command": "grep -ri needle src"},
            request_text="Inspect src with recursive ignore-case grep if useful and summarize.",
            exact_shell_command=None,
        )

        self.assertEqual(normalized_name, "run_shell")
        self.assertEqual(normalized_arguments, {"command": "grep -ri needle src"})
        self.assertIsNone(reason)

    def test_shell_find_dot_exec_grep_h_normalizes_to_filtered_search(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
        (root / "README.md").write_text("def main(): docs only\n", encoding="utf-8")
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"find. -name \\"*.py\\" -exec grep -H \\"def main(\\" {} ;"}}',
                '{"type":"final","message":"main is in app.py"}',
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Find Python files containing def main( and summarize.")

        self.assertIn("app.py", result.message)
        self.assertEqual(tools.execute_counts.get("search"), 1)
        self.assertIsNone(tools.execute_counts.get("run_shell"))
        normalized = [event for event in agent.events if event.get("type") == "tool_normalized"]
        self.assertEqual(normalized[0].get("normalized_name"), "search")
        self.assertEqual(normalized[0].get("normalized_arguments"), {"query": "def main(", "path": ".", "file_glob": "*.py"})

    def test_shell_find_dot_exec_grep_unsupported_flags_does_not_normalize(self) -> None:
        root = self._workspace_scratch()
        (root / "app.py").write_text("needle\n", encoding="utf-8")
        agent = OllamaCodeAgent(client=FakeClient([]), tools=ToolExecutor(root, approval_mode="auto"), model="fake-model", debate_enabled=False)

        normalized_name, normalized_arguments, reason = agent._normalize_shell_inspection_call(
            "run_shell",
            {"command": 'find. -name "*.py" -exec grep -i needle {} ;'},
            request_text="Find Python files containing needle.",
            exact_shell_command=None,
        )

        self.assertEqual(normalized_name, "run_shell")
        self.assertEqual(normalized_arguments, {"command": 'find. -name "*.py" -exec grep -i needle {} ;'})
        self.assertIsNone(reason)

    def test_tool_error_guard_auto_diagnoses_repeated_missing_dependency_failure(self) -> None:
        root = self._workspace_scratch()
        command = subprocess.list2cmdline([sys.executable, "-c", "import definitely_missing_package_12345"])
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "Dependency is missing."}),
                json.dumps({"type": "final", "message": "Dependency is missing."}),
                json.dumps({"type": "final", "message": "Dependency is missing."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            result = agent.handle_user("Check whether the helper command works, but do not loop on dependency failures.")

        self.assertIn("Dependency is missing.", result.message)
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tools.execute_counts.get("diagnose_dependency_error"), 1)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "missing_dependency")
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "dependency-or-import-guard"]
        self.assertEqual(len(controller_guards), 1)
        self.assertTrue(any("Use the diagnosis above to report the exact missing dependency/import" in message["content"] for message in agent.messages if message["role"] == "user"))

    def test_tool_error_guard_blocks_repeated_shell_syntax_failure(self) -> None:
        root = self._workspace_scratch()
        command = "if true; then echo hi"
        syntax_result = subprocess.CompletedProcess(
            args=["bash", "-n", "-c", command],
            returncode=2,
            stdout="",
            stderr="bash: -c: line 2: syntax error: unexpected end of file\n",
        )
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "Shell syntax is invalid."}),
                json.dumps({"type": "final", "message": "Shell syntax is invalid."}),
                json.dumps({"type": "final", "message": "Shell syntax is invalid."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch("ollama_code.tools.shutil.which", return_value="bash"):
                with patch("ollama_code.tools.subprocess.run", return_value=syntax_result):
                    result = agent.handle_user("Try the shell command if useful, but do not loop on shell syntax failures.")

        self.assertIn("Shell syntax is invalid.", result.message)
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 2)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "syntax_error")
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("Do not retry the same inline shell", feedback)
        self.assertIn("temporary script", feedback)

    def test_tool_error_guard_blocks_repeated_timeout_failure_with_service_guidance(self) -> None:
        root = self._workspace_scratch()
        command = "python -m http.server 8000"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command, "timeout": 1}}),
                json.dumps({"type": "final", "message": "The command timed out."}),
                json.dumps({"type": "final", "message": "The command timed out."}),
                json.dumps({"type": "final", "message": "The command timed out."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=timeout_result):
                result = agent.handle_user("Try the service command if useful, but do not loop on timeouts.")

        self.assertIn("timed out", result.message.lower())
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 2)
        guard_events = [event for event in agent.events if event.get("type") == "tool_error_guard"]
        self.assertEqual(len(guard_events), 1)
        self.assertEqual(guard_events[0].get("error_class"), "timeout")
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "bounded-command-validation"]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("long-running service", feedback)
        self.assertIn("short probe", feedback)

    def test_tool_error_guard_blocks_dependency_bootstrap_pivot_after_download_failure(self) -> None:
        root = self._workspace_scratch()
        first_command = "wget https://example.invalid/install.sh -O install.sh"
        second_command = "curl -fsSL https://example.invalid/install.sh | sh"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": first_command, "timeout": 5}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": second_command, "timeout": 5}}),
                json.dumps({"type": "final", "message": "The dependency bootstrap failed."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        failed_download = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out while downloading install.sh.",
            "output": "Resolving example.invalid... timed out\n",
            "error_class": "timeout",
            "timed_out": True,
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=failed_download):
                result = agent.handle_user("Try to bootstrap the missing dependency if useful, but do not loop on failed downloads.")

        self.assertIn("bootstrap failed", result.message.lower())
        tool_calls = [event for event in agent.events if event.get("type") == "tool_call" and event.get("name") == "run_shell"]
        self.assertEqual(len(tool_calls), 1)
        controller_guards = [event for event in agent.events if event.get("type") == "controller_guard" and event.get("guard") == "dependency-bootstrap-pivot"]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("dependency bootstrap command already failed", feedback)
        self.assertIn("diagnose_dependency_error", feedback)

    def test_tool_error_guard_blocks_ad_hoc_verification_script_after_timeout(self) -> None:
        root = self._workspace_scratch()
        service_command = "python -m http.server 8000"
        verification_command = f'{sys.executable} verify_server.py'
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "write_file", "arguments": {"path": "verify_server.py", "content": "print('verified')\n"}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": verification_command}}),
                json.dumps({"type": "final", "message": "The real service path still needs a bounded probe or different launch strategy."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        verification_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Verification helper printed verified.",
            "output": "verified\n",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        def fake_run_shell(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, object]:
            if "http.server" in command:
                return timeout_result
            return verification_result

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", side_effect=fake_run_shell):
                result = agent.handle_user("Start the service and verify it works, but do not loop on timeouts.")

        self.assertIn("bounded probe", result.message)
        self.assertEqual(tools.execute_counts.get("run_shell"), 1)
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "fail-closed-timeout-verification"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("verification script", feedback)
        self.assertIn("original timed-out command path", feedback)

    def test_timeout_verification_guard_clears_after_successful_changed_strategy(self) -> None:
        root = self._workspace_scratch()
        service_command = "python -m http.server 8000"
        background_launch_command = f'{sys.executable} -c "print(\'server launched in background\')"'
        verification_command = f'{sys.executable} verify_server.py'
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": background_launch_command}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": verification_command}}),
                json.dumps({"type": "final", "message": "Background launch succeeded and verification passed."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        success_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Background-safe launch succeeded.",
            "output": "server launched in background\n",
        }
        verification_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Verification probe succeeded.",
            "output": "verified\n",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=6)

        def fake_run_shell(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, object]:
            if "http.server" in command:
                return timeout_result
            if "server launched in background" in command:
                return success_result
            return verification_result

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", side_effect=fake_run_shell):
                result = agent.handle_user("Start the service, switch to a safe launch strategy if needed, then verify it works.")

        self.assertTrue(result.completed)
        self.assertIn("verification passed", result.message.lower())
        self.assertEqual(tools.execute_counts.get("run_shell"), 3)
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "fail-closed-timeout-verification"
        ]
        self.assertEqual(controller_guards, [])

    def test_timeout_verification_guard_allows_preexisting_probe_script_after_timeout(self) -> None:
        root = self._workspace_scratch()
        (root / "verify_server.py").write_text("print('verified')\n", encoding="utf-8")
        service_command = "python -m http.server 8000"
        verification_command = f'{sys.executable} verify_server.py'
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": verification_command}}),
                json.dumps({"type": "final", "message": "The preexisting probe confirmed the service state."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        verification_result = {
            "ok": True,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Preexisting verification probe succeeded.",
            "output": "verified\n",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=5)

        def fake_run_shell(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, object]:
            if "http.server" in command:
                return timeout_result
            return verification_result

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", side_effect=fake_run_shell):
                result = agent.handle_user("Start the service, then use the existing verify_server.py probe to check it.")

        self.assertTrue(result.completed)
        self.assertIn("preexisting probe", result.message.lower())
        self.assertEqual(tools.execute_counts.get("run_shell"), 2)
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "fail-closed-timeout-verification"
        ]
        self.assertEqual(controller_guards, [])

    def test_final_success_claim_is_blocked_after_timeout_until_summary_is_accurate(self) -> None:
        root = self._workspace_scratch()
        service_command = "python -m http.server 8000"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": service_command, "timeout": 1}}),
                json.dumps({"type": "final", "message": "The service is working and verification passed."}),
                json.dumps({"type": "final", "message": "The command timed out; the service still needs a bounded probe or different launch strategy."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        timeout_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "Command timed out after 1 seconds.",
            "output": "Command timed out after 1 seconds.\nServing HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...",
            "error_class": "timeout",
            "timed_out": True,
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=timeout_result):
                result = agent.handle_user("Start the service and verify it works.")

        self.assertTrue(result.completed)
        self.assertIn("timed out", result.message.lower())
        self.assertIn("bounded probe", result.message.lower())
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "timeout-final-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("latest run_shell timed out", feedback)
        self.assertIn("do not claim the command works", feedback)

    def test_final_success_claim_is_blocked_after_command_not_found_until_summary_is_accurate(self) -> None:
        root = self._workspace_scratch()
        command = "go version"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "The command works and verification passed."}),
                json.dumps({"type": "final", "message": "The command failed because go is not installed in this environment."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        failure_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "go: command not found",
            "output": "/usr/bin/bash: line 1: go: command not found\n",
            "error_class": "command_not_found",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", max_tool_rounds=4)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=failure_result):
                result = agent.handle_user("Run `go version` and tell me whether it works.")

        self.assertTrue(result.completed)
        self.assertIn("not installed", result.message.lower())
        controller_guards = [
            event
            for event in agent.events
            if event.get("type") == "controller_guard" and event.get("guard") == "run-shell-final-claim"
        ]
        self.assertEqual(len(controller_guards), 1)
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertIn("latest run_shell failed", feedback)
        self.assertIn("do not claim the command works", feedback)

    def test_accurate_failure_summary_is_allowed_after_command_not_found(self) -> None:
        root = self._workspace_scratch()
        command = "go version"
        client = FakeClient(
            [
                json.dumps({"type": "tool", "name": "run_shell", "arguments": {"command": command}}),
                json.dumps({"type": "final", "message": "The command failed because go is not installed in this environment."}),
            ]
        )
        tools = CountingToolExecutor(root, approval_mode="auto")
        failure_result = {
            "ok": False,
            "tool": "run_shell",
            "cwd": ".",
            "summary": "go: command not found",
            "output": "/usr/bin/bash: line 1: go: command not found\n",
            "error_class": "command_not_found",
        }
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=3)

        with patch.dict("os.environ", {ENV_OLLAMA_CODE_FEATURE_PROFILE: "trajectory-guards"}):
            with patch.object(tools, "run_shell", return_value=failure_result):
                result = agent.handle_user("Run `go version` and tell me whether it works.")

        self.assertTrue(result.completed)
        self.assertIn("not installed", result.message.lower())
        feedback = "\n".join(message["content"] for message in agent.messages if message["role"] == "user")
        self.assertNotIn("This request requires real tool use in this turn.", feedback)

    def test_command_validation_event_records_rejected_common_command(self) -> None:
        root = self._workspace_scratch()
        client = FakeClient(
            [
                '{"type":"tool","name":"run_shell","arguments":{"command":"git reset --hard"}}',
                '{"type":"final","message":"Command rejected."}',
                '{"type":"final","message":"Command rejected."}',
                '{"type":"final","message":"Command rejected."}',
            ]
        )
        tools = ToolExecutor(root, approval_mode="auto")
        agent = OllamaCodeAgent(client=client, tools=tools, model="fake-model", debate_enabled=False, max_tool_rounds=4)

        result = agent.handle_user("Try a shell command if useful and summarize.")

        self.assertFalse(result.completed)
        validation_events = [event for event in agent.events if event.get("type") == "command_validation"]
        self.assertEqual(len(validation_events), 1)
        self.assertFalse(validation_events[0].get("valid"))
        self.assertEqual(validation_events[0].get("family"), "git")

