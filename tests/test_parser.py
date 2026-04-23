from __future__ import annotations

import unittest

from ollama_code.agent import extract_json_response


class ParserTests(unittest.TestCase):
    def test_extracts_plain_json(self) -> None:
        payload = extract_json_response('{"type":"final","message":"ok"}')
        self.assertEqual(payload, {"type": "final", "message": "ok"})

    def test_extracts_json_from_fenced_block(self) -> None:
        payload = extract_json_response('```json\n{"type":"tool","name":"list_files","arguments":{}}\n```')
        self.assertEqual(payload, {"type": "tool", "name": "list_files", "arguments": {}})

    def test_strips_think_tags(self) -> None:
        payload = extract_json_response('<think>hidden</think>{"type":"final","message":"done"}')
        self.assertEqual(payload, {"type": "final", "message": "done"})

    def test_prefers_last_agent_payload_over_earlier_non_agent_json(self) -> None:
        payload = extract_json_response(
            'context {"foo":1}\n{"type":"tool","name":"list_files","arguments":{}}'
        )
        self.assertEqual(payload, {"type": "tool", "name": "list_files", "arguments": {}})

    def test_prefers_last_agent_payload_when_multiple_agent_objects_appear(self) -> None:
        payload = extract_json_response(
            '{"type":"final","message":"wrong"}\n{"type":"final","message":"right"}'
        )
        self.assertEqual(payload, {"type": "final", "message": "right"})
