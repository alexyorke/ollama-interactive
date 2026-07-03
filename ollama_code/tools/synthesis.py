from __future__ import annotations

import re
from typing import Any


def human_test_name(name: str) -> str:
    text = re.sub(r"^test_?", "", name.strip())
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or name


def select_test_spec_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in examples:
        symbol = str(item.get("symbol") or "")
        if symbol not in grouped:
            grouped[symbol] = []
            order.append(symbol)
        grouped[symbol].append(item)
    if len(examples) <= limit:
        return examples
    selected: list[dict[str, Any]] = []
    per_symbol_limit = max(1, min(8, (limit + max(1, len(order)) - 1) // max(1, len(order))))
    cursors = {symbol: 0 for symbol in order}
    while len(selected) < limit:
        progressed = False
        for symbol in order:
            if len(selected) >= limit:
                break
            cursor = cursors[symbol]
            if cursor >= len(grouped[symbol]) or cursor >= per_symbol_limit:
                continue
            selected.append(grouped[symbol][cursor])
            cursors[symbol] += 1
            progressed = True
        if progressed:
            continue
        for symbol in order:
            while cursors[symbol] < len(grouped[symbol]) and len(selected) < limit:
                selected.append(grouped[symbol][cursors[symbol]])
                cursors[symbol] += 1
        break
    return selected


def split_test_example(example: str) -> tuple[str, str, str]:
    if " -> " in example:
        expr, expected = example.split(" -> ", 1)
        return "value", expr.strip(), expected.strip()
    match = re.match(r"^(?P<expr>.+?)\s+raises\s+(?P<raises>[A-Za-z_][\w.]*)(?:\((?P<message>.*)\))?$", example)
    if match:
        expected = str(match.group("raises") or "").strip()
        message = str(match.group("message") or "").strip()
        if message:
            expected += f"({message})"
        return "raises", str(match.group("expr") or "").strip(), expected
    return "", "", ""


def test_example_probe_expressions(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        kind, expr, expected = split_test_example(str(item.get("example") or ""))
        if not kind or not expr or not expected:
            continue
        if re.search(r"\bself\.", expr) or re.search(r"\bself\.", expected):
            continue
        rows.append(
            {
                "kind": kind,
                "expr": expr,
                "expected": expected,
                "symbol": str(item.get("symbol") or ""),
                "line": int(item.get("line") or 1),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows
