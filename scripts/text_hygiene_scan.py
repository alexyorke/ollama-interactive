from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

WINDOWS_TEXT_SUFFIXES = {".ps1", ".psm1", ".psd1"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo_root)
    return [repo_root / Path(item) for item in output.decode("utf-8").split("\0") if item]


def _is_binary(data: bytes) -> bool:
    return b"\0" in data


def scan_paths(paths: list[Path], repo_root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if _is_binary(data):
            continue
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        if data.startswith(b"\xef\xbb\xbf"):
            findings.append({"path": rel, "finding": "utf-8-bom"})
        if path.suffix.lower() not in WINDOWS_TEXT_SUFFIXES and b"\r\n" in data:
            findings.append({"path": rel, "finding": "crlf"})
    return findings


def scan(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repo_root()
    findings = scan_paths(tracked_files(root), root)
    return {"ok": not findings, "findings": findings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan tracked repo text files for BOM and unexpected CRLF endings.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)
    result = scan()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("text hygiene scan passed")
    else:
        print("text hygiene scan failed")
        for item in result["findings"]:
            print(json.dumps(item, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
