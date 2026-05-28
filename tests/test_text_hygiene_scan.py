from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import text_hygiene_scan


class TextHygieneScanTests(unittest.TestCase):
    def test_current_repo_has_no_bom_or_unexpected_crlf(self) -> None:
        result = text_hygiene_scan.scan()

        self.assertEqual(result["findings"], [])

    def test_scan_paths_flags_bom_and_non_powershell_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.txt"
            sample.write_bytes(b"\xef\xbb\xbfline\r\n")
            powershell = root / "script.ps1"
            powershell.write_bytes(b"Write-Output 'ok'\r\n")

            findings = text_hygiene_scan.scan_paths([sample, powershell], root)

        self.assertEqual(
            findings,
            [
                {"path": "sample.txt", "finding": "utf-8-bom"},
                {"path": "sample.txt", "finding": "crlf"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
