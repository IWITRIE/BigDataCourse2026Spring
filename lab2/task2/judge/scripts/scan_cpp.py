#!/usr/bin/env python3
"""Small source scanner for C++ submissions.

This is not a complete sandbox. It rejects common file/process/thread escape
surfaces before the code is compiled, while Docker provides the hard boundary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


FORBIDDEN_PATTERNS = [
    r"#\s*include\s*<fstream>",
    r"#\s*include\s*<filesystem>",
    r"#\s*include\s*<unistd\.h>",
    r"#\s*include\s*<sys/",
    r"\bsystem\s*\(",
    r"\bpopen\s*\(",
    r"\bfork\s*\(",
    r"\bexec[a-zA-Z_]*\s*\(",
    r"\bstd::ifstream\b",
    r"\bstd::ofstream\b",
    r"\bfreopen\s*\(",
    r"\bfopen\s*\(",
]


def fail(message: str) -> None:
    print(
        json.dumps(
            {
                "status": "failed",
                "error": message,
                "time_sec": None,
                "rmse_base": None,
                "rmse": None,
                "valid": False,
            }
        )
    )
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: scan_cpp.py <solution.cpp>")
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            fail(f"forbidden C++ construct matched: {pattern}")


if __name__ == "__main__":
    main()
