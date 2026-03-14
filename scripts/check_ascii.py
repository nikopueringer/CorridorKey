"""Check for non-ASCII characters in docs and source files.

Exits with code 1 and prints offending lines if any are found.
"""

import pathlib
import sys

EXTENSIONS = ("*.md", "*.py", "*.toml")
EXCLUDE = (".venv", "site", ".git", "__pycache__")


def main() -> int:
    hits: list[tuple[str, int, str]] = []

    for ext in EXTENSIONS:
        for path in pathlib.Path(".").rglob(ext):
            if any(part in str(path) for part in EXCLUDE):
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
                if any(ord(c) > 127 for c in line):
                    hits.append((str(path), i + 1, line))

    for filepath, lineno, line in hits:
        print(f"{filepath}:{lineno}: {line}")

    if hits:
        print(f"\n{len(hits)} non-ASCII line(s) found.")
        return 1

    print("All files are ASCII-clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
