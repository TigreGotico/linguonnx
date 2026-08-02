#!/usr/bin/env python3
"""Execute every python code block in the README and in ``docs/``.

A code sample that does not run is worse than no sample: the reader copies it,
it fails, and the whole document loses its credibility. So the samples are
tests. This script finds every ```python fence in the markdown, runs the
blocks of one file in order inside one shared namespace, and fails on the
first exception.

Blocks are opt-out, not opt-in. Put an HTML comment on the line before the
fence to change how a block is treated::

    <!-- doc-check: skip needs a 5 GB MADLAD download -->
    <!-- doc-check: norun illustrative, not executable on its own -->

``skip`` means the code is correct but running it here would cost a large
download; the reason is printed so nobody forgets why. ``norun`` is for
fragments that are deliberately not a whole program.

Usage::

    python scripts/check_docs.py                # README.md + docs/*.md
    python scripts/check_docs.py docs/detect.md # just one file
    python scripts/check_docs.py --list         # show what would run
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

FENCE = re.compile(
    r"(?:<!--\s*doc-check:\s*(?P<directive>\w+)(?P<reason>[^>]*?)-->\s*\n)?"
    r"^```python\n(?P<code>.*?)^```",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Block:
    path: Path
    line: int
    code: str
    directive: Optional[str]
    reason: str


def blocks(path: Path) -> List[Block]:
    text = path.read_text(encoding="utf-8")
    out = []
    for match in FENCE.finditer(text):
        out.append(Block(
            path=path,
            line=text[:match.start("code")].count("\n") + 1,
            code=match.group("code"),
            directive=match.group("directive"),
            reason=(match.group("reason") or "").strip(),
        ))
    return out


def label(path: Path) -> str:
    """A repo-relative path when the file is inside the repo, else as given."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check(path: Path, verbose: bool) -> int:
    namespace = {"__name__": "__doc_check__"}
    failures = 0
    for block in blocks(path):
        where = f"{label(path)}:{block.line}"
        if block.directive == "norun":
            print(f"  norun {where}")
            continue
        if block.directive == "skip":
            print(f"  SKIP  {where}  ({block.reason or 'no reason given'})")
            continue
        try:
            exec(compile(block.code, where, "exec"), namespace)
        except Exception:
            failures += 1
            print(f"  FAIL  {where}")
            traceback.print_exc()
        else:
            print(f"  ok    {where}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--list", action="store_true",
                        help="list the blocks without running them")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    paths = args.files or [REPO_ROOT / "README.md"] + sorted(
        (REPO_ROOT / "docs").glob("*.md"))

    if args.list:
        for path in paths:
            for block in blocks(path):
                mark = block.directive or "run"
                print(f"{label(path)}:{block.line}\t{mark}")
        return 0

    failures = 0
    for path in paths:
        print(label(path))
        failures += check(path, args.verbose)
    print()
    print("all documented samples ran" if not failures
          else f"{failures} block(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
