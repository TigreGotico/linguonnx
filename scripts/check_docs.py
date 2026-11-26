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

A ``print(...)`` line with a ``# `` comment on the same line, or on the next
line, is a value check: the printed text, or the ``repr`` of the printed
values, must equal the comment. A comment on the next line continues on the
lines below that start with ``#  ``. ``...`` in
the comment matches any text. A note after `` — `` or after three spaces is
not part of the value.

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


PRINT = re.compile(r"^(?P<indent>\s*)print\((?P<args>.*?)\)\s*(?:#\s?(?P<comment>.*))?$")


def expected_value(comment: str) -> str:
    return re.split(r" — |   +", comment.strip(), maxsplit=1)[0].rstrip()


DECIMAL = re.compile(r"-?\d+\.\d+")


def matches(printed: str, expected: str) -> bool:
    """``...`` matches any text. A decimal in the comment matches a printed
    decimal that rounds to it, so ``0.998`` matches ``0.998534``."""
    pattern, decimals = "", []
    for k, part in enumerate(expected.split("...")):
        if k:
            pattern += ".*?"
        last = 0
        for m in DECIMAL.finditer(part):
            pattern += re.escape(part[last:m.start()]) + r"(-?\d+\.\d+(?:e-?\d+)?)"
            decimals.append(m.group())
            last = m.end()
        pattern += re.escape(part[last:])
    found = re.fullmatch(pattern, printed, re.DOTALL)
    if found is None:
        return False
    return all(round(float(got), len(want.split(".")[1])) == float(want)
               for got, want in zip(found.groups(), decimals))


def checks(block: Block) -> List[tuple]:
    """(line, expression, expected) for each value check in the block."""
    lines = block.code.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = PRINT.match(line)
        if not m:
            continue
        comment = m.group("comment")
        j = i + 1
        if comment is None and j < len(lines) and lines[j].lstrip().startswith("# "):
            comment = lines[j].lstrip()[2:]
            j += 1
            while j < len(lines) and lines[j].lstrip().startswith("#  "):
                comment += " " + lines[j].lstrip()[1:].strip()
                j += 1
        if comment is not None:
            out.append((block.line + i, m.group("args"), expected_value(comment)))
    return out


def instrument(block: Block) -> str:
    """The block with each checked print replaced by a call to ``_doc_print``."""
    lines = block.code.splitlines()
    checked = {line - block.line: expected for line, _, expected in checks(block)}
    for i, expected in checked.items():
        m = PRINT.match(lines[i])
        lines[i] = f"{m.group('indent')}_doc_print({block.line + i}, {expected!r}, {m.group('args')})"
    return "\n".join(lines) + "\n"


class ValueMismatch(AssertionError):
    pass


def _doc_print(line, expected, *args):
    printed = " ".join(str(a) for a in args)
    print(printed)
    if not (matches(printed, expected) or matches(" ".join(repr(a) for a in args), expected)):
        raise ValueMismatch(f"line {line}: printed {printed!r}, the comment says {expected!r}")


def label(path: Path) -> str:
    """A repo-relative path when the file is inside the repo, else as given."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check(path: Path, verbose: bool) -> int:
    namespace = {"__name__": "__doc_check__", "_doc_print": _doc_print}
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
            exec(compile(instrument(block), where, "exec"), namespace)
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
                for line, expr, expected in (checks(block) if mark == "run" else []):
                    print(f"    {line}\t{expr}\t# {expected}")
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
