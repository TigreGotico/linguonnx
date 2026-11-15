"""Cut a line of text into sentences for translation."""

from __future__ import annotations

import re
from typing import List, Tuple

#: Scripts that put no space between sentences.
UNSPACED_SCRIPTS = frozenset({"Jpan", "Hans", "Hant", "Thai"})

_CLOSERS = "\"'”’»)\\]」』）》】"
# The two lookbehinds keep `e.g.` and `Dr.` inside their sentence. CJK stops
# need no whitespace after them; every other stop does.
_SENTENCE_END = re.compile(
    rf"[.!?…؟۔।॥]+(?<!\w\.\w.)(?<![A-Z][a-z]\.)[{_CLOSERS}]*\s+"
    rf"|[。！？｡]+[{_CLOSERS}]*\s*")


def split_sentences(line: str) -> List[Tuple[str, str]]:
    """Cut ``line`` into ``(sentence, whitespace after it)`` pairs.

    Joining every pair gives ``line`` back. A cut needs a letter before it, so
    ``1.`` stays with the list item it numbers.
    """
    parts: List[Tuple[str, str]] = []
    start = 0
    for match in _SENTENCE_END.finditer(line):
        end = match.start() + len(match.group().rstrip())
        if match.end() < len(line) \
                and any(char.isalpha() for char in line[start:end]):
            parts.append((line[start:end], line[end:match.end()]))
            start = match.end()
    parts.append((line[start:], ""))
    return parts
