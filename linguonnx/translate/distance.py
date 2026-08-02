"""Linguistic distance between languages, used to rank pivot candidates.

This is an **optional** capability. linguonnx stays lightweight, so the
`orthography2ipa <https://github.com/TigreGotico/orthography2ipa>`_ package is
not a hard dependency: install ``linguonnx[distance]`` to get it. Without it
every function here reports "no distance information" and the routing layer
keeps its curated table order.

Why this metric
---------------

Pivot choice is a *linguistic* question - which intermediate language loses the
least on the way through - so the metric has to measure language similarity.

``langcodes.tag_distance`` was measured and rejected. It is a CLDR
*locale-matching* score, built to decide which translation file to serve a
user, not how alike two languages are. It scores ``pt->es`` and ``pt->en``
identically (84 each) and calls ``pt->gl`` as distant as ``pt->en``. It cannot
rank pivots.

``orthography2ipa.distance.phonological_distance(...).combined`` gives the
orderings the routing layer needs, e.g. ``pt->es`` 0.21 against ``pt->en``
0.32, and ``es->eu`` 0.25 against ``en->eu`` 0.45.

Do **not** switch this to ``full_distance`` or ``ancestry_similarity``. Their
ancestry component is unreliable today: every Romance medieval stage in
orthography2ipa (``ca-x-medieval``, ``es-ES-x-medieval``, ``pt-PT-x-medieval``,
``roa-x-galaicopt``) carries an empty ancestry list, so Ibero-Romance languages
never meet at a shared ancestor. ``full_distance`` therefore ranks English
closer to Catalan than Spanish is. Check that data gap first if you ever want
to revisit this.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = ["available", "pair_distance", "clear_cache"]

#: Memo of every ``(a, b)`` pair already scored, including the misses. A pair
#: that orthography2ipa does not know is cached as ``None`` so an unknown tag
#: costs one lookup for the life of the process, not one per routing call.
_CACHE: Dict[Tuple[str, str], Optional[float]] = {}

_BACKEND = None  # (get, phonological_distance), or False once known missing


def _backend():
    """Import orthography2ipa once, lazily. ``False`` means "not installed"."""
    global _BACKEND
    if _BACKEND is None:
        try:
            from orthography2ipa import get
            from orthography2ipa.distance import phonological_distance
            _BACKEND = (get, phonological_distance)
        except Exception:
            _BACKEND = False
    return _BACKEND


def available() -> bool:
    """True when orthography2ipa can be imported."""
    return bool(_backend())


def clear_cache() -> None:
    """Drop the memo. For tests, and for reloading after an install."""
    _CACHE.clear()


def pair_distance(a: str, b: str) -> Optional[float]:
    """Phonological distance between two language tags, lower = closer.

    Returns ``None`` when the backend is absent or when either tag is unknown
    to it. ``None`` means *no information*, never *identical*: the caller must
    fall back to table order for that candidate rather than treat it as 0.
    """
    if a == b:
        return 0.0
    key = (a, b) if a <= b else (b, a)
    if key in _CACHE:
        return _CACHE[key]
    backend = _backend()
    if not backend:
        _CACHE[key] = None
        return None
    get, phonological_distance = backend
    try:
        spec_a, spec_b = get(key[0]), get(key[1])
        # get() may return None as well as raise, depending on the tag shape.
        score = None if spec_a is None or spec_b is None else float(
            phonological_distance(spec_a, spec_b).combined)
    except Exception:
        score = None
    _CACHE[key] = score
    return score
