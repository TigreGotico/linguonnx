"""Measured translation quality, and the flag policy built on it.

Where it comes from
--------------------

A model's ``quality`` field, when present, comes from an actual re-measurement
against a human reference corpus, not a guess and not the library's old n=5-20
hand-written, exact-match-scored parity numbers - a campaign that turned out
to be noise-sized (``opus-mt-az-en`` beam4 read 40%, then 100%, then 14%
across three sample sizes on the same model). See ``docs/routing.md`` for the
full write-up of why chrF-vs-reference is the metric.

FLORES-200 devtest is the corpus wherever the language is in it. It is not
always: Ghomala' and Mossi are in no FLORES release, and
``m2m100_418M_bbj_fr_rel_news_ft``/``m2m100_418M_mos_fr_rel_news_ft`` are
measured on MAFAND-MT test instead. ``corpus`` says which, per entry - see
"Corpora are not interchangeable" below for what that does and does not buy.

The registry entry carries, when measured:

``corpus``
    Which reference corpus, e.g. ``"flores200-devtest"`` or
    ``"mafand-test"``. Read it before comparing two scores; nothing in this
    module does that for you.
``metric``
    ``"chrf"`` for chrF against the `corpus` human reference (the headline
    number - "is this precision's output actually good"), or
    ``"chrf_agreement_int8_vs_fp32"`` for chrF of int8's output scored
    against fp32's own output (a secondary diagnostic - "do the two
    precisions agree with each other" - recorded only when a
    against-the-reference number was not measured for that entry, so its
    presence never gets mistaken for the headline).
``mode``
    Decoding mode the number was measured under, ``"greedy"`` or ``"beam4"``.
``max_new_tokens`` / ``length_penalty``
    The rest of the decode configuration, when recorded. ``mode`` alone does
    not pin a score down: :class:`~linguonnx.translate.decode.GenerationConfig`
    caps generation at 128 tokens by default while several of these models'
    own ``generation_config.json`` says 512, and a cap that truncates long
    outputs moves chrF. Older entries predate these two keys and were taken
    at the library defaults of the day; new measurements should record them
    so the number can be reproduced from the repo.
``n``
    Sample size. Always shown next to the score - an n=5 or n=20 score reads
    very differently from an n=100 one, and this campaign's worst failure
    mode was a score with no visible sample size next to it.
``chrf_vs_ref``
    This precision's own chrF against the ``corpus`` human reference. Present only
    when actually measured against the reference; a model can be genuinely
    unmeasured, and that must never be confused with "measured and bad" - so
    its absence, not a zero or a guess, is how "not measured" is spelled.
``chrf_vs_fp32``
    chrF of this precision's output against the *other* precision's output
    (agreement, not quality). May accompany ``chrf_vs_ref`` as a secondary
    number, or stand alone when only agreement was measured.

A missing ``quality`` key entirely, or one with neither ``chrf_vs_ref`` nor
``chrf_vs_fp32``, means "not measured" - :func:`quality_flag_reasons` never
flags an unmeasured entry, because a flag is a claim about a number that was
actually seen.

Flag policy
-----------

Two independent checks, either sufficient on its own:

- **Absolute floor** (:data:`QUALITY_ABSOLUTE_FLOOR`): either precision's
  chrF-vs-reference below 40 flags the entry, regardless of the other
  precision's score. This is what should have caught the Azerbaijani
  opus-mt trio and ``m2m100_418M_en_hau`` from the start - a weak base
  model is a weak base model in both precisions.
- **int8 gap** (:data:`QUALITY_INT8_GAP`): int8's chrF-vs-reference trailing
  fp32's by more than 2 points flags the int8 entry. Every pair actually
  measured in the FLORES-200 campaign - weak (Azerbaijani, Hausa) and strong
  (Helsinki en-es/en-fr/en-de) alike - showed int8 within ~0.6 chrF of fp32
  against the reference, so 2.0 leaves headroom before flagging while still
  catching a real regression.

Corpora are not interchangeable
-------------------------------

Both checks above compare ``chrf_vs_ref`` as a bare number, and so does
``min_chrf=`` in :func:`linguonnx.translate._select_entries`. Neither reads
``corpus``. chrF is not calibrated across corpora - a hard low-resource
reference and an easy high-resource one do not put "good" at the same number -
so a floor applied across a mixed registry is a rough instrument, not a fair
comparison.

It is left rough on purpose rather than made corpus-aware, because
corpus-awareness would need a per-corpus floor, and there is no principled way
to set one for a corpus measured on two models. What the field does buy is
that a *human* can always tell the scores apart, and that the flag message
names the corpus it is quoting, so nobody reads 23.9 on ``mafand-test`` as if
it were 23.9 on FLORES-200. If a third corpus ever arrives, revisit this.

Flagging never excludes a model from the registry - the "publish everything"
rule holds regardless of quality. It only informs :func:`_select_entries`
callers (``exclude_flagged=``, ``min_chrf=``) who choose to filter at
runtime, exactly like the existing ``precision=``/``max_model_mb=`` knobs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = [
    "QUALITY_ABSOLUTE_FLOOR", "QUALITY_INT8_GAP", "counterpart_model_id",
    "quality_flag_reasons", "quality_flag_reasons_for", "is_quality_flagged",
]

#: chrF-vs-reference below this, in either precision, is a weak-model flag
#: independent of quantisation. See module docstring.
QUALITY_ABSOLUTE_FLOOR = 40.0

#: How many chrF-vs-reference points int8 may trail fp32 by before the int8
#: entry is flagged. See module docstring.
QUALITY_INT8_GAP = 2.0


def counterpart_model_id(model_id: str) -> str:
    """The registry id of the same export in the other precision.

    Precision variants are published as two registry entries sharing a model
    id except for a literal ``-int8`` suffix - see
    ``scripts/sync_registry.py``'s ``translate_entries``, which is the only
    place that suffix is minted. Always returns an id to *look up*; whether
    that id exists in a given registry is for the caller to check.
    """
    if model_id.endswith("-int8"):
        return model_id[: -len("-int8")]
    return f"{model_id}-int8"


def quality_flag_reasons(entry: Dict, counterpart_entry: Optional[Dict]) -> Tuple[str, ...]:
    """Why ``entry`` is quality-flagged, or ``()`` when it is not.

    Pure function over two registry entry dicts (this model's and its
    precision counterpart's, or ``None`` when the counterpart is not in the
    registry/selection) - no I/O, no graph, so it is trivial to unit test
    against synthetic entries.
    """
    quality = entry.get("quality")
    if not quality or "chrf_vs_ref" not in quality:
        return ()
    chrf = quality["chrf_vs_ref"]
    reasons = []
    if chrf < QUALITY_ABSOLUTE_FLOOR:
        reasons.append(
            f"chrF-vs-reference {chrf:.1f} is below the "
            f"{QUALITY_ABSOLUTE_FLOOR:.0f} floor "
            f"({quality.get('corpus', 'unknown corpus')}, n={quality.get('n', '?')})")
    if entry.get("precision") == "int8" and counterpart_entry:
        counterpart_quality = counterpart_entry.get("quality")
        if counterpart_quality and "chrf_vs_ref" in counterpart_quality:
            fp32_chrf = counterpart_quality["chrf_vs_ref"]
            gap = fp32_chrf - chrf
            if gap > QUALITY_INT8_GAP:
                reasons.append(
                    f"int8 chrF-vs-reference {chrf:.1f} trails fp32's "
                    f"{fp32_chrf:.1f} by {gap:.1f}, over the "
                    f"{QUALITY_INT8_GAP:.1f} threshold")
    return tuple(reasons)


def quality_flag_reasons_for(model_id: str,
                             registry: Optional[Dict[str, Dict]] = None) -> Tuple[str, ...]:
    """:func:`quality_flag_reasons` for a registry id, looking up its own
    counterpart. ``registry`` defaults to the full committed translate
    registry; pass the dict a :class:`~linguonnx.translate.Translator` was
    built from to ask "as this Translator sees it" instead.
    """
    if registry is None:
        from linguonnx.model_manager import list_models
        registry = list_models(kind="translate")
    entry = registry.get(model_id)
    if entry is None:
        return ()
    counterpart = registry.get(counterpart_model_id(model_id))
    return quality_flag_reasons(entry, counterpart)


def is_quality_flagged(model_id: str, registry: Optional[Dict[str, Dict]] = None) -> bool:
    """Whether ``model_id`` carries any :func:`quality_flag_reasons`."""
    return bool(quality_flag_reasons_for(model_id, registry))
