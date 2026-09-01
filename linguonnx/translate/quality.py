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

Per-language flags
------------------

Everything above is **whole-model**: one ``chrf_vs_ref`` per entry, so a
model is good or bad for all of its languages at once. That is the wrong
shape for the failure this section exists for.

``madlad400-3b-mt`` advertises Chuvash. Ask it for ``en -> cv`` and it
answers in Russian: ``"Good day, my friend."`` comes back as
``"Добрый день, мой друг."``. It is not a routing bug - the ``<2cv>``
control piece is SentencePiece id 222, distinct from ``<2ru>``'s 118, so the
model is being asked the right question and answering in the wrong language.
MADLAD is a good model; it is a good model that cannot write Chuvash. One
chrF number for 400 languages cannot say that, and a whole-model flag would
throw away the other 399.

So an entry may carry ``language_flags``, a dict of BCP-47 tag -> evidence:

.. code-block:: json

    "language_flags": {
      "cv": {"reason": "answers in ru",
             "evidence": "en->cv 'Good day, my friend.' -> 'Добрый день, мой друг.'",
             "detector": "glotlid=ru",
             "date": "2026-08-10",
             "method": "int8-sweep",
             "side": "target"}
    }

``reason``
    One short clause, written to be read inside a sentence: the failure
    message reads "MADLAD covers cv but is flagged: answers in ru".
``evidence``
    The observation itself - input, output, direction. A flag with no
    observation behind it does not belong here; this field is where that
    rule is enforced by eye.
``detector`` / ``date`` / ``method``
    How the wrong language was identified, when, and in which sweep. Enough
    to re-run the check and to decide whether it has gone stale.
``side``
    Which direction the flag covers. See below.

Which side a flag applies to
----------------------------

**A flag applies to the side its** ``side`` **key names, and to nothing
else.** ``"target"`` (the default when the key is absent) means the model
must not be asked to *write* this language; ``"source"`` means it must not
be asked to *read* it; ``"both"`` means neither.

Target and source failures are different failures and one does not imply the
other. The MADLAD case is target-side: the model clearly has Chuvash text in
its training data - it just cannot generate it on request, and ``cv -> en``
may well work. Defaulting an observation of one direction into a claim about
the other would delete coverage nobody measured, which is the same sin as
publishing a coverage claim nobody measured.

The default is ``"target"`` because that is the direction a language-token
mechanism can fail in at all: the target is *selected* (a ``<2xx>`` piece, a
forced BOS id), while the source is merely *read*. Committed entries state
``side`` explicitly anyway - ``test_registry_invariants`` requires it - so
the default only ever covers a hand-built dict in a caller's own code.

Precision counterparts
----------------------

**A language flag on one precision applies to its** ``counterpart_model_id``
**too**, and :func:`language_flag_reason_for` unions the two.

Unlike the int8-gap check, this is not a comparison between the precisions:
it is a claim about the weights both precisions were exported from. A model
that answers Chuvash in Russian does so because Chuvash is not really in the
model, and quantising it to int8 does not teach it any. The two entries are
one export at two bit depths, so a flag observed on either is a fact about
both.

The consequence worth stating: a flag curated on only one precision cannot
be escaped by routing to the other. Both entries are still populated
explicitly, because a curated flag is meant to be readable in the registry
where a human will look for it, not inferred by a reader who knows the rule.

What a flag is *not*
--------------------

It is not a whole-model flag. :func:`is_quality_flagged` and
``exclude_flagged=`` are deliberately left alone: dropping all of MADLAD
over one language would cost the ~400 it does fine. The routing integration
- skip the flagged language, fall through to the next candidate model, keep
the model - is a separate change.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

__all__ = [
    "QUALITY_ABSOLUTE_FLOOR", "QUALITY_INT8_GAP", "LANGUAGE_FLAG_SIDES",
    "counterpart_model_id",
    "quality_flag_reasons", "quality_flag_reasons_for", "is_quality_flagged",
    "language_flag_reason", "flagged_languages", "language_flag_reason_for",
    "is_language_flagged", "entry_target_languages",
    "flagged_target_languages",
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


# ---------------------------------------------------------------------------
# Per-language flags
# ---------------------------------------------------------------------------

#: The values ``language_flags[lang]["side"]`` may take. ``"target"`` is the
#: default when the key is absent - see the module docstring for why the
#: two directions are not allowed to imply each other.
LANGUAGE_FLAG_SIDES: Tuple[str, ...] = ("target", "source", "both")


def _flag_covers(flag: Dict, side: str) -> bool:
    """Whether a single flag dict covers the direction ``side`` asks about."""
    declared = flag.get("side", "target")
    if declared not in LANGUAGE_FLAG_SIDES:
        raise ValueError(
            f"language flag side {declared!r} is not one of "
            f"{LANGUAGE_FLAG_SIDES}")
    return declared == "both" or declared == side


def language_flag_reason(entry: Dict, lang: str,
                         side: str = "target") -> Optional[str]:
    """Why ``entry`` is flagged for ``lang``, or ``None`` when it is not.

    Pure function over one registry entry dict - no I/O, no registry lookup,
    so the counterpart rule is *not* applied here. Use
    :func:`language_flag_reason_for` when you want it.

    ``side`` is the direction being asked about: ``"target"`` for "may this
    model be asked to write ``lang``", ``"source"`` for "may it be asked to
    read it". A flag declaring the other side does not answer.

    The returned string is the ``reason`` clause, meant to be read inside a
    sentence: ``f"{model_id} covers {lang} but is flagged: {reason}"``.
    """
    if side not in LANGUAGE_FLAG_SIDES:
        raise ValueError(f"side must be one of {LANGUAGE_FLAG_SIDES}")
    flag = (entry.get("language_flags") or {}).get(lang)
    if not flag or not _flag_covers(flag, side):
        return None
    return flag.get("reason") or "flagged, no reason recorded"


def flagged_languages(entry: Dict, side: str = "target") -> FrozenSet[str]:
    """Every language ``entry`` is flagged for in the direction ``side``."""
    return frozenset(
        lang for lang in (entry.get("language_flags") or {})
        if language_flag_reason(entry, lang, side) is not None)


def language_flag_reason_for(model_id: str, lang: str,
                             side: str = "target",
                             registry: Optional[Dict[str, Dict]] = None
                             ) -> Optional[str]:
    """:func:`language_flag_reason` for a registry id, **including its
    precision counterpart's flags**.

    A wrong-language answer is a property of the weights, not of the
    quantisation, so a flag curated on ``madlad400-3b-mt`` holds for
    ``madlad400-3b-mt-int8`` and the other way round. Following the shape of
    :func:`quality_flag_reasons_for`: ``registry`` defaults to the full
    committed translate registry.
    """
    if registry is None:
        from linguonnx.model_manager import list_models
        registry = list_models(kind="translate")
    entry = registry.get(model_id)
    if entry is not None:
        reason = language_flag_reason(entry, lang, side)
        if reason is not None:
            return reason
    counterpart = registry.get(counterpart_model_id(model_id))
    if counterpart is not None:
        return language_flag_reason(counterpart, lang, side)
    return None


def is_language_flagged(model_id: str, lang: str, side: str = "target",
                        registry: Optional[Dict[str, Dict]] = None) -> bool:
    """Whether ``model_id`` carries a language flag for ``lang``."""
    return language_flag_reason_for(model_id, lang, side, registry) is not None


def entry_target_languages(entry: Dict) -> FrozenSet[str]:
    """Every BCP-47 tag ``entry`` claims it can **write**.

    Mirrors the three coverage shapes
    :func:`linguonnx.translate.models.capability_from_entry` reads: a
    bilingual ``pair`` writes its second element, a directional multilingual
    model writes ``tgt_languages``, and an any-to-any one writes everything
    in ``languages``.
    """
    from linguonnx.translate.graph import normalize_tag

    pair = entry.get("pair")
    if pair:
        return frozenset({normalize_tag(pair[1])})
    tgt = entry.get("tgt_languages")
    if tgt is not None:
        return frozenset(normalize_tag(code) for code in tgt)
    return frozenset(normalize_tag(code) for code in entry.get("languages", ()))


def flagged_target_languages(entries: Dict[str, Dict],
                             registry: Optional[Dict[str, Dict]] = None
                             ) -> Dict[str, Tuple[str, ...]]:
    """``lang -> reasons`` for every language ``entries`` covers as a target
    and **every** covering model is flagged for.

    This is the honest half of a coverage count. A language one model is
    flagged for and another is not stays usable and is absent here; a
    language only ``madlad400-3b-mt`` reaches, and which MADLAD is flagged
    for, is not usable as a target at all no matter how the router chooses.

    ``entries`` is the selection being counted (a ``Translator``'s own
    models). ``registry`` is where counterparts are looked up for the
    precision rule and defaults to ``entries`` itself, so a caller who
    filtered to ``precision="int8"`` can still pass the full registry and
    have the fp32 entry's flags apply.
    """
    if registry is None:
        registry = entries
    reasons: Dict[str, Tuple[str, ...]] = {}
    unflagged: set = set()
    for model_id, entry in entries.items():
        for lang in entry_target_languages(entry):
            reason = language_flag_reason_for(model_id, lang, "target",
                                              registry)
            if reason is None:
                unflagged.add(lang)
            else:
                reasons[lang] = reasons.get(lang, ()) + \
                    (f"{model_id}: {reason}",)
    return {lang: found for lang, found in sorted(reasons.items())
            if lang not in unflagged}
