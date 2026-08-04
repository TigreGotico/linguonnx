"""The quality field: what flags, what does not, and how selection honours it.

Two layers, tested separately:

- :mod:`linguonnx.translate.quality` is pure functions over registry-entry
  dicts - no I/O, no graph - so the flag policy itself is tested against
  small synthetic entries where every input is explicit.
- ``load_translator``/``_select_entries`` are tested against the real
  committed ``translate.json``, so a change to the actual measured numbers
  (or to which entries carry them) is caught here, not just in the policy
  unit tests.
"""

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import _select_entries, load_translator
from linguonnx.translate.quality import (QUALITY_ABSOLUTE_FLOOR,
                                         QUALITY_INT8_GAP,
                                         counterpart_model_id,
                                         is_quality_flagged,
                                         quality_flag_reasons,
                                         quality_flag_reasons_for)

REGISTRY = list_models(kind="translate")


# ---------------------------------------------------------------------------
# counterpart_model_id
# ---------------------------------------------------------------------------

def test_counterpart_strips_the_int8_suffix():
    assert counterpart_model_id("opus-mt-az-en-int8") == "opus-mt-az-en"


def test_counterpart_adds_the_int8_suffix():
    assert counterpart_model_id("opus-mt-az-en") == "opus-mt-az-en-int8"


# ---------------------------------------------------------------------------
# quality_flag_reasons - pure policy, synthetic entries
# ---------------------------------------------------------------------------

def _entry(precision, chrf_vs_ref=None, chrf_vs_fp32=None, n=100):
    quality = {}
    if chrf_vs_ref is not None:
        quality["chrf_vs_ref"] = chrf_vs_ref
    if chrf_vs_fp32 is not None:
        quality["chrf_vs_fp32"] = chrf_vs_fp32
    entry = {"precision": precision}
    if quality:
        quality.setdefault("corpus", "flores200-devtest")
        quality.setdefault("n", n)
        entry["quality"] = quality
    return entry


def test_unmeasured_entry_is_never_flagged():
    entry = _entry("fp32")
    assert quality_flag_reasons(entry, None) == ()


def test_agreement_only_entry_is_not_flagged_by_itself():
    # Only chrf_vs_fp32 recorded (no chrf_vs_ref) - exactly the shape used for
    # the Azerbaijani tr-az/en-az entries, where only agreement was measured.
    entry = _entry("int8", chrf_vs_fp32=74.8)
    assert quality_flag_reasons(entry, None) == ()


def test_below_absolute_floor_flags_regardless_of_precision():
    fp32_entry = _entry("fp32", chrf_vs_ref=QUALITY_ABSOLUTE_FLOOR - 0.1)
    reasons = quality_flag_reasons(fp32_entry, None)
    assert reasons
    assert "floor" in reasons[0]


def test_above_absolute_floor_with_no_counterpart_is_not_flagged():
    entry = _entry("fp32", chrf_vs_ref=QUALITY_ABSOLUTE_FLOOR + 10)
    assert quality_flag_reasons(entry, None) == ()


def test_int8_trailing_fp32_over_the_gap_is_flagged():
    fp32 = _entry("fp32", chrf_vs_ref=60.0)
    int8 = _entry("int8", chrf_vs_ref=60.0 - QUALITY_INT8_GAP - 0.1)
    reasons = quality_flag_reasons(int8, fp32)
    assert reasons
    assert "trails fp32" in reasons[0]


def test_int8_within_the_gap_is_not_flagged():
    fp32 = _entry("fp32", chrf_vs_ref=60.0)
    int8 = _entry("int8", chrf_vs_ref=60.0 - QUALITY_INT8_GAP + 0.1)
    assert quality_flag_reasons(int8, fp32) == ()


def test_fp32_side_is_never_gap_flagged_against_its_own_int8():
    # The gap check is asymmetric on purpose: it protects against int8
    # regressing relative to fp32, not the other way around - fp32 trailing
    # a (hypothetically better-scoring) int8 counterpart is not a quantisation
    # safety problem.
    int8 = _entry("int8", chrf_vs_ref=80.0)
    fp32 = _entry("fp32", chrf_vs_ref=80.0 - QUALITY_INT8_GAP - 5)
    assert quality_flag_reasons(fp32, int8) == ()


def test_both_checks_can_fire_together():
    fp32 = _entry("fp32", chrf_vs_ref=90.0)
    int8 = _entry("int8", chrf_vs_ref=QUALITY_ABSOLUTE_FLOOR - 5)
    reasons = quality_flag_reasons(int8, fp32)
    assert len(reasons) == 2


def test_quality_flag_reasons_for_missing_model_is_empty():
    assert quality_flag_reasons_for("does-not-exist", registry={}) == ()


# ---------------------------------------------------------------------------
# The real registry: which entries were actually measured, and how they flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", [
    "opus-mt-az-en",
    "m2m100_418M_en_hau_rel_news_ft",
    "m2m100_418M_en_hau_rel_news_ft-int8",
    # A live sweep flagged these two for looping multi-word phrases. Measured
    # on 100 MAFAND-MT test sentences (the corpus they were fine-tuned for -
    # neither language is in FLORES-200): chrF 26.1/25.5 for Ghomala->French
    # and 23.7/23.9 for Mossi->French, fp32/int8, with 3-8 of the 100 outputs
    # ending in a phrase loop. Upstream `transformers` fp32 loops on the same
    # inputs, so this is base-model quality and belongs here rather than in a
    # decode-parameter change.
    "m2m100_418M_bbj_fr_rel_news_ft",
    "m2m100_418M_bbj_fr_rel_news_ft-int8",
    "m2m100_418M_mos_fr_rel_news_ft",
    "m2m100_418M_mos_fr_rel_news_ft-int8",
])
def test_known_weak_models_are_flagged(model_id):
    assert is_quality_flagged(model_id, REGISTRY), (
        f"{model_id} scored below the absolute floor in the FLORES-200 "
        "re-measurement and must be flagged")


@pytest.mark.parametrize("model_id", [
    "opus-mt-en-es", "opus-mt-en-es-int8", "opus-mt-en-fr", "opus-mt-en-de",
])
def test_known_strong_models_are_not_flagged(model_id):
    assert not is_quality_flagged(model_id, REGISTRY)


@pytest.mark.parametrize("model_id", [
    # Agreement-only measurements (no chrf-vs-reference at n=100): the
    # quantitative flag must stay silent - the qualitative hallucination
    # finding for these lives in `notes`, not in this field, on purpose.
    "opus-mt-tr-az", "opus-mt-tr-az-int8",
    "opus-mt-en-az", "opus-mt-en-az-int8",
])
def test_agreement_only_registry_entries_are_not_quantitatively_flagged(model_id):
    assert not is_quality_flagged(model_id, REGISTRY)


@pytest.mark.parametrize("model_id", [
    "opus-mt-az-en", "opus-mt-az-en-int8",
    "opus-mt-tr-az", "opus-mt-tr-az-int8",
    "opus-mt-en-az", "opus-mt-en-az-int8",
    "m2m100_418M_en_hau_rel_news_ft", "m2m100_418M_en_hau_rel_news_ft-int8",
    "m2m100_418M_bbj_fr_rel_news_ft", "m2m100_418M_bbj_fr_rel_news_ft-int8",
    "m2m100_418M_mos_fr_rel_news_ft", "m2m100_418M_mos_fr_rel_news_ft-int8",
])
def test_hallucinating_models_carry_a_notes_caveat(model_id):
    notes = REGISTRY[model_id].get("notes") or ""
    assert "hallucinat" in notes.lower()


@pytest.mark.parametrize("model_id", sorted(REGISTRY))
def test_every_quality_field_has_a_sample_size_and_corpus(model_id):
    quality = REGISTRY[model_id].get("quality")
    if quality is None:
        return
    assert quality.get("n"), f"{model_id} quality data has no sample size"
    assert quality.get("corpus"), f"{model_id} quality data has no corpus"
    assert quality.get("metric"), f"{model_id} quality data has no metric"
    # Exactly one of the two numeric fields is the bare minimum; either is
    # acceptable, but a quality block with neither is meaningless.
    assert "chrf_vs_ref" in quality or "chrf_vs_fp32" in quality


# ---------------------------------------------------------------------------
# Selection: exclude_flagged / min_chrf, and explicit models= overriding both
# ---------------------------------------------------------------------------

def test_exclude_flagged_drops_the_weak_models_in_both_precisions():
    entries = _select_entries(None, False, None, exclude_flagged=True)
    # opus-mt-az-en's fp32 side was measured below the absolute floor, so it
    # is quantitatively flagged and dropped.
    assert "opus-mt-az-en" not in entries
    # Its int8 sibling only has an agreement-vs-fp32 number on file (n=100,
    # no reference measurement), so the quantitative flag never fires for
    # it - the hallucination caveat for it lives in `notes`, checked
    # separately in test_hallucinating_models_carry_a_notes_caveat.
    assert "opus-mt-az-en-int8" in entries
    assert "m2m100_418M_en_hau_rel_news_ft" not in entries
    assert "m2m100_418M_en_hau_rel_news_ft-int8" not in entries
    # Not flagged quantitatively - agreement-only entries stay.
    assert "opus-mt-tr-az" in entries
    # Strong pairs stay too.
    assert "opus-mt-en-es" in entries
    assert "opus-mt-en-es-int8" in entries


def test_exclude_flagged_can_leave_fp32_as_the_only_survivor_of_a_pair():
    # A synthetic case where only the int8 side trails: exclude_flagged with
    # precision=None should keep fp32 and drop int8, giving the caller "fall
    # back to fp32 automatically" through the existing filter surface.
    registry = {
        "strong-fp32": _entry("fp32", chrf_vs_ref=90.0),
        "strong-fp32-int8": _entry("int8", chrf_vs_ref=90.0 - QUALITY_INT8_GAP - 1),
    }
    for model_id, entry in registry.items():
        entry["model_id"] = model_id
        entry["license_tier"] = "permissive"
    from linguonnx.model_manager import list_models as real_list_models

    def fake_list_models(kind="translate"):
        return registry if kind == "translate" else real_list_models(kind=kind)

    import linguonnx.translate as translate_mod
    original = translate_mod.list_models
    translate_mod.list_models = fake_list_models
    try:
        entries = _select_entries(None, False, None, exclude_flagged=True)
    finally:
        translate_mod.list_models = original
    assert "strong-fp32" in entries
    assert "strong-fp32-int8" not in entries


def test_min_chrf_excludes_only_models_measured_below_it():
    entries = _select_entries(None, False, None, min_chrf=50.0)
    assert "opus-mt-az-en" not in entries  # measured 25.9, below 50
    assert "opus-mt-en-es" in entries       # measured 54.9, above 50
    # Unmeasured is not zero: never excluded by a floor it was never checked
    # against.
    assert "opus-mt-tr-az" in entries


def test_explicit_models_overrides_exclude_flagged_and_min_chrf():
    entries = _select_entries(None, False, ["opus-mt-az-en"],
                              exclude_flagged=True, min_chrf=100.0)
    assert set(entries) == {"opus-mt-az-en"}


def test_load_translator_exposes_exclude_flagged():
    translator = load_translator(precision=None, exclude_flagged=True)
    assert "opus-mt-az-en" not in translator._entries
    assert "opus-mt-en-es" in translator._entries


def test_load_translator_exposes_min_chrf():
    translator = load_translator(precision=None, min_chrf=50.0)
    assert "opus-mt-az-en" not in translator._entries


def test_load_translator_models_still_overrides_quality_filters():
    translator = load_translator(models=["opus-mt-az-en"],
                                 exclude_flagged=True, min_chrf=100.0)
    assert set(translator._entries) == {"opus-mt-az-en"}


def test_translator_quality_flag_reasons_matches_the_module_function():
    translator = load_translator(precision=None)
    assert translator.quality_flag_reasons("opus-mt-az-en") == \
        quality_flag_reasons_for("opus-mt-az-en")
    assert translator.quality_flag_reasons("opus-mt-en-es") == ()


@pytest.mark.parametrize("model_id", [
    "m2m100_418M_bbj_fr_rel_news_ft", "m2m100_418M_bbj_fr_rel_news_ft-int8",
    "m2m100_418M_mos_fr_rel_news_ft", "m2m100_418M_mos_fr_rel_news_ft-int8",
])
def test_the_looping_masakhane_models_say_the_loop_is_upstream(model_id):
    """The caveat has to name *what kind* of problem it is, not just that
    there is one.

    A caller reading "repeats phrases" reasonably reaches for
    `no_repeat_ngram_size`. The note has to say that the knob exists, that it
    does break the loops, and that it does not make the output correct -
    otherwise the honest annotation reads as an unfixed bug.
    """
    notes = REGISTRY[model_id].get("notes") or ""
    assert "no_repeat_ngram_size" in notes
    assert "transformers" in notes


def test_the_repetition_guard_is_still_off_by_default():
    """The two looping models must not have moved the default for everyone.

    `transformers` defaults `no_repeat_ngram_size` to 0 and every parity and
    chrF number in this registry was measured against it at 0. Turning it on
    globally to hide two weak fine-tunes would silently re-score ~200 models
    against numbers measured under different settings.
    """
    from linguonnx.translate.decode import GenerationConfig

    assert GenerationConfig().no_repeat_ngram_size == 0
