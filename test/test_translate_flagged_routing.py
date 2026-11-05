"""A flagged model serves a pair only when nothing else does.

A quality flag ranks a model last. A language flag removes the edge it names,
because the model answers that language in another one.
"""

import pytest

from linguonnx.translate import load_translator
from linguonnx.translate.graph import NoRouteError, TranslationGraph
from linguonnx.translate.models import capability_from_entry

BAD = {"corpus": "flores", "n": 100, "chrf_vs_ref": 10.0}


def entry(model_id, size_mb=100, **fields):
    return {"model_id": model_id, "arch": "marian", "license": "MIT",
            "license_tier": "permissive", "size_mb": size_mb, **fields}


def graph(*entries, **kwargs):
    return TranslationGraph([capability_from_entry(e) for e in entries],
                            pivot_ranking="table", **kwargs)


def test_a_flagged_dedicated_model_loses_to_an_unflagged_multilingual_one():
    route = graph(entry("bad", pair=["en", "ja"], quality=BAD),
                  entry("wide", size_mb=900, languages=["en", "ja", "fr"])
                  ).route("en", "ja")
    assert route.model_ids == ("wide",)
    assert route.n_flagged_hops == 0


@pytest.mark.parametrize("prefer", ["fewest_hops", "dedicated"])
def test_two_clean_hops_beat_one_flagged_hop(prefer):
    route = graph(entry("bad", pair=["en", "ja"], quality=BAD),
                  entry("en-fr", pair=["en", "fr"]),
                  entry("fr-ja", pair=["fr", "ja"]),
                  pivot_preference=["fr"]).route("en", "ja", prefer=prefer)
    assert route.model_ids == ("en-fr", "fr-ja")


def test_a_flagged_model_still_serves_a_pair_nothing_else_covers():
    route = graph(entry("bad", pair=["en", "ja"], quality=BAD)).route("en", "ja")
    assert route.model_ids == ("bad",)
    assert route.n_flagged_hops == 1
    assert "flagged" in str(route.hops[0])


def test_a_flagged_leg_loses_inside_a_pivot():
    route = graph(entry("bad", pair=["en", "fr"], quality=BAD),
                  entry("wide", size_mb=900, languages=["en", "fr"]),
                  entry("fr-ja", pair=["fr", "ja"]),
                  pivot_preference=["fr"]).route("en", "ja")
    assert route.model_ids == ("wide", "fr-ja")


def test_a_flagged_model_loses_a_leg_to_a_larger_dedicated_one():
    route = graph(entry("bad", pair=["en", "fr"], quality=BAD),
                  entry("good", size_mb=900, pair=["en", "fr"]),
                  entry("fr-ja", pair=["fr", "ja"]),
                  pivot_preference=["fr"]).route("en", "ja")
    assert route.model_ids == ("good", "fr-ja")


def test_the_int8_gap_reads_the_other_precision_from_the_registry():
    registry = {"m": entry("m", precision="fp32",
                           quality={**BAD, "chrf_vs_ref": 60.0}),
                "m-int8": entry("m-int8", pair=["en", "ja"], precision="int8",
                                quality={**BAD, "chrf_vs_ref": 50.0})}
    assert capability_from_entry(registry["m-int8"], registry).flagged
    assert not capability_from_entry(registry["m-int8"]).flagged


LANGS = ["en", "cv", "ru"]
TARGET_FLAG = {"cv": {"reason": "answers in ru"}}


def test_a_target_flag_removes_the_edge_and_the_error_quotes_it():
    built = graph(entry("wide", languages=LANGS, language_flags=TARGET_FLAG))
    with pytest.raises(NoRouteError,
                       match="wide covers cv but is flagged: answers in ru"):
        built.route("en", "cv")


def test_a_target_flag_keeps_the_rest_of_the_model():
    built = graph(entry("wide", languages=LANGS, language_flags=TARGET_FLAG))
    assert built.route("cv", "en").model_ids == ("wide",)
    assert built.route("en", "ru").model_ids == ("wide",)
    assert "cv" in built.languages


def test_a_target_flag_is_not_a_pivot():
    built = graph(entry("wide", languages=["en", "cv"], language_flags=TARGET_FLAG),
                  entry("cv-ru", pair=["cv", "ru"]), pivot_preference=["cv"])
    with pytest.raises(NoRouteError):
        built.route("en", "ru")
    assert built.route("cv", "ru").model_ids == ("cv-ru",)


def test_another_model_still_serves_a_flagged_language():
    route = graph(entry("wide", languages=LANGS, language_flags=TARGET_FLAG),
                  entry("en-cv", size_mb=900, pair=["en", "cv"])).route("en", "cv")
    assert route.model_ids == ("en-cv",)


def test_a_source_flag_removes_only_the_reading_edge():
    built = graph(entry("wide", languages=LANGS, language_flags={
        "cv": {"reason": "reads it as ru", "side": "source"}}))
    assert built.route("en", "cv").model_ids == ("wide",)
    with pytest.raises(NoRouteError, match="reads it as ru"):
        built.route("cv", "en")


def test_a_language_flagged_on_both_sides_is_not_advertised():
    built = graph(entry("wide", languages=LANGS, language_flags={
        "cv": {"reason": "no", "side": "both"}}))
    assert built.languages == {"en", "ru"}


def test_the_other_precisions_language_flag_counts():
    registry = {"m": entry("m", languages=LANGS, language_flags=TARGET_FLAG),
                "m-int8": entry("m-int8", languages=LANGS)}
    capability = capability_from_entry(registry["m-int8"], registry)
    assert capability.barred_targets == {"cv": "answers in ru"}
    assert not capability.covers("en", "cv")
    assert capability.covers("en", "ru")


@pytest.fixture(scope="module")
def translator():
    return load_translator()


def test_english_to_japanese_leaves_the_flagged_marian_model(translator):
    assert translator.route("en", "ja").model_ids == ("m2m100-418M-int8",)


@pytest.mark.parametrize("src", ["en", "es", "es-ES"])
def test_chuvash_has_no_route_that_answers_in_russian(translator, src):
    with pytest.raises(NoRouteError, match="flagged: answers in ru"):
        translator.route(src, "cv")


def test_chuvash_is_still_read(translator):
    assert translator.route("cv", "en").model_ids == ("madlad400-3b-mt-int8",)


@pytest.mark.parametrize("src, tgt, model", [
    ("bbj", "fr", "m2m100_418M_bbj_fr_rel_news_ft-int8"),
    ("mos", "fr", "m2m100_418M_mos_fr_rel_news_ft-int8"),
])
def test_a_flagged_model_keeps_the_pair_only_it_covers(translator, src, tgt, model):
    route = translator.route(src, tgt)
    assert route.model_ids == (model,)
    assert route.n_flagged_hops == 1


def test_exclude_flagged_stays_a_hard_exclusion():
    with pytest.raises(NoRouteError):
        load_translator(exclude_flagged=True).route("bbj", "fr", max_hops=1)


def test_a_pinned_model_names_the_flag():
    tx = load_translator(model="madlad400-3b-mt-int8")
    with pytest.raises(NoRouteError, match="flagged: answers in ru"):
        tx.translate("hello", src="en", tgt="cv")
