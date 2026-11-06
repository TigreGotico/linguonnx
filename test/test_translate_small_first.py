"""With no budget set, routing prefers small models and keeps every language.

A cap somebody chose stays a filter. These tests pin both halves, because the
second one is what stops the first from overruling an operator.
"""

import pytest

from linguonnx import limits
from linguonnx.limits import DEFAULT_MAX_MODEL_MB
from linguonnx.model_manager import DEFAULT_MAX_DOWNLOAD_MB, list_models
from linguonnx.translate import NoRouteError, load_translator
from linguonnx.translate import graph as graph_module
from linguonnx.translate.graph import Capability, TranslationGraph

MADLAD = "madlad400-3b-mt-int8"
ALL_LANGUAGES = 606


@pytest.fixture(autouse=True)
def _no_operator_budget(monkeypatch, tmp_path):
    monkeypatch.delenv("LINGUONNX_MAX_MODEL_MB", raising=False)
    monkeypatch.delenv("LINGUONNX_MAX_DOWNLOAD_MB", raising=False)
    monkeypatch.setattr("linguonnx.model_manager.is_cached",
                        lambda model_id, kind="lid": False)


def _operator_caps_models(monkeypatch, value):
    monkeypatch.setenv("LINGUONNX_MAX_MODEL_MB", str(value))
    monkeypatch.setattr(limits, "MAX_MODEL_MB", value)
    monkeypatch.setattr(graph_module, "MAX_MODEL_MB", value)


def test_the_default_cap_sits_between_the_models_it_was_chosen_around():
    registry = list_models(kind="translate")
    assert registry["m2m100-418M-int8"]["size_mb"] < DEFAULT_MAX_MODEL_MB
    assert registry["aina-translator-en-ca-int8"]["size_mb"] < DEFAULT_MAX_MODEL_MB
    assert registry[MADLAD]["size_mb"] > DEFAULT_MAX_MODEL_MB
    assert DEFAULT_MAX_MODEL_MB < DEFAULT_MAX_DOWNLOAD_MB


def test_the_default_is_the_small_cap_with_the_fallback_on():
    translator = load_translator()
    assert translator.max_model_mb == DEFAULT_MAX_MODEL_MB
    assert translator.oversize_fallback is True
    assert translator.count_cached_as_free is False


def test_a_pair_a_small_model_serves_never_reaches_madlad():
    route = load_translator().route("pt", "ru")
    assert route.model_ids == ("m2m100-418M-int8",)
    assert route.waived_size_cap is None


def test_a_pair_only_madlad_serves_still_routes():
    route = load_translator().route("en", "tyv")
    assert route.model_ids == (MADLAD,)
    assert route.waived_size_cap == DEFAULT_MAX_MODEL_MB


def test_no_language_is_lost():
    assert len(load_translator().available_languages) == ALL_LANGUAGES


def test_a_warm_cache_does_not_turn_the_preference_off(monkeypatch):
    monkeypatch.setattr("linguonnx.model_manager.is_cached",
                        lambda model_id, kind="lid": model_id == MADLAD)
    assert load_translator().route("pt", "ru").model_ids == ("m2m100-418M-int8",)


def test_a_regional_spanish_stays_on_the_small_spanish_model():
    assert load_translator().route("en", "es-ES").model_ids == ("opus-mt-en-es-int8",)


def test_a_cap_from_the_caller_is_a_filter():
    translator = load_translator(max_model_mb=500)
    assert translator.oversize_fallback is False
    assert translator.count_cached_as_free is True
    assert not translator.can_translate("en", "tyv")


def test_a_cap_from_the_operator_is_a_filter(monkeypatch):
    _operator_caps_models(monkeypatch, 500)
    translator = load_translator()
    assert translator.max_model_mb == 500
    assert translator.oversize_fallback is False
    assert not translator.can_translate("en", "tyv")


def test_a_caller_can_ask_for_the_fallback_under_their_own_cap():
    translator = load_translator(max_model_mb=500, oversize_fallback=True)
    assert translator.route("en", "tyv").waived_size_cap == 500


def test_a_caller_can_turn_the_fallback_off_under_the_default_cap():
    translator = load_translator(oversize_fallback=False)
    assert translator.max_model_mb == DEFAULT_MAX_MODEL_MB
    with pytest.raises(NoRouteError) as raised:
        translator.route("en", "tyv")
    assert f"{DEFAULT_MAX_MODEL_MB} MB size cap" in str(raised.value)


def test_a_per_call_cap_is_a_filter_on_a_default_translator():
    translator = load_translator()
    assert not translator.can_translate("en", "tyv", max_model_mb=500)
    assert "tyv" not in translator.graph.languages_under(500)


def test_a_per_call_cap_keeps_the_fallback_when_the_call_asks_for_it():
    route = load_translator().route("en", "tyv", max_model_mb=500,
                                    oversize_fallback=True)
    assert route.waived_size_cap == 500


def test_a_constructor_fallback_also_covers_a_per_call_cap():
    translator = load_translator(oversize_fallback=True)
    assert translator.route("en", "tyv", max_model_mb=500).waived_size_cap == 500


def test_a_lower_download_budget_lowers_the_cap_and_stops_the_fallback(monkeypatch):
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "1000")
    translator = load_translator()
    assert translator.max_model_mb == 1000
    assert not translator.can_translate("en", "tyv")


def test_a_higher_download_budget_leaves_the_preference_alone(monkeypatch):
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "16000")
    translator = load_translator()
    assert translator.max_model_mb == DEFAULT_MAX_MODEL_MB
    assert translator.route("pt", "ru").model_ids == ("m2m100-418M-int8",)
    assert translator.route("en", "tyv").model_ids == (MADLAD,)


def test_a_disabled_download_budget_keeps_the_preference(monkeypatch):
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "0")
    translator = load_translator()
    assert translator.max_model_mb == DEFAULT_MAX_MODEL_MB
    assert translator.route("en", "tyv").model_ids == (MADLAD,)


def test_a_named_model_is_not_bound_by_the_default_cap():
    translator = load_translator(models=[MADLAD])
    assert translator.max_model_mb is None
    assert translator.route("en", "tyv").waived_size_cap is None


def _dedicated(model_id, size_mb, flagged=False, pair=("az", "ru")):
    return Capability(model_id=model_id, arch="marian", license="Apache-2.0",
                      license_tier="permissive", size_mb=size_mb,
                      directions=frozenset({pair}), flagged=flagged)


def test_a_flagged_model_under_the_cap_gives_way_to_a_clean_larger_one():
    graph = TranslationGraph([_dedicated("weak", 300, flagged=True),
                              _dedicated("clean", 3000)])
    route = graph.route("az", "ru")
    assert route.model_ids == ("clean",)
    assert route.waived_size_cap == DEFAULT_MAX_MODEL_MB


def test_the_smallest_clean_model_wins_that_escalation():
    graph = TranslationGraph([_dedicated("weak", 300, flagged=True),
                              _dedicated("clean", 3000),
                              _dedicated("cleaner-but-huge", 7000)])
    assert graph.route("az", "ru").model_ids == ("clean",)


def test_a_flagged_model_stays_when_only_flagged_models_are_larger():
    graph = TranslationGraph([_dedicated("weak", 300, flagged=True),
                              _dedicated("also-weak", 3000, flagged=True)])
    route = graph.route("az", "ru")
    assert route.model_ids == ("weak",)
    assert route.waived_size_cap is None


def test_a_filter_keeps_the_flagged_model_a_cap_left_alone():
    graph = TranslationGraph([_dedicated("weak", 300, flagged=True),
                              _dedicated("clean", 3000)], max_model_mb=500)
    assert graph.route("az", "ru").model_ids == ("weak",)


def test_a_clean_chain_under_the_cap_beats_escalating_past_a_flagged_direct_model():
    graph = TranslationGraph([_dedicated("weak", 300, flagged=True),
                              _dedicated("clean-big", 3000),
                              _dedicated("az-en", 100, pair=("az", "en")),
                              _dedicated("en-ru", 100, pair=("en", "ru"))])
    route = graph.route("az", "ru")
    assert route.model_ids == ("az-en", "en-ru")
    assert route.waived_size_cap is None
