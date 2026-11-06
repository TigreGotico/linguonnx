"""`NoRouteError` names the filter that dropped a model, and no other."""

import pytest

import linguonnx.translate as translate
from linguonnx import load_translator
from linguonnx.translate import NoRouteError


def entry(model_id, tgt, precision="int8", tier="permissive", chrf=None):
    return {"model_id": model_id, "arch": "marian", "license": "MIT",
            "license_tier": tier, "size_mb": 100, "precision": precision,
            "pair": ["en", tgt],
            "quality": {"corpus": "flores", "n": 100, "chrf_vs_ref": chrf}
            if chrf is not None else None}


REGISTRY = {e["model_id"]: e for e in [
    entry("kept-int8", "fr", chrf=70.0),
    entry("full-size", "de", precision="fp32"),
    entry("paid-int8", "es", tier="non-commercial"),
    entry("paid-fp32", "es", precision="fp32", tier="non-commercial"),
    entry("weak-int8", "it", chrf=10.0),
    entry("middling-int8", "pt", chrf=50.0),
]}


@pytest.fixture
def translator(monkeypatch):
    monkeypatch.setattr(translate, "list_models", lambda kind="lid": REGISTRY)
    return load_translator(exclude_flagged=True, min_chrf=60.0, pivot_ranking="table",
                           max_model_mb=None, count_cached_as_free=False)


@pytest.mark.parametrize("tgt, expected", [
    ("de", "model(s) left out by precision='int8' cover this pair: full-size; "
           "pass precision=None to use them"),
    ("es", "excluded non-commercial model(s) cover this pair: paid-fp32, "
           "paid-int8; pass include_noncommercial=True to use them"),
    ("it", "flagged model(s) cover this pair: weak-int8; pass "
           "exclude_flagged=False to use them"),
    ("pt", "model(s) below min_chrf=60.0 cover this pair: middling-int8; "
           "lower min_chrf to use them"),
])
def test_the_error_names_the_filter_that_dropped_the_model(translator, tgt, expected):
    with pytest.raises(NoRouteError) as caught:
        translator.route("en", tgt)
    assert str(caught.value).endswith(f"-- {expected}")


def test_a_kept_model_is_not_excluded(translator):
    assert translator.route("en", "fr").model_ids == ("kept-int8",)


def test_two_filters_on_one_pair_each_get_their_sentence(monkeypatch):
    registry = {**REGISTRY, "paid-de-int8": entry("paid-de-int8", "de",
                                                  tier="non-commercial")}
    monkeypatch.setattr(translate, "list_models", lambda kind="lid": registry)
    tx = load_translator(pivot_ranking="table", max_model_mb=None,
                         count_cached_as_free=False)
    with pytest.raises(NoRouteError) as caught:
        tx.route("en", "de")
    assert "non-commercial model(s) cover this pair: paid-de-int8;" in str(caught.value)
    assert "precision='int8' cover this pair: full-size;" in str(caught.value)


def test_a_models_list_does_not_call_a_permissive_model_non_commercial():
    tx = load_translator(models=["opus-mt-pt-en-int8"])
    with pytest.raises(NoRouteError) as caught:
        tx.route("en", "fr")
    message = str(caught.value)
    assert "model(s) not named in models= cover this pair: " in message
    assert "opus-mt-en-fr-int8" in message
    assert "non-commercial" not in message
