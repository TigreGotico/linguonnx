"""Public API tests. Models are stubbed, so nothing downloads and no ONNX
session is built - what is under test is which model gets asked, in what order.
"""

import pytest

from linguonnx import load_translator
from linguonnx.translate import Translator
from linguonnx.translate.graph import Hop, NoRouteError, Route

REGISTRY = {
    "multi": {
        "model_id": "multi", "hf_repo": "x/multi", "arch": "m2m100",
        "languages": ["en", "pt", "es", "ru"], "license": "MIT",
        "license_tier": "permissive", "size_mb": 1200, "precision": "int8",
    },
    "nc-multi": {
        "model_id": "nc-multi", "hf_repo": "x/nc", "arch": "nllb",
        # No Basque here on purpose: eu must come from the bilingual model, the
        # same shape the real registry has once NLLB is left out.
        "languages": ["eng_Latn", "por_Latn", "rus_Cyrl"],
        "license": "CC-BY-NC-4.0", "license_tier": "non-commercial",
        "size_mb": 1800, "precision": "int8",
    },
    "pt-en": {
        "model_id": "pt-en", "hf_repo": "x/pt-en", "arch": "marian",
        "pair": ["pt", "en"], "license": "Apache-2.0",
        "license_tier": "permissive", "size_mb": 350, "precision": "int8",
    },
    "en-eu": {
        "model_id": "en-eu", "hf_repo": "x/en-eu", "arch": "marian",
        "pair": ["en", "eu"], "license": "Apache-2.0",
        "license_tier": "permissive", "size_mb": 350, "precision": "int8",
    },
    "en-ru": {
        "model_id": "en-ru", "hf_repo": "x/en-ru", "arch": "marian",
        "pair": ["en", "ru"], "license": "Apache-2.0",
        "license_tier": "permissive", "size_mb": 350, "precision": "int8",
    },
}


class FakeModel:
    """Records every hop it is asked to do, and tags the text so a chain shows."""

    def __init__(self, model_id, entry):
        self.model_id = model_id
        from linguonnx.translate.models import capability_from_entry
        self.capability = capability_from_entry(entry)
        self.calls = []

    def translate(self, text, src, tgt, config=None, target_token=None):
        self.calls.append((text, src, tgt))
        return f"{text}|{self.model_id}:{src}->{tgt}"


@pytest.fixture
def tx(monkeypatch):
    made = {}

    def factory(model_id, entry=None):
        made.setdefault(model_id, FakeModel(model_id, entry or REGISTRY[model_id]))
        return made[model_id]

    monkeypatch.setattr("linguonnx.translate.TranslationModel", factory)
    translator = Translator(REGISTRY)
    translator._made = made
    return translator


# --- routing through the public surface -----------------------------------

def test_translate_returns_a_string(tx):
    assert tx.translate("hi", src="en", tgt="pt").startswith("hi|")


def test_return_route_exposes_the_hops(tx):
    out, route = tx.translate("bom dia", src="pt", tgt="eu", return_route=True)
    assert isinstance(route, Route)
    assert route.n_hops == 2
    assert route.model_ids == ("pt-en", "en-eu")
    assert out.count("|") == 2, "a two-hop chain must run two models"


def test_available_languages_is_the_union(tx):
    assert {"en", "pt", "es", "ru", "eu"} <= tx.available_languages


def test_route_inspects_without_translating(tx):
    route = tx.route("pt", "eu")
    assert route.model_ids == ("pt-en", "en-eu")
    assert not tx._made, "route() must not load any model"


def test_routes_lists_alternatives_with_licences(tx):
    found = tx.routes("pt", "ru")
    assert len(found) > 1
    assert found[0].model_ids == tx.route("pt", "ru").model_ids
    assert all(route.licenses for route in found)


# --- the escape hatches ----------------------------------------------------

def test_a_supplied_route_is_executed_verbatim(tx):
    """Even when it is not the top-ranked one."""
    ranked = tx.routes("pt", "ru")
    best, alternative = ranked[0], next(r for r in ranked
                                        if r.model_ids != ranked[0].model_ids)
    assert alternative.model_ids != best.model_ids

    out, used = tx.translate("ola", route=alternative, return_route=True)
    assert used.model_ids == alternative.model_ids
    assert [m for m in tx._made] == list(alternative.model_ids)
    for hop in alternative.hops:
        assert f"{hop.model_id}:{hop.src}->{hop.tgt}" in out


def test_a_supplied_route_ignores_the_policy(tx):
    """A hand-built 2-hop route runs even under prefer='fewest_hops'."""
    detour = Route("pt", "ru", (
        Hop("pt-en", "pt", "en", "marian", "Apache-2.0", "permissive", 350, True),
        Hop("en-ru", "en", "ru", "marian", "Apache-2.0", "permissive", 350, True),
    ))
    assert tx.route("pt", "ru").n_hops == 1
    out, used = tx.translate("ola", route=detour, return_route=True)
    assert used.n_hops == 2
    assert out.count("|") == 2


def test_model_pins_one_model_and_bypasses_routing(tx):
    out, used = tx.translate("ola", src="pt", tgt="ru", model="multi",
                             return_route=True)
    assert used.model_ids == ("multi",)
    assert used.prefer == "pinned"
    assert list(tx._made) == ["multi"]


def test_pinning_a_bilingual_model_needs_no_src_or_tgt(tx):
    out, used = tx.translate("bom dia", model="pt-en", return_route=True)
    assert used.hops[0].src == "pt" and used.hops[0].tgt == "en"


def test_pinning_a_model_that_cannot_do_the_pair_raises(tx):
    with pytest.raises(NoRouteError):
        tx.translate("ola", src="pt", tgt="eu", model="multi")


def test_translate_needs_some_way_to_decide(tx):
    with pytest.raises(ValueError):
        tx.translate("ola")


# --- policy / cap plumbing -------------------------------------------------

def test_policy_is_settable_per_call_and_per_translator(tx):
    assert tx.route("pt", "ru").n_hops == 1
    assert tx.route("pt", "ru", prefer="dedicated").n_hops == 2
    dedicated = Translator(REGISTRY, prefer="dedicated")
    assert dedicated.route("pt", "ru").n_hops == 2
    assert dedicated.prefer == "dedicated"


def test_max_hops_is_settable_per_call_and_per_translator(tx):
    assert tx.max_hops == 2
    assert tx.route("pt", "eu").n_hops == 2
    with pytest.raises(NoRouteError):
        tx.route("pt", "eu", max_hops=1)
    strict = Translator(REGISTRY, max_hops=1)
    assert strict.max_hops == 1
    with pytest.raises(NoRouteError):
        strict.route("pt", "eu")


# --- the shipped registry --------------------------------------------------

def test_default_graph_is_permissive_and_int8():
    translator = load_translator()
    assert translator.models, "the default graph must not be empty"
    for entry in translator.models.values():
        assert entry["license_tier"] == "permissive"
        assert entry["precision"] == "int8"
    assert "m2m100-418M-int8" in translator.models
    assert "nllb-600M-int8" not in translator.models, \
        "NLLB is CC-BY-NC and must be opt-in"


def test_non_commercial_models_are_opt_in():
    translator = load_translator(include_noncommercial=True)
    assert "nllb-600M-int8" in translator.models


def test_basque_is_routed_away_from_m2m100_in_the_real_registry():
    """M2M100 has no Basque. A flat default would silently drop it."""
    translator = load_translator()
    assert "eu" in translator.available_languages
    route = translator.route("pt", "eu")
    for hop in route.hops:
        assert not hop.model_id.startswith("m2m100") or "eu" not in (hop.src, hop.tgt)
    assert route.model_ids == ("opus-mt-pt-en-int8", "opus-mt-en-eu-int8")


def test_english_to_portuguese_prefers_the_dedicated_model():
    assert load_translator().route("en", "pt").model_ids == ("opus-mt-en-pt-int8",)


def test_fp32_precision_can_be_selected():
    translator = load_translator(precision="fp32")
    assert all(e["precision"] == "fp32" for e in translator.models.values())


def test_unknown_model_id_is_rejected():
    with pytest.raises(ValueError):
        load_translator(models=["not-a-model"])
