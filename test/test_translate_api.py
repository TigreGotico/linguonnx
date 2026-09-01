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

    def factory(model_id, entry=None,
                enforce_download_budget=True, providers=None):
        # `Translator` passes `enforce_download_budget` and `providers` to
        # every `TranslationModel` it builds; this double stands in for that
        # constructor and has to have its shape.
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


def test_the_size_budget_is_settable_per_call_and_per_translator(tx):
    """Same shape as max_hops: constructor value, per-call override.

    The unset default is no longer "no cap": it is the cold-download budget,
    so routing cannot promise a model ``ensure_model_files`` will refuse to
    fetch. Every model in this fixture is far under it, so the rest of the
    test is unaffected.
    """
    from linguonnx.model_manager import DEFAULT_MAX_DOWNLOAD_MB

    assert tx.max_model_mb == DEFAULT_MAX_DOWNLOAD_MB
    assert tx.route("pt", "ru").model_ids == ("multi",)
    assert tx.route("pt", "ru", max_model_mb=500).model_ids == ("pt-en", "en-ru")
    frugal = Translator(REGISTRY, max_model_mb=500)
    assert frugal.max_model_mb == 500
    assert frugal.route("pt", "ru").model_ids == ("pt-en", "en-ru")
    # Omitting inherits; None lifts.
    assert frugal.route("pt", "ru", max_model_mb=None).model_ids == ("multi",)


def test_available_languages_follows_the_budget(monkeypatch):
    """What the translator advertises is what it can serve, cap included."""
    monkeypatch.setattr("linguonnx.model_manager.is_cached",
                        lambda model_id, kind="lid": False)
    frugal = Translator(REGISTRY, max_model_mb=500)
    assert "es" in Translator(REGISTRY).available_languages
    assert "es" not in frugal.available_languages   # only `multi` has Spanish
    assert frugal.can_translate("pt", "es") is False


def test_translating_under_a_budget_runs_the_chain(tx):
    out = tx.translate("olá", src="pt", tgt="ru", max_model_mb=500)
    assert out == "olá|pt-en:pt->en|en-ru:en->ru"


def test_a_cached_model_stays_usable_over_the_budget(monkeypatch):
    """The budget is on the download; `count_cached_as_free` says so."""
    monkeypatch.setattr("linguonnx.model_manager.is_cached",
                        lambda model_id, kind="lid": model_id == "multi")
    assert Translator(REGISTRY, max_model_mb=500).route(
        "pt", "ru").model_ids == ("multi",)
    strict = Translator(REGISTRY, max_model_mb=500, count_cached_as_free=False)
    assert strict.count_cached_as_free is False
    assert strict.route("pt", "ru").model_ids == ("pt-en", "en-ru")


def test_load_translator_takes_a_budget(monkeypatch):
    monkeypatch.setattr("linguonnx.model_manager.is_cached",
                        lambda model_id, kind="lid": False)
    tx = load_translator(max_model_mb=500)
    assert tx.max_model_mb == 500
    assert all(hop.size_mb <= 500 for hop in tx.route("pt", "ru").hops)
    assert tx.route("pt", "ru").download_size_mb < 500


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
    """M2M100 has no Basque. A flat default would silently drop it.

    MADLAD carries Basque directly, so under the default `fewest_hops` policy
    it now wins outright over the two-hop opus-mt chain - fewer hops always
    beats more, and MADLAD is Apache-2.0, same tier as opus-mt.
    """
    translator = load_translator()
    assert "eu" in translator.available_languages
    route = translator.route("pt", "eu")
    for hop in route.hops:
        assert not hop.model_id.startswith("m2m100") or "eu" not in (hop.src, hop.tgt)
    assert route.n_hops == 1
    assert route.model_ids == ("madlad400-3b-mt-int8",)
    # A dedicated bilingual chain is still available under `prefer="dedicated"`.
    dedicated_route = translator.route("pt", "eu", prefer="dedicated")
    assert dedicated_route.n_multilingual_hops == 0
    assert dedicated_route.hops[-1].tgt == "eu"


def test_english_to_portuguese_prefers_the_dedicated_model():
    assert load_translator().route("en", "pt").model_ids == ("opus-mt-en-pt-int8",)


def test_fp32_precision_can_be_selected():
    translator = load_translator(precision="fp32")
    assert all(e["precision"] == "fp32" for e in translator.models.values())


def test_unknown_model_id_is_rejected():
    with pytest.raises(ValueError):
        load_translator(models=["not-a-model"])


def test_norouteerror_names_excluded_noncommercial_models():
    """A language only a non-commercial model covers must not look unsupported.

    Kabuverdianu is reachable only through NLLB (CC-BY-NC-4.0), which the
    default graph leaves out so nobody inherits a non-commercial licence
    unasked. Excluding it silently is the wrong way to enforce that — the
    error has to name the model and the opt-in flag.
    """
    from linguonnx import load_translator
    from linguonnx.translate.graph import NoRouteError
    import pytest as _pytest

    tx = load_translator()
    with _pytest.raises(NoRouteError) as excinfo:
        tx.route("pt", "kea")
    message = str(excinfo.value)
    assert "non-commercial" in message
    assert "nllb" in message.lower()
    assert "include_noncommercial=True" in message


def test_norouteerror_stays_quiet_for_genuinely_unknown_languages():
    """No spurious licence hint when nothing covers the pair at all."""
    from linguonnx import load_translator
    from linguonnx.translate.graph import NoRouteError
    import pytest as _pytest

    tx = load_translator()
    with _pytest.raises(NoRouteError) as excinfo:
        tx.route("pt", "zzz")
    assert "non-commercial" not in str(excinfo.value)


def test_noncommercial_opt_in_reaches_kabuverdianu():
    from linguonnx import load_translator
    tx = load_translator(include_noncommercial=True)
    route = tx.route("pt", "kea")
    assert any("nllb" in hop.model_id for hop in route.hops)


def test_supplied_route_cannot_run_a_directional_model_backwards():
    """`route=` escapes the scoring policy, not correctness.

    A one-directional model (IndicTrans2 en->indic, the nos-coda pairs,
    liv4ever) has both languages in its code map, so a hand-built backwards
    hop tokenises and decodes happily and returns fluent text translated the
    wrong way round. Nothing would raise, which is why a supplied route is
    validated rather than trusted.
    """
    import pytest as _pytest
    from linguonnx.translate.graph import (Capability, Hop, InvalidRouteError,
                                           Route, TranslationGraph)

    one_way = Capability(
        model_id="fake-en-xx", arch="marian", license="apache-2.0",
        license_tier="permissive", size_mb=1,
        src_languages=frozenset({"en"}), tgt_languages=frozenset({"xx"}))
    graph = TranslationGraph([one_way])

    def hop(src, tgt):
        return Hop(model_id="fake-en-xx", src=src, tgt=tgt, arch="marian",
                   license="apache-2.0", license_tier="permissive",
                   size_mb=1, dedicated=True)

    forwards = Route(src="en", tgt="xx", hops=(hop("en", "xx"),))
    assert graph.validate_route(forwards) is forwards

    backwards = Route(src="xx", tgt="en", hops=(hop("xx", "en"),))
    with _pytest.raises(InvalidRouteError):
        graph.validate_route(backwards)
