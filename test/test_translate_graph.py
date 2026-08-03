"""Routing tests. No model is downloaded and no ONNX session is built here -
the graph is pure data, which is the point of building it first.
"""

import time

import pytest

from linguonnx.translate.graph import (Capability, Hop, InvalidRouteError,
                                       MalformedTagError, NoRouteError, Route,
                                       TranslationGraph, entry_runnability,
                                       normalize_tag)

# --- fixtures: a miniature registry ---------------------------------------

M2M = Capability(model_id="m2m", arch="m2m100", license="MIT",
                 license_tier="permissive", size_mb=1200,
                 languages=frozenset({"en", "pt", "es", "ca", "gl", "ru", "de"}))
NLLB = Capability(model_id="nllb", arch="nllb", license="CC-BY-NC-4.0",
                  license_tier="non-commercial", size_mb=1800,
                  languages=frozenset({"en", "pt", "es", "ca", "gl", "ru", "de", "eu"}))


def marian(src, tgt, size=350, license="Apache-2.0", tier="permissive"):
    return Capability(model_id=f"opus-{src}-{tgt}", arch="marian",
                      license=license, license_tier=tier, size_mb=size,
                      pair=(src, tgt))


BILINGUAL = [
    marian("en", "pt"), marian("pt", "en"),
    marian("en", "es"), marian("es", "en"),
    marian("en", "ru"), marian("ru", "en"),
    marian("en", "eu"), marian("eu", "en"),
    marian("es", "ca"), marian("ca", "es"),
    marian("es", "gl"), marian("gl", "es"),
]


@pytest.fixture
def graph():
    return TranslationGraph([M2M, NLLB] + BILINGUAL)


# --- direct resolution -----------------------------------------------------

def test_direct_hit_is_one_hop(graph):
    route = graph.route("en", "pt")
    assert route.n_hops == 1
    assert route.hops[0].model_id == "opus-en-pt"


def test_dedicated_beats_multilingual_for_the_same_pair(graph):
    """Same pair, same hop count: the model that *is* the pair wins."""
    assert graph.route("en", "ru").hops[0].model_id == "opus-en-ru"
    assert graph.route("en", "ru").hops[0].dedicated is True


def test_multilingual_used_when_no_dedicated_model_exists(graph):
    route = graph.route("pt", "ru")
    assert route.n_hops == 1
    assert route.hops[0].model_id == "m2m"


def test_permissive_licence_beats_non_commercial(graph):
    """m2m (MIT) and nllb (CC-BY-NC) both cover de->ru; m2m must win."""
    route = graph.route("de", "ru")
    assert route.hops[0].model_id == "m2m"
    assert route.license_tier == "permissive"


def test_smaller_model_breaks_a_tie():
    big = Capability(model_id="big", arch="m2m100", license="MIT",
                     license_tier="permissive", size_mb=9000,
                     languages=frozenset({"en", "pt"}))
    small = Capability(model_id="small", arch="m2m100", license="MIT",
                       license_tier="permissive", size_mb=1200,
                       languages=frozenset({"en", "pt"}))
    assert TranslationGraph([big, small]).route("en", "pt").hops[0].model_id == "small"


# --- hop counting ----------------------------------------------------------

def test_one_multilingual_hop_beats_two_bilingual_hops(graph):
    """pt->ru: m2m does it in one; opus pt->en->ru would take two."""
    route = graph.route("pt", "ru")
    assert route.n_hops == 1
    assert route.hops[0].model_id == "m2m"


def test_two_hops_when_no_single_model_covers_the_pair(graph):
    """eu is only in nllb (non-commercial) and the opus eu pair."""
    permissive = TranslationGraph([M2M] + BILINGUAL)
    route = permissive.route("pt", "eu")
    assert route.n_hops == 2
    assert route.pivots == ("en",)
    assert route.model_ids == ("opus-pt-en", "opus-en-eu")


def test_three_hops_are_never_returned_by_default():
    """ru->eu needs ru->en->eu; a third hop must never appear."""
    graph = TranslationGraph(BILINGUAL)
    route = graph.route("ru", "eu")
    assert route.n_hops == 2
    for candidate in graph.routes("ru", "eu"):
        assert candidate.n_hops <= 2


def test_no_route_error_when_unreachable(graph):
    with pytest.raises(NoRouteError):
        graph.route("en", "kv")  # Komi is in no model here


def test_no_route_error_when_the_cap_is_too_low(graph):
    permissive = TranslationGraph([M2M] + BILINGUAL)
    permissive.route("pt", "eu")  # fine at the default 2 hops
    with pytest.raises(NoRouteError):
        permissive.route("pt", "eu", max_hops=1)


def test_max_hops_one_still_allows_a_direct_model(graph):
    assert graph.route("en", "pt", max_hops=1).n_hops == 1


def test_max_hops_three_is_allowed():
    """3+ hops is discouraged, not forbidden - the caller may know better."""
    chain = [marian("a", "b"), marian("b", "c"), marian("c", "d")]
    graph = TranslationGraph(chain, pivot_preference=("b", "c"))
    with pytest.raises(NoRouteError):
        graph.route("a", "d")
    route = graph.route("a", "d", max_hops=3)
    assert route.n_hops == 3
    assert route.pivots == ("b", "c")


def test_same_language_is_not_a_route(graph):
    with pytest.raises(NoRouteError):
        graph.route("pt", "pt")


# --- pivot preference ------------------------------------------------------

def test_pivot_prefers_a_linguistically_closer_language_than_english():
    """gl->ca is Iberian: pivot through Spanish, not the default English."""
    caps = [marian("gl", "es"), marian("es", "ca"),
            marian("gl", "en"), marian("en", "ca")]
    graph = TranslationGraph(caps)
    route = graph.route("gl", "ca")
    assert route.pivots == ("es",)


def test_pivot_preference_is_data_not_hardcoded():
    """Reordering the preference list changes the pivot, with no code change."""
    caps = [marian("xx", "en"), marian("en", "yy"),
            marian("xx", "de"), marian("de", "yy")]
    assert TranslationGraph(caps).route("xx", "yy").pivots == ("en",)
    reordered = TranslationGraph(caps, pivot_preference=("de", "en"))
    assert reordered.route("xx", "yy").pivots == ("de",)


def test_basque_routes_away_from_m2m100(graph):
    """M2M100 has no Basque. Every hop touching eu must avoid it."""
    assert "eu" not in M2M.languages
    permissive = TranslationGraph([M2M] + BILINGUAL)
    for route in permissive.routes("pt", "eu"):
        for hop in route.hops:
            if "eu" in (hop.src, hop.tgt):
                assert hop.model_id != "m2m"


def test_basque_is_reachable_at_all(graph):
    permissive = TranslationGraph([M2M] + BILINGUAL)
    assert permissive.can_translate("pt", "eu")
    assert permissive.can_translate("es", "eu")


# --- policies --------------------------------------------------------------

def test_the_two_policies_pick_different_winners_from_one_candidate_set(graph):
    """The documented difference: pt->ru, one m2m hop versus two opus hops."""
    fewest = graph.route("pt", "ru", prefer="fewest_hops")
    dedicated = graph.route("pt", "ru", prefer="dedicated")
    assert fewest.n_hops == 1 and fewest.hops[0].model_id == "m2m"
    assert dedicated.n_hops == 2
    assert dedicated.model_ids == ("opus-pt-en", "opus-en-ru")
    # ...over the same candidates, ranked differently, not a different search.
    ids = {r.model_ids for r in graph.routes("pt", "ru", prefer="fewest_hops")}
    assert dedicated.model_ids in ids


def test_dedicated_policy_still_prefers_one_hop_for_the_same_pair(graph):
    """prefer='dedicated' must not make en->pt take a detour."""
    route = graph.route("en", "pt", prefer="dedicated")
    assert route.n_hops == 1 and route.hops[0].model_id == "opus-en-pt"


def test_dedicated_policy_still_capped_at_two_hops(graph):
    for route in graph.routes("pt", "ru", prefer="dedicated"):
        assert route.n_hops <= 2


def test_unknown_policy_is_rejected_at_construction_time():
    with pytest.raises(ValueError):
        TranslationGraph([M2M], prefer="cheapest")


def test_route_reports_the_policy_that_produced_it(graph):
    route = graph.route("pt", "ru", prefer="dedicated", max_hops=2)
    assert route.prefer == "dedicated"
    assert route.max_hops == 2


# --- routes() --------------------------------------------------------------

def test_routes_returns_more_than_the_winner(graph):
    found = graph.routes("pt", "ru")
    assert len(found) > 1
    assert found[0].model_ids == graph.route("pt", "ru").model_ids


def test_routes_ranking_is_stable_and_carries_licences(graph):
    first = graph.routes("pt", "ru")
    second = graph.routes("pt", "ru")
    assert [r.model_ids for r in first] == [r.model_ids for r in second]
    for route in first:
        assert route.licenses and all(isinstance(l, str) for l in route.licenses)
        assert route.license_tier in {"permissive", "share-alike", "non-commercial"}


def test_routes_is_bounded(graph):
    assert len(graph.routes("pt", "ru", limit=2)) == 2
    assert len(graph.routes("pt", "ru")) <= 10


def test_routes_are_deduplicated(graph):
    signatures = [r.model_ids + r.pivots for r in graph.routes("pt", "ru")]
    assert len(signatures) == len(set(signatures))


def test_a_route_licence_tier_is_its_worst_hop():
    mixed = Route("a", "c", (
        Hop("x", "a", "b", "marian", "Apache-2.0", "permissive", 1, True),
        Hop("y", "b", "c", "nllb", "CC-BY-NC-4.0", "non-commercial", 1, False),
    ))
    assert mixed.license_tier == "non-commercial"


# --- pruning + performance -------------------------------------------------

def test_pivoting_inside_one_multilingual_model_is_pruned(graph):
    """m2m covers pt->ru directly; pt->(m2m)->es->(m2m)->ru is never offered."""
    for route in graph.routes("pt", "ru", prefer="dedicated"):
        if route.n_hops == 2:
            assert set(route.model_ids) != {"m2m"}


def test_tags_are_normalised_before_routing(graph):
    """por_Latn, PT and pt-BR are all the same node."""
    assert normalize_tag("por_Latn") == "pt"
    assert normalize_tag("eng_Latn") == "en"
    assert graph.route("eng_Latn", "por_Latn").hops[0].model_id == "opus-en-pt"


def _realistic_graph():
    """The real registry, so a future registry change cannot go quadratic."""
    from linguonnx.model_manager import list_models
    from linguonnx.translate.models import capability_from_entry
    entries = [e for e in list_models(kind="translate").values()
               if e["precision"] == "int8"]
    return TranslationGraph([capability_from_entry(e) for e in entries])


def test_resolution_stays_fast_on_the_real_registry():
    """NLLB alone is a 202-language clique. Routing must not walk it."""
    graph = _realistic_graph()
    assert len(graph.languages) > 190
    pairs = [("en", "pt"), ("pt", "ru"), ("pt", "eu"), ("gl", "ca"),
             ("en", "ja"), ("ru", "zh"), ("ast", "eu"), ("en", "kab")]
    start = time.perf_counter()
    for _ in range(20):
        for src, tgt in pairs:
            try:
                graph.route(src, tgt)
            except NoRouteError:
                pass
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"routing 160 pairs took {elapsed:.2f}s"


def test_listing_routes_stays_fast_on_the_real_registry():
    graph = _realistic_graph()
    start = time.perf_counter()
    for _ in range(5):
        graph.routes("pt", "eu")
        graph.routes("gl", "ca", prefer="dedicated")
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"listing routes took {elapsed:.2f}s"


# --- runnability -----------------------------------------------------------

def _unrunnable(model_id="broken", **kwargs):
    return Capability(model_id=model_id, arch="indictrans2", license="MIT",
                      license_tier="permissive", size_mb=100,
                      runnable=False,
                      unrunnable_reason="no preprocessing pipeline", **kwargs)


def test_a_capability_is_runnable_unless_it_says_otherwise():
    assert marian("en", "pt").is_runnable is True
    assert marian("en", "pt").unrunnable_because is None
    assert _unrunnable(pair=("en", "kn")).is_runnable is False


def test_an_unrunnable_capability_is_not_a_language_in_the_graph():
    graph = TranslationGraph([marian("en", "pt"), _unrunnable(pair=("en", "kn"))])
    assert "kn" not in graph.languages
    assert "pt" in graph.languages


def test_can_translate_is_false_when_only_an_unrunnable_model_covers_the_pair():
    """The whole point of `can_translate`: it must not promise what fails later."""
    graph = TranslationGraph([marian("en", "pt"), _unrunnable(pair=("en", "kn"))])
    assert graph.can_translate("en", "kn") is False
    with pytest.raises(NoRouteError):
        graph.route("en", "kn")
    assert graph.routes("en", "kn") == []


def test_no_route_names_the_model_that_covers_the_pair_but_cannot_run():
    """"Unsupported" and "not implemented here" are different things to fix."""
    graph = TranslationGraph([marian("en", "pt"), _unrunnable(pair=("en", "kn"))])
    with pytest.raises(NoRouteError) as err:
        graph.route("en", "kn")
    assert "broken" in str(err.value)
    assert "no preprocessing pipeline" in str(err.value)


def test_an_unrunnable_model_is_never_a_leg_of_a_multi_hop_route():
    graph = TranslationGraph([marian("en", "pt"),
                              _unrunnable(model_id="broken", pair=("pt", "kn"))],
                             pivot_preference=("pt",))
    assert graph.can_translate("en", "kn") is False


def test_runnability_comes_from_the_registry_when_the_capability_is_silent():
    """`capability_from_entry` states nothing, so the entry decides.

    This is what keeps `route` and `translate` in agreement without every
    caller having to read the registry, and it is why a pipeline landing later
    only has to clear the flag in the registry.

    Every registry entry is runnable today - the IndicTrans2 and OpenNMT-BPE
    pipelines landed - so the flag is exercised through a synthetic entry
    rather than a real one. The wiring is what matters, not which model
    happens to be waiting for a pipeline this month.
    """
    from linguonnx.translate.models import capability_from_entry
    from linguonnx.translate.graph import entry_runnability

    entry = {"model_id": "future-arch-model", "arch": "someday",
             "license": "MIT", "license_tier": "permissive", "size_mb": 1,
             "pair": ["en", "pt"], "runnable": False,
             "unrunnable_reason": "no pipeline for this architecture yet"}
    assert entry_runnability(entry) == (False, "no pipeline for this architecture yet")
    cap = capability_from_entry(entry)
    assert cap.runnable is None, "the entry, not the capability, states it"


def test_entry_runnability_defaults_to_runnable():
    assert entry_runnability(
        {"model_id": "x", "arch": "m2m100", "languages": ["en", "pt"]}) == (True, None)


def test_entry_runnability_refuses_a_multi_target_marian_with_no_token(caplog):
    """No prefix token means the decoder picks a target language on its own."""
    entry = {"model_id": "opus-mt-en-sla-int8", "arch": "marian",
             "languages": ["pl", "cs", "ru"]}
    runnable, reason = entry_runnability(entry)
    assert runnable is False
    assert "target token" in reason
    assert "opus-mt-en-sla-int8" in caplog.text


def test_entry_runnability_accepts_a_multi_target_marian_with_a_token():
    assert entry_runnability({"model_id": "x", "arch": "marian",
                              "languages": ["pl", "cs"],
                              "target_token": ">>pol<<"})[0] is True
    assert entry_runnability({"model_id": "x", "arch": "marian",
                              "languages": ["liv", "et"],
                              "target_token_template": "<2{code}>"})[0] is True


def test_a_dedicated_marian_needs_no_target_token():
    assert entry_runnability({"model_id": "x", "arch": "marian",
                              "pair": ["en", "pt"]})[0] is True


# --- max_hops is validated everywhere, not only in the constructor ---------

@pytest.mark.parametrize("max_hops", [0, -1])
def test_constructor_rejects_max_hops_below_one(max_hops):
    with pytest.raises(ValueError):
        TranslationGraph([marian("en", "pt")], max_hops=max_hops)


@pytest.mark.parametrize("max_hops", [0, -1])
def test_route_rejects_max_hops_below_one(graph, max_hops):
    """It used to behave as `max_hops=1`, contradicting the constructor."""
    with pytest.raises(ValueError):
        graph.route("en", "pt", max_hops=max_hops)


@pytest.mark.parametrize("max_hops", [0, -1])
def test_routes_rejects_max_hops_below_one(graph, max_hops):
    with pytest.raises(ValueError):
        graph.routes("en", "pt", max_hops=max_hops)


@pytest.mark.parametrize("max_hops", [0, -1])
def test_can_translate_rejects_max_hops_below_one(graph, max_hops):
    with pytest.raises(ValueError):
        graph.can_translate("en", "pt", max_hops=max_hops)


# --- validating a caller-supplied route ------------------------------------

DIRECTIONAL = Capability(
    model_id="en-indic", arch="indictrans2", license="MIT",
    license_tier="permissive", size_mb=480, runnable=True,
    src_languages=frozenset({"en"}), tgt_languages=frozenset({"hi", "ta"}))


def _hop(cap, src, tgt):
    return Hop(model_id=cap.model_id, src=src, tgt=tgt, arch=cap.arch,
               license=cap.license, license_tier=cap.license_tier,
               size_mb=cap.size_mb, dedicated=cap.dedicated)


def test_validate_route_accepts_a_route_the_graph_itself_produced(graph):
    route = graph.route("en", "ru")
    assert graph.validate_route(route) is route


def test_validate_route_refuses_a_backwards_hop_on_a_directional_model():
    """Both tags are in the model's code map, so nothing else would catch it."""
    graph = TranslationGraph([DIRECTIONAL])
    backwards = Route("hi", "en", (_hop(DIRECTIONAL, "hi", "en"),))
    with pytest.raises(InvalidRouteError) as err:
        graph.validate_route(backwards)
    assert "en-indic" in str(err.value)


def test_validate_route_refuses_a_reversed_dedicated_pair(graph):
    cap = marian("en", "pt")
    backwards = Route("pt", "en", (_hop(cap, "pt", "en"),))
    with pytest.raises(InvalidRouteError):
        graph.validate_route(backwards)


def test_validate_route_accepts_the_supported_direction():
    graph = TranslationGraph([DIRECTIONAL])
    forwards = Route("en", "hi", (_hop(DIRECTIONAL, "en", "hi"),))
    assert graph.validate_route(forwards) is forwards


def test_validate_route_refuses_an_unknown_model(graph):
    stranger = marian("en", "pt")
    route = Route("en", "pt", (Hop("not-registered", "en", "pt", "marian",
                                   "MIT", "permissive", 1, True),))
    with pytest.raises(InvalidRouteError):
        graph.validate_route(route)
    assert stranger.model_id in {c.model_id for c in graph.capabilities}


def test_validate_route_refuses_an_unrunnable_model():
    cap = _unrunnable(pair=("en", "kn"))
    graph = TranslationGraph([marian("en", "pt"), cap])
    route = Route("en", "kn", (_hop(cap, "en", "kn"),))
    with pytest.raises(InvalidRouteError) as err:
        graph.validate_route(route)
    assert "no preprocessing pipeline" in str(err.value)


def test_validate_route_refuses_a_broken_chain(graph):
    hops = (_hop(marian("en", "pt"), "en", "pt"),
            _hop(marian("es", "ca"), "es", "ca"))
    with pytest.raises(InvalidRouteError):
        graph.validate_route(Route("en", "ca", hops))


def test_validate_route_refuses_hops_that_do_not_match_the_endpoints(graph):
    route = Route("en", "ru", (_hop(marian("en", "pt"), "en", "pt"),))
    with pytest.raises(InvalidRouteError):
        graph.validate_route(route)


def test_validate_route_refuses_an_empty_route(graph):
    with pytest.raises(InvalidRouteError):
        graph.validate_route(Route("en", "pt", ()))


# --- malformed caller input vs unsupported language ------------------------

def test_normalize_tag_is_lenient_and_says_so(caplog):
    assert normalize_tag("!!!") == "!!!"
    assert "not a parseable language tag" in caplog.text


def test_normalize_tag_strict_refuses_junk():
    with pytest.raises(MalformedTagError):
        normalize_tag("!!!", strict=True)


def test_normalize_tag_refuses_an_empty_tag():
    with pytest.raises(MalformedTagError):
        normalize_tag("   ")


def test_routing_a_malformed_tag_is_not_reported_as_an_unsupported_language(graph):
    """`NoRouteError` would read as "no model for your language". It is not that."""
    with pytest.raises(MalformedTagError):
        graph.route("<script>", "pt")
    with pytest.raises(MalformedTagError):
        graph.routes("en", "en--")


def test_an_unsupported_but_well_formed_tag_is_still_a_no_route(graph):
    with pytest.raises(NoRouteError):
        graph.route("en", "kea")


def test_a_node_the_registry_minted_stays_addressable():
    """Lenient normalisation at build time must not make a model unreachable."""
    graph = TranslationGraph([marian("!!!", "pt")])
    assert graph.route("!!!", "pt").n_hops == 1


def test_a_region_subtag_routes_as_its_language(graph):
    """No model distinguishes pt-BR from pt; refusing the region helps nobody."""
    assert graph.route("pt-BR", "en").hops[0].model_id == "opus-pt-en"


# --- asymmetric capabilities ----------------------------------------------

SRC_ONLY = Capability(model_id="src-only", arch="m2m100", license="MIT",
                      license_tier="permissive", size_mb=10,
                      languages=frozenset({"en", "hi", "ta"}),
                      src_languages=frozenset({"en"}))
TGT_ONLY = Capability(model_id="tgt-only", arch="m2m100", license="MIT",
                      license_tier="permissive", size_mb=10,
                      languages=frozenset({"en", "hi", "ta"}),
                      tgt_languages=frozenset({"en"}))


def test_a_capability_with_only_src_languages_is_directional():
    """`languages` fills the side that is not declared, and only that side."""
    assert SRC_ONLY.directional is True
    assert SRC_ONLY.covers("en", "hi") is True
    assert SRC_ONLY.covers("hi", "en") is False
    assert SRC_ONLY.covers("hi", "ta") is False
    assert SRC_ONLY.endpoints() == frozenset({"en", "hi", "ta"})


def test_a_capability_with_only_tgt_languages_is_directional():
    assert TGT_ONLY.directional is True
    assert TGT_ONLY.covers("hi", "en") is True
    assert TGT_ONLY.covers("en", "hi") is False
    assert TGT_ONLY.covers("ta", "en") is True


@pytest.mark.parametrize("cap", [SRC_ONLY, TGT_ONLY])
def test_an_asymmetric_capability_never_covers_a_language_with_itself(cap):
    assert cap.covers("en", "en") is False


def test_routing_over_an_asymmetric_capability_respects_its_direction():
    graph = TranslationGraph([SRC_ONLY])
    assert graph.route("en", "hi").hops[0].model_id == "src-only"
    with pytest.raises(NoRouteError):
        graph.route("hi", "en")


# --- per-model size budget -------------------------------------------------
#
# `count_cached_as_free` defaults to True, and the fixtures below are models
# that no registry knows, so `is_cached` answers False for all of them and the
# cap applies. The cached path gets its own tests, with the cache stubbed.

BIG_MULTI = Capability(model_id="big-multi", arch="madlad", license="Apache-2.0",
                       license_tier="permissive", size_mb=4945,
                       languages=frozenset({"pt", "en", "ru", "ja"}))
SMALL_PT_EN = marian("pt", "en", size=80)
SMALL_EN_RU = marian("en", "ru", size=80)


@pytest.fixture
def capped():
    """One 4.9 GB model that does pt->ru directly, plus two 80 MB models."""
    return [BIG_MULTI, SMALL_PT_EN, SMALL_EN_RU]


def test_without_a_cap_the_big_model_wins_in_one_hop(capped):
    route = TranslationGraph(capped).route("pt", "ru")
    assert route.model_ids == ("big-multi",)


def test_a_cap_forces_a_chain_of_small_models(capped):
    route = TranslationGraph(capped, max_model_mb=500).route("pt", "ru")
    assert route.model_ids == ("opus-pt-en", "opus-en-ru")
    assert route.pivots == ("en",)


def test_every_hop_of_the_chain_fits_the_cap(capped):
    route = TranslationGraph(capped, max_model_mb=500).route("pt", "ru")
    assert all(hop.size_mb <= 500 for hop in route.hops)


def test_the_cap_applies_to_routes_as_well_as_route(capped):
    graph = TranslationGraph(capped, max_model_mb=500)
    assert all("big-multi" not in r.model_ids for r in graph.routes("pt", "ru"))


def test_available_languages_agrees_with_what_the_cap_can_serve(capped):
    """A language only the excluded model reaches is not advertised."""
    graph = TranslationGraph(capped, max_model_mb=500)
    assert "ja" not in graph.languages
    assert graph.can_translate("pt", "ja") is False
    with pytest.raises(NoRouteError):
        graph.route("pt", "ja")


def test_every_advertised_language_is_actually_routable_under_a_cap(capped):
    """`languages` and `route` must not disagree, in either direction."""
    graph = TranslationGraph(capped, max_model_mb=500)
    for src in graph.languages:
        assert any(graph.can_translate(src, tgt)
                   or graph.can_translate(tgt, src)
                   for tgt in graph.languages if tgt != src)


def test_the_cap_is_a_per_call_override_like_max_hops(capped):
    graph = TranslationGraph(capped)
    assert graph.route("pt", "ru", max_model_mb=500).n_hops == 2
    assert graph.route("pt", "ru").n_hops == 1


def test_a_per_call_none_lifts_a_cap_the_graph_was_built_with(capped):
    """`None` is "no cap", so it cannot also mean "inherit"; omitting does that."""
    graph = TranslationGraph(capped, max_model_mb=500)
    assert graph.route("pt", "ru", max_model_mb=None).model_ids == ("big-multi",)
    assert graph.route("pt", "ru").model_ids != ("big-multi",)


def test_languages_under_answers_for_a_per_call_cap(capped):
    graph = TranslationGraph(capped)
    assert "ja" in graph.languages
    assert "ja" not in graph.languages_under(500)


def test_can_translate_honours_a_per_call_cap(capped):
    graph = TranslationGraph(capped)
    assert graph.can_translate("pt", "ja") is True
    assert graph.can_translate("pt", "ja", max_model_mb=500) is False


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_below_one_is_rejected(capped, cap):
    with pytest.raises(ValueError):
        TranslationGraph(capped, max_model_mb=cap)
    with pytest.raises(ValueError):
        TranslationGraph(capped).route("pt", "ru", max_model_mb=cap)


def test_the_environment_supplies_the_default_cap(monkeypatch, capped):
    monkeypatch.setattr("linguonnx.translate.graph.MAX_MODEL_MB", 500)
    assert TranslationGraph(capped).route("pt", "ru").n_hops == 2
    # An explicit None still overrules the environment.
    assert TranslationGraph(capped, max_model_mb=None).route("pt", "ru").n_hops == 1


def test_the_excluded_model_is_remembered_not_dropped(capped):
    graph = TranslationGraph(capped, max_model_mb=500)
    assert [c.model_id for c in graph.oversized_capabilities] == ["big-multi"]


# --- what the cap does to the error ---------------------------------------

def test_no_route_names_the_size_cap_when_that_is_what_bound(capped):
    graph = TranslationGraph(capped, max_model_mb=500)
    with pytest.raises(NoRouteError) as err:
        graph.route("pt", "ja")
    assert "size cap" in str(err.value)
    assert "big-multi" in str(err.value)
    assert "4945 MB" in str(err.value)


def test_no_route_does_not_blame_the_cap_for_a_pair_nothing_covers(capped):
    graph = TranslationGraph(capped, max_model_mb=500)
    with pytest.raises(NoRouteError) as err:
        graph.route("pt", "kab")
    assert "size cap" not in str(err.value)


def test_no_route_does_not_mention_the_cap_when_there_is_none(capped):
    with pytest.raises(NoRouteError) as err:
        TranslationGraph(capped).route("pt", "kab")
    assert "size cap" not in str(err.value)


def test_the_hop_cap_is_named_when_it_is_the_binding_one(capped):
    """A cap that costs a hop must not silently buy the hop back."""
    graph = TranslationGraph(capped, max_model_mb=500, max_hops=1)
    with pytest.raises(NoRouteError) as err:
        graph.route("pt", "ru")
    message = str(err.value)
    assert "max_hops=1" in message
    assert "2-hop route exists" in message


# --- download cost, not memory cost ---------------------------------------

def _stub_cache(monkeypatch, *cached_ids):
    monkeypatch.setattr("linguonnx.model_manager.is_cached",
                        lambda model_id, kind="lid": model_id in cached_ids)


def test_a_cached_model_is_exempt_from_the_cap_by_default(monkeypatch, capped):
    """The budget is on the download, and a cached model costs no download."""
    _stub_cache(monkeypatch, "big-multi")
    graph = TranslationGraph(capped, max_model_mb=500)
    assert graph.route("pt", "ru").model_ids == ("big-multi",)
    assert graph.can_translate("pt", "ja") is True


def test_count_cached_as_free_false_applies_the_cap_regardless(monkeypatch, capped):
    """The other reading of the same number: a small disk, not a slow link."""
    _stub_cache(monkeypatch, "big-multi")
    graph = TranslationGraph(capped, max_model_mb=500,
                             count_cached_as_free=False)
    assert graph.route("pt", "ru").model_ids == ("opus-pt-en", "opus-en-ru")
    assert graph.can_translate("pt", "ja") is False


def test_count_cached_as_free_is_a_per_call_override(monkeypatch, capped):
    _stub_cache(monkeypatch, "big-multi")
    graph = TranslationGraph(capped, max_model_mb=500)
    assert graph.route("pt", "ru", count_cached_as_free=False).n_hops == 2
    assert graph.route("pt", "ru").n_hops == 1


def test_an_uncached_oversized_model_is_excluded_either_way(monkeypatch, capped):
    _stub_cache(monkeypatch)   # nothing is cached
    for free in (True, False):
        graph = TranslationGraph(capped, max_model_mb=500,
                                 count_cached_as_free=free)
        assert graph.route("pt", "ru").n_hops == 2


def test_a_capability_outside_the_registry_is_never_cached():
    """`is_cached` cannot invent the file list of a model it does not know."""
    assert BIG_MULTI.is_cached is False


# --- what a route costs to fetch ------------------------------------------

def test_a_route_reports_its_download_size(monkeypatch, capped):
    _stub_cache(monkeypatch)
    route = TranslationGraph(capped).route("pt", "ru")
    assert route.download_size_mb == 4945
    assert route.cached_size_mb == 0


def test_a_cached_route_costs_no_download(monkeypatch, capped):
    _stub_cache(monkeypatch, "big-multi")
    route = TranslationGraph(capped).route("pt", "ru")
    assert route.download_size_mb == 0
    assert route.cached_size_mb == 4945


def test_a_partly_cached_chain_splits_its_cost(monkeypatch, capped):
    _stub_cache(monkeypatch, "opus-pt-en")
    route = TranslationGraph(capped, max_model_mb=500).route("pt", "ru")
    assert route.cached_size_mb == 80
    assert route.download_size_mb == 80
    assert route.models_size_mb == 160


def test_a_model_used_twice_is_downloaded_once(monkeypatch):
    """Per-hop `total_size_mb` double-counts on purpose; the fetch cost cannot."""
    _stub_cache(monkeypatch)
    both = Capability(model_id="multi", arch="m2m100", license="MIT",
                      license_tier="permissive", size_mb=1200,
                      languages=frozenset({"pt", "xx"}))
    graph = TranslationGraph([both, marian("xx", "ru", size=80)])
    route = graph.routes("pt", "ru")[0]
    assert route.model_ids == ("multi", "opus-xx-ru")
    assert route.download_size_mb == 1280
    assert route.models_size_mb == route.total_size_mb == 1280


def test_download_and_cached_sizes_always_add_up(monkeypatch, capped):
    _stub_cache(monkeypatch, "opus-en-ru")
    route = TranslationGraph(capped, max_model_mb=500).route("pt", "ru")
    assert route.download_size_mb + route.cached_size_mb == route.models_size_mb


# --- the cap against the real registry ------------------------------------

def test_a_cap_trades_long_tail_languages_for_short_chains():
    """What a cap really costs on the shipped registry.

    Two things are true at once and both have to be tested, because only the
    first one is obvious. A chain of small bilingual models replaces the big
    multilingual hop for the pairs those bilingual models exist for - and for
    the long tail they do not exist for, nothing replaces it: MADLAD is the
    only model in the registry that has Chuvash at all. So the cap keeps the
    well-served pairs routable and drops the tail, rather than trimming the
    model list evenly.
    """
    from linguonnx.model_manager import list_models
    from linguonnx.translate.models import capability_from_entry
    caps = [capability_from_entry(e) for e in list_models(kind="translate").values()
            if e["precision"] == "int8" and e["license_tier"] != "non-commercial"]
    # count_cached_as_free=False: whether this host happens to hold MADLAD must
    # not decide what the test measures.
    full = TranslationGraph(caps, count_cached_as_free=False)
    small = TranslationGraph(caps, max_model_mb=500, count_cached_as_free=False)
    # The ratio tightened from 1/4 once `opus-mt-tc-big-itc-itc-int8` (453 MB,
    # under the cap) joined the registry with ~85 languages of its own via a
    # `>>xxx<<` prefix token - a genuinely well-covered small model, not a
    # test regression. The cap still trims the tail hard; it just does not
    # trim it as hard as before this one model's coverage was recognised.
    assert len(small.languages) < len(full.languages) / 2
    assert all(cap.size_mb > 500 for cap in small.oversized_capabilities)
    # pt->ru was one M2M100 hop; under the cap it is a chain of small models.
    chain = small.route("pt", "ru")
    assert chain.n_hops == 2
    assert all(hop.size_mb <= 500 for hop in chain.hops)
    assert chain.models_size_mb < 500


def test_the_real_registry_advertises_only_what_it_can_route_under_a_cap():
    """The languages/routes agreement, on the set where it is hard to hold."""
    from linguonnx.model_manager import list_models
    from linguonnx.translate.models import capability_from_entry
    caps = [capability_from_entry(e) for e in list_models(kind="translate").values()
            if e["precision"] == "int8" and e["license_tier"] != "non-commercial"]
    small = TranslationGraph(caps, max_model_mb=500, count_cached_as_free=False)
    dropped = TranslationGraph(caps, count_cached_as_free=False).languages \
        - small.languages
    assert dropped, "a 500 MB cap must drop the languages only MADLAD reaches"
    for lang in sorted(dropped)[:40]:
        assert not small.can_translate("en", lang)
        assert not small.can_translate(lang, "en")
