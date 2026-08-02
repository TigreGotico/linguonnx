"""Pivot ranking by real linguistic distance.

The routing graph keeps its curated ``REGIONAL_PIVOTS`` table, but when
``orthography2ipa`` is installed the candidates it produces are reordered by
phonological distance. Both paths are tested: with the package, and with it
mocked away.
"""

import pytest

from linguonnx.translate import distance
from linguonnx.translate.graph import Capability, TranslationGraph

# --- fixtures: a miniature registry with two viable Basque pivots ----------

M2M = Capability(model_id="m2m", arch="m2m100", license="MIT",
                 license_tier="permissive", size_mb=1200,
                 languages=frozenset({"en", "pt", "es", "ca", "gl", "ru", "de"}))


def marian(src, tgt, size=350):
    return Capability(model_id=f"opus-{src}-{tgt}", arch="marian",
                      license="Apache-2.0", license_tier="permissive",
                      size_mb=size, pair=(src, tgt))


def both(a, b, size=350):
    return [marian(a, b, size), marian(b, a, size)]


#: Every leg is dedicated and the same size, so hop count, licence and size all
#: tie and the pivot ordering is what actually decides the route.
BILINGUAL = (both("en", "pt") + both("en", "es") + both("en", "eu")
             + both("en", "ru") + both("en", "ca") + both("en", "gl")
             + both("es", "pt") + both("es", "eu") + both("es", "ru")
             + both("es", "ca") + both("es", "gl"))


def build(**kwargs):
    return TranslationGraph([M2M] + BILINGUAL, **kwargs)


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each test starts from a cold memo so call counting is meaningful."""
    distance.clear_cache()
    yield
    distance.clear_cache()


@pytest.fixture
def no_o2i(monkeypatch):
    """Pretend orthography2ipa is not installed."""
    monkeypatch.setattr(distance, "_BACKEND", False)
    distance.clear_cache()
    return monkeypatch


requires_o2i = pytest.mark.skipif(
    not distance.available(), reason="orthography2ipa is not installed")


# --- phonological ranking --------------------------------------------------

@requires_o2i
def test_pt_to_eu_pivots_through_spanish_not_english():
    """The headline case. Spanish keeps far more of Basque's contact lexicon."""
    route = build(pivot_ranking="phonological").route("pt", "eu",
                                                     prefer="dedicated")
    assert route.n_hops == 2
    assert route.pivots == ("es",)
    assert route.pivot_basis == "phonological"


@requires_o2i
def test_pt_to_eu_scores_spanish_below_english():
    def through(pivot):
        return (distance.pair_distance("pt", pivot)
                + distance.pair_distance(pivot, "eu"))

    assert through("es") < through("en")


@requires_o2i
def test_gl_to_ca_pivots_through_spanish():
    """No gl<->ca model exists, so this pair must pivot; es beats en."""
    route = build(pivot_ranking="phonological").route("gl", "ca",
                                                      prefer="dedicated")
    assert route.pivots == ("es",)


@requires_o2i
def test_pt_to_ru_ranks_candidates_by_summed_distance():
    graph = build(pivot_ranking="phonological")
    candidates = graph._pivot_candidates("pt", "ru")
    scored = [(distance.pair_distance("pt", c) + distance.pair_distance(c, "ru"), c)
              for c in candidates
              if distance.pair_distance("pt", c) is not None
              and distance.pair_distance(c, "ru") is not None]
    assert scored == sorted(scored, key=lambda s: s[0]) or \
        [c for _, c in scored] == [c for _, c in sorted(scored)]
    assert candidates[0] == min(scored)[1]


@requires_o2i
def test_ranking_reorders_but_never_widens_the_candidate_set():
    """o2i is allowed to sort the candidates; it may not add or drop any."""
    table = build(pivot_ranking="table")._pivot_candidates("pt", "eu")
    phono = build(pivot_ranking="phonological")._pivot_candidates("pt", "eu")
    assert sorted(table) == sorted(phono)
    assert table != phono or len(table) < 2


@requires_o2i
def test_unknown_language_does_not_sort_first_and_does_not_raise():
    exotic = "qqx"  # not a language orthography2ipa knows
    graph = build(pivot_ranking="phonological")
    ordered = graph._rank_phonologically(
        "pt", "eu", ["en", exotic, "es"])
    assert ordered[0] == "es"
    assert ordered[-1] == exotic
    assert distance.pair_distance("pt", exotic) is None


@requires_o2i
def test_distances_are_memoized_across_route_calls(monkeypatch):
    calls = {"n": 0}
    real = distance._backend()
    get, phon = real

    def counting(a, b):
        calls["n"] += 1
        return phon(a, b)

    monkeypatch.setattr(distance, "_BACKEND", (get, counting))
    distance.clear_cache()

    graph = build(pivot_ranking="phonological")
    graph.route("pt", "eu", prefer="dedicated")
    first = calls["n"]
    assert first > 0
    graph.route("pt", "eu", prefer="dedicated")
    assert calls["n"] == first


# --- fallback when orthography2ipa is missing ------------------------------

def test_auto_falls_back_to_table_when_o2i_missing(no_o2i):
    graph = build(pivot_ranking="auto")
    assert graph.pivot_ranking == "table"
    route = graph.route("pt", "eu", prefer="dedicated")
    assert route.n_hops == 2
    assert route.pivot_basis == "table"


def test_table_ranking_still_routes_sensibly(no_o2i):
    """REGIONAL_PIVOTS already prefers Spanish for Basque; that must survive."""
    route = build(pivot_ranking="table").route("pt", "eu", prefer="dedicated")
    assert route.pivots == ("es",)


def test_explicit_phonological_without_o2i_is_an_error(no_o2i):
    with pytest.raises(ValueError, match="orthography2ipa"):
        build(pivot_ranking="phonological")


def test_unknown_pivot_ranking_name_is_rejected():
    with pytest.raises(ValueError, match="pivot_ranking"):
        build(pivot_ranking="vibes")


def test_pair_distance_reports_none_without_backend(no_o2i):
    assert distance.pair_distance("pt", "es") is None
    assert distance.available() is False


# --- basis reporting -------------------------------------------------------

def test_direct_route_still_reports_the_basis():
    graph = build(pivot_ranking="auto")
    route = graph.route("en", "pt")
    assert route.n_hops == 1
    assert route.pivot_basis == graph.pivot_ranking
    assert route.pivot_basis in ("phonological", "table")


def test_translator_exposes_pivot_ranking():
    from linguonnx import load_translator
    tx = load_translator(pivot_ranking="table")
    assert tx.pivot_ranking == "table"
    assert tx.route("pt", "en").pivot_basis == "table"
