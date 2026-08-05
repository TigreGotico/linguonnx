"""`max_model_mb` deprioritises an oversized model; it does not delete a language.

The regression this file guards shipped to translate.openvoiceos.pt.

The registry grew the `aina-translator-*` family - 1.8 GB bilingual exports of
pairs that a 165 MB `opus-mt` model already served. Under `prefer="dedicated"`
they are dedicated bilingual models like any other, so they won every pair they
touched, and five production pairs went from sub-6-second to 139-250 seconds:
`en->ca` 5.9s -> 165s on `aina-en-ca` alone, `fr->de` 1.5s -> timeout on the
3469 MB `aina-fr-ca` + `aina-ca-de` chain. Nothing was downloading; that is
ONNX session-load time on multi-gigabyte weights that a 4-slot model cache
cannot keep warm.

`max_model_mb=500` fixes all five and is not usable, because the cap *filters*:
it removed MADLAD (4945 MB), NLLB (1866 MB) and M2M100-418M (1207 MB) from the
graph, and with them every language that exists only inside those three -
routable languages fell from 586 to 249 (`test_translate_registry_numbers.py`
pins both against the real registry). Chuvash and Mirandese are not served by
a 500 MB model, by any chain of 500 MB models, or at all.

So the cap has to mean "prefer to stay under this", not "pretend nothing above
this exists": under the cap for every pair a small model can serve, and above
it only for a pair nothing under the cap covers. These tests pin both halves,
and the three properties that make the first half hold - that the fallback is
per request, that it takes the smallest sufficient model, and that it never
runs for a pair the capped search already answered.
"""

import pytest

from linguonnx.translate.graph import (Capability, NoRouteError,
                                       TranslationGraph)


def marian(src, tgt, size=165, model_id=None):
    return Capability(model_id=model_id or f"opus-{src}-{tgt}", arch="marian",
                      license="Apache-2.0", license_tier="permissive",
                      size_mb=size, pair=(src, tgt))


# The five-pair regression, in miniature. `fr->de` has no direct small model
# and is served by a chain through `en`, the way production serves it.
SMALL = [
    marian("pt", "eu"), marian("en", "ca"), marian("es", "gl"),
    marian("fr", "en", size=237), marian("en", "de", size=237),
]

# The `aina-translator-*` shape: a dedicated bilingual model, on a pair a small
# model already serves, at ten times the size.
AINA_EN_CA = marian("en", "ca", size=1823, model_id="aina-en-ca")
AINA_FR_CA = marian("fr", "ca", size=1750, model_id="aina-fr-ca")
AINA_CA_DE = marian("ca", "de", size=1719, model_id="aina-ca-de")

# The long tail. `cv` (Chuvash) and `mwl` (Mirandese) exist in no small model.
MADLAD = Capability(model_id="madlad", arch="madlad", license="Apache-2.0",
                    license_tier="permissive", size_mb=4945,
                    languages=frozenset({"en", "fr", "de", "pt", "es", "ca",
                                         "gl", "eu", "cv", "mwl"}))
M2M100 = Capability(model_id="m2m100", arch="m2m100", license="MIT",
                    license_tier="permissive", size_mb=1207,
                    languages=frozenset({"en", "fr", "de", "pt", "es", "ca",
                                         "gl", "eu", "cv"}))

REGISTRY = SMALL + [AINA_EN_CA, AINA_FR_CA, AINA_CA_DE, MADLAD, M2M100]


@pytest.fixture
def graph():
    """The deployed shape: `prefer="dedicated"`, 500 MB cap, fallback on."""
    return TranslationGraph(REGISTRY, prefer="dedicated", max_model_mb=500,
                            oversize_fallback=True)


# --------------------------------------------------------------------------
# The regression itself: a pair a small model serves never sees a big one.

class TestASmallModelKeepsThePairsItAlreadyServed:

    @pytest.mark.parametrize("src,tgt,expected", [
        ("pt", "eu", ("opus-pt-eu",)),
        ("en", "ca", ("opus-en-ca",)),
        ("es", "gl", ("opus-es-gl",)),
    ])
    def test_the_direct_small_model_wins(self, graph, src, tgt, expected):
        assert graph.route(src, tgt).model_ids == expected

    def test_a_chain_of_small_models_beats_a_chain_of_big_ones(self, graph):
        """`fr->de`: 237+237 MB through `en`, not 1750+1719 MB through `ca`.

        Both are two dedicated hops, so the cost model alone cannot separate
        them - only the cap can, and only if the oversized chain is never
        enumerated at all.
        """
        route = graph.route("fr", "de")
        assert route.model_ids == ("opus-fr-en", "opus-en-de")
        assert route.total_size_mb == 474

    @pytest.mark.parametrize("src,tgt", [
        ("pt", "eu"), ("en", "ca"), ("es", "gl"), ("fr", "de"),
    ])
    def test_no_hop_of_a_served_pair_is_over_the_cap(self, graph, src, tgt):
        assert all(hop.size_mb <= 500 for hop in graph.route(src, tgt).hops)

    @pytest.mark.parametrize("src,tgt", [
        ("pt", "eu"), ("en", "ca"), ("es", "gl"), ("fr", "de"),
    ])
    def test_a_served_pair_does_not_report_a_waiver(self, graph, src, tgt):
        """The exception is not merely unused here - it must be *visibly* unused."""
        route = graph.route(src, tgt)
        assert route.waived_size_cap is None
        assert route.used_oversize_fallback is False

    def test_routes_never_offers_an_oversized_alternative_either(self, graph):
        """`routes()` ranks; it must rank the same set `route()` chose from."""
        offered = {m for r in graph.routes("en", "ca") for m in r.model_ids}
        assert "aina-en-ca" not in offered


# --------------------------------------------------------------------------
# The other half: the tail survives the cap.

class TestTheLongTailStillRoutes:

    @pytest.mark.parametrize("src,tgt", [("en", "mwl"), ("mwl", "en")])
    def test_a_language_only_the_biggest_model_reaches_is_routable(
            self, graph, src, tgt):
        route = graph.route(src, tgt)
        assert route.model_ids == ("madlad",)
        assert graph.can_translate(src, tgt) is True

    def test_the_waiver_is_reported_on_the_route(self, graph):
        route = graph.route("en", "mwl")
        assert route.used_oversize_fallback is True
        assert route.waived_size_cap == 500

    def test_the_waiver_is_visible_in_the_human_readable_form(self, graph):
        assert "500 MB size cap waived" in str(graph.route("en", "mwl"))

    def test_the_smallest_sufficient_model_wins_not_the_first_one(self, graph):
        """`cv` is in both M2M100 (1207 MB) and MADLAD (4945 MB).

        Falling back "past the cap" in one step would hand the pair to
        whichever the cost model liked - and size is the *fifth* tie-break,
        after recency, so the 4.9 GB model can win it. Escalating one model
        size at a time makes the smaller one win by construction.
        """
        route = graph.route("en", "cv")
        assert route.model_ids == ("m2m100",)

    def test_advertised_languages_include_the_tail(self, graph):
        """A server publishes `languages`; a language it can serve belongs there."""
        assert {"cv", "mwl"} <= graph.languages

    def test_the_fallback_is_what_keeps_them(self):
        without = TranslationGraph(REGISTRY, prefer="dedicated",
                                   max_model_mb=500)
        assert not {"cv", "mwl"} & without.languages
        assert {"cv", "mwl"} <= graph_with_fallback().languages


def graph_with_fallback():
    return TranslationGraph(REGISTRY, prefer="dedicated", max_model_mb=500,
                            oversize_fallback=True)


# --------------------------------------------------------------------------
# The properties that keep the first half true.

class TestTheFallbackIsPerRequestAndNotGlobal:
    """The failure to avoid is a fallback that widens the graph once, for
    everyone. `en->mwl` needs MADLAD; `en->ca` must not get it because of that.
    """

    def test_serving_a_tail_pair_does_not_change_a_served_pair(self, graph):
        assert graph.route("en", "mwl").model_ids == ("madlad",)
        assert graph.route("en", "ca").model_ids == ("opus-en-ca",)

    def test_order_does_not_matter(self, graph):
        assert graph.route("en", "ca").model_ids == ("opus-en-ca",)
        assert graph.route("en", "mwl").model_ids == ("madlad",)
        assert graph.route("en", "ca").model_ids == ("opus-en-ca",)


class TestTheCapIsPerModelNotPerRoute:
    """Stated deliberately, because a chain is the case where the two differ.

    `fr->de` under a 500 MB cap is 474 MB in two hops. Nothing checks the
    total, and nothing should: the cap bounds one ONNX session load, and a
    chain pays that cost in pieces the model cache can hold. A per-route
    budget would reject this chain and fall back to a 3469 MB one, which is
    the opposite of what the cap is for.
    """

    def test_a_chain_may_exceed_the_cap_in_total(self):
        graph = TranslationGraph(
            [marian("fr", "en", size=400), marian("en", "de", size=400)],
            prefer="dedicated", max_model_mb=500, oversize_fallback=True)
        route = graph.route("fr", "de")
        assert route.total_size_mb == 800
        assert route.waived_size_cap is None


class TestTheFallbackIsOptIn:
    """Off by default, because `max_model_mb` unset already reads the download
    budget - and a fallback that silently downloads a 4.9 GB model on a host
    whose operator capped it at 500 would be the same class of surprise this
    whole knob exists to prevent.
    """

    def test_off_by_default_the_cap_still_filters(self):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500)
        with pytest.raises(NoRouteError):
            graph.route("en", "mwl")

    def test_it_can_be_turned_on_for_one_call(self):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500)
        assert graph.route("en", "mwl", oversize_fallback=True).model_ids \
            == ("madlad",)

    def test_it_can_be_turned_off_for_one_call(self, graph):
        with pytest.raises(NoRouteError):
            graph.route("en", "mwl", oversize_fallback=False)

    def test_languages_under_follows_the_per_call_override(self, graph):
        assert "mwl" not in graph.languages_under(oversize_fallback=False)
        assert "mwl" in graph.languages_under(oversize_fallback=True)


class TestNoCapMeansNoFallback:
    """With no cap there is nothing to waive, and the ordinary cost model runs
    unchanged - `aina-en-ca` wins `en->ca` again. That is the behaviour the
    cap exists to override, and it must still be reachable.
    """

    def test_an_uncapped_graph_is_unaffected(self):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=None, oversize_fallback=True)
        assert graph.route("en", "mwl").waived_size_cap is None


class TestTheCachedExemptionDoesNotSilentlyDisableTheCap:
    """`count_cached_as_free=True` exempts anything already on disk. That is
    right for a download budget and fatal for this one: the production host
    has every model cached, so the exemption would waive the cap for all 369
    entries and `aina-en-ca` would win `en->ca` again with the cap set.
    """

    def test_the_fallback_flips_the_default(self):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True)
        assert graph.count_cached_as_free is False

    def test_without_the_fallback_the_default_is_unchanged(self):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500)
        assert graph.count_cached_as_free is True

    def test_an_explicit_value_still_wins(self):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True,
                                 count_cached_as_free=True)
        assert graph.count_cached_as_free is True


class TestTheFallbackRespectsTheDownloadBudget:
    """Routing and downloading have to keep agreeing. A model over
    `LINGUONNX_MAX_DOWNLOAD_MB` cannot be fetched, so escalating to it plans a
    route that `translate()` refuses with `DownloadTooLargeError` - a later,
    more confusing failure than the honest "no route".
    """

    def test_a_model_over_the_download_budget_is_not_a_fallback(self, monkeypatch):
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "2000")
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True)
        # `cv` is in M2M100 (1207 MB, affordable) and MADLAD (4945 MB, not).
        assert graph.route("en", "cv").model_ids == ("m2m100",)
        # `mwl` is in MADLAD only, so it stays unroutable rather than
        # routing through a model the downloader will reject.
        with pytest.raises(NoRouteError):
            graph.route("en", "mwl")
        assert "mwl" not in graph.languages


class TestTheFailureMessageNamesTheKnobThatActuallyBlocked:
    """A "no route" under a waived cap must not blame the cap.

    With `oversize_fallback` on, `max_model_mb` is a preference, and the
    fallback already waived it for exactly the pair that failed. What is left
    holding the pair back is `LINGUONNX_MAX_DOWNLOAD_MB`, which the escalation
    refuses to cross so routing keeps agreeing with `ensure_model_files`.

    Naming `max_model_mb` there is not merely unhelpful, it is wrong advice:
    an explicit `max_model_mb` is *not* clamped to the download budget, so
    following it makes the route appear and then `translate()` fails with
    `DownloadTooLargeError`. The operator is walked from a clear failure into
    a confusing one.
    """

    @pytest.fixture
    def blocked(self, monkeypatch):
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "2000")
        return TranslationGraph(REGISTRY, prefer="dedicated",
                                max_model_mb=500, oversize_fallback=True)

    def test_it_names_the_download_budget(self, blocked):
        with pytest.raises(NoRouteError) as excinfo:
            blocked.route("en", "mwl")
        assert "LINGUONNX_MAX_DOWNLOAD_MB" in str(excinfo.value)

    def test_it_does_not_tell_the_operator_to_raise_the_size_cap(self, blocked):
        with pytest.raises(NoRouteError) as excinfo:
            blocked.route("en", "mwl")
        message = str(excinfo.value)
        assert "Raise the cap" not in message
        assert "max_model_mb will not help" in message

    def test_it_still_names_the_model_that_would_have_served_the_pair(self, blocked):
        with pytest.raises(NoRouteError) as excinfo:
            blocked.route("en", "mwl")
        assert "madlad (4945 MB)" in str(excinfo.value)

    def test_with_the_fallback_off_the_size_cap_is_still_the_right_answer(self):
        """The old message is correct when the cap really is a filter."""
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=False)
        with pytest.raises(NoRouteError) as excinfo:
            graph.route("en", "cv")
        message = str(excinfo.value)
        assert "size cap (max_model_mb)" in message
        assert "Raise the cap" in message


class TestTheEscalationDoesNotRescanOnEveryMiss:
    """An unroutable pair must not pay one full search per escalation step.

    The steps are nested indexes, so the widest of them decides whether *any*
    of them can route. Probing it first turns a miss from one search per step
    into two, and a miss is the request-facing path - it is what a translation
    server answers a bad pair with, on every retry, forever.
    """

    @pytest.fixture
    def counted(self, monkeypatch):
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True)
        calls = []
        real = graph._enumerate

        def spy(src, tgt, *args, **kwargs):
            calls.append((src, tgt))
            return real(src, tgt, *args, **kwargs)

        monkeypatch.setattr(graph, "_enumerate", spy)
        return graph, calls

    def test_an_unroutable_pair_costs_two_searches_not_one_per_step(self, counted):
        graph, calls = counted
        # Three distinct oversized sizes touch `en`: 1207, 1823 and 4945.
        assert len(graph._fallback_steps(graph._index, "en", "zz")) == 3
        assert graph.routes("en", "zz") == []
        # The capped search, plus one probe at the widest step. Without the
        # probe this is one search per step, and grows with the registry.
        assert len(calls) == 2

    def test_the_smallest_sufficient_model_still_wins(self, counted):
        """The shortcut must not cost the property it is optimising around."""
        graph, _ = counted
        # `cv` is served by M2M100 (1207 MB) and MADLAD (4945 MB).
        assert graph.route("en", "cv").model_ids == ("m2m100",)
        assert graph.route("en", "cv").waived_size_cap == 500

    def test_a_pair_only_the_widest_step_serves_still_routes(self, counted):
        graph, _ = counted
        assert graph.route("en", "mwl").model_ids == ("madlad",)


class TestThePublicEntryPointsPassTheKnobThrough:
    """`load_translator` and `Translator` are what deployments actually call.

    The graph is where the fallback is implemented and where it was tested;
    it is not where an operator sets it. A knob wired into the graph but
    dropped on the way through `load_translator` is indistinguishable, from
    the deployment's side, from a knob that does not work - and it is exactly
    the passthrough that `ovos-plugin-linguonnx` will depend on.

    These run against the real registry. No network: building a graph parses
    JSON and nothing else.
    """

    @pytest.fixture(autouse=True)
    def _no_inherited_budget(self, monkeypatch):
        # A sibling test that sets this leaks a download ceiling into these,
        # which silently changes which models the fallback may escalate to.
        monkeypatch.delenv("LINGUONNX_MAX_DOWNLOAD_MB", raising=False)
        monkeypatch.delenv("LINGUONNX_MAX_MODEL_MB", raising=False)

    def test_load_translator_reaches_the_graph(self):
        from linguonnx.translate import load_translator

        tx = load_translator(max_model_mb=500, oversize_fallback=True)
        assert tx.oversize_fallback is True
        assert tx.graph.oversize_fallback is True

    def test_load_translator_defaults_it_off(self):
        from linguonnx.translate import load_translator

        assert load_translator(max_model_mb=500).oversize_fallback is False

    def test_the_tail_survives_the_cap_through_the_public_api(self):
        """The whole point, asked the way a deployment asks it."""
        from linguonnx.translate import load_translator

        # `count_cached_as_free=False` on both sides is not decoration: it is
        # what the constructor's own `oversize_fallback` default already does,
        # and without it on the `capped` side this test asks a question about
        # the local cache. On a host that has MADLAD on disk the cap exempts
        # it, `capped.can_translate("en", "cv")` is True, and the test fails
        # for a reason that has nothing to do with the fallback. CI's cache is
        # cold, so it passed there and only ever broke on a warm developer box.
        capped = load_translator(max_model_mb=500, count_cached_as_free=False)
        kept = load_translator(max_model_mb=500, oversize_fallback=True)
        # `cv` (Chuvash) lives only in models far over 500 MB.
        assert not capped.can_translate("en", "cv")
        assert kept.can_translate("en", "cv")
        assert kept.route("en", "cv").waived_size_cap == 500
        assert "cv" in kept.graph.languages
        assert "cv" not in capped.graph.languages
        # ...and a pair a small model serves is untouched by any of it.
        assert kept.route("en", "ca").model_ids == capped.route("en", "ca").model_ids

    def test_a_per_call_override_reaches_the_graph(self):
        """Per-call, `count_cached_as_free` is not defaulted with the flag.

        The constructor flips `count_cached_as_free` to False when
        `oversize_fallback` is on, because the cap is then a load-latency
        budget. A per-call `oversize_fallback=True` on a graph built without
        it inherits that graph's `count_cached_as_free=True` instead, so on a
        warm cache the cap already exempts the cached oversized model and no
        cap is waived - `waived_size_cap` is None, correctly, because nothing
        was waived. Pass both knobs per call to get the constructor's
        semantics for one request; that is what this asserts, and what
        `docs/routing.md` now documents.
        """
        from linguonnx.translate import load_translator

        tx = load_translator(max_model_mb=500)
        assert tx.can_translate("en", "cv", oversize_fallback=True)
        route = tx.route("en", "cv", oversize_fallback=True,
                         count_cached_as_free=False)
        assert route.waived_size_cap == 500
        assert tx.routes("en", "cv", oversize_fallback=True)

    def test_the_cached_exemption_default_reaches_the_graph(self):
        """The tri-state default is public signature; it has to survive the trip."""
        from linguonnx.translate import load_translator

        assert load_translator(max_model_mb=500).count_cached_as_free is True
        assert load_translator(max_model_mb=500,
                               oversize_fallback=True).count_cached_as_free is False
        assert load_translator(max_model_mb=500, oversize_fallback=True,
                               count_cached_as_free=True).count_cached_as_free is True


# --------------------------------------------------------------------------
# Smallest-sufficient, pinned where the cost model wants the *larger* model.

def multi(model_id, size, languages):
    return Capability(model_id=model_id, arch="nllb", license="Apache-2.0",
                      license_tier="permissive", size_mb=size,
                      languages=frozenset(languages))


class TestSmallestSufficientBeatsTheCostModel:
    """The ascending scan must decide, not the route ranking.

    Every other test in this file is served by a case where the cost model
    already prefers the smaller model, so deleting the ascending scan (return
    the widest step's routes unconditionally) changes no answer and all of
    them stay green. That makes the headline guarantee - "the *smallest*
    oversized model that serves the pair" - unguarded, and a refactor can drop
    the loop silently.

    This shape breaks the tie the other way: under `prefer="dedicated"` a
    huge bilingual model outranks a smaller multilingual one on the first
    element of the sort key, so the widest step returns the 4945 MB model and
    only the ascending scan brings back the 1866 MB one.
    """

    NLLB = multi("nllb-small", 1866, {"en", "cv"})
    HUGE = marian("en", "cv", size=4945, model_id="dedicated-huge")

    @pytest.fixture
    def graph(self):
        return TranslationGraph([self.NLLB, self.HUGE], prefer="dedicated",
                                max_model_mb=500, oversize_fallback=True,
                                count_cached_as_free=False)

    def test_the_widest_step_alone_would_pick_the_larger_model(self, graph):
        """The premise: the cost model, unaided, prefers 4945 MB."""
        widest = graph._index_for(4945, count_cached_as_free=False)
        found = graph._enumerate("en", "cv", max_hops=2, prefer="dedicated",
                                 short_circuit=False, index=widest)
        assert found[0].model_ids == ("dedicated-huge",)

    def test_the_smaller_model_wins_anyway(self, graph):
        route = graph.route("en", "cv")
        assert route.model_ids == ("nllb-small",)
        assert route.total_size_mb == 1866
        assert route.waived_size_cap == 500

    def test_routes_ranks_from_the_smallest_sufficient_step_too(self, graph):
        """`routes()` must not offer the 4945 MB model as an alternative."""
        offered = {m for r in graph.routes("en", "cv") for m in r.model_ids}
        assert offered == {"nllb-small"}


class TestSmallestSufficientAgainstABruteForceReference:
    """Differential test: the search must equal an ascending brute force.

    The optimisation under test (probe the widest step, then scan upwards)
    exists only for speed. The reference is the definition: try every step in
    ascending order and take the first that routes. Any divergence is a bug
    in the optimisation, by construction.
    """

    REGISTRY = REGISTRY + [multi("nllb-small", 1866, {"en", "cv", "mwl", "br"}),
                           marian("en", "cv", size=4945,
                                  model_id="dedicated-huge")]

    # `ca -> de` is in here deliberately: it is the only pair in this registry
    # where ascending and descending scans disagree (1207 MB `m2m100` versus
    # the 1719 MB `aina-ca-de`), so it is the only pair that can catch a
    # reversed loop. Without it, `for step in reversed(steps[:-1])` passes.
    PAIRS = [(s, t) for s in ("en", "fr", "pt", "cv", "mwl", "br")
             for t in ("en", "fr", "pt", "cv", "mwl", "br", "zz")
             if s != t] + [("ca", "de"), ("de", "ca")]

    def reference(self, graph, src, tgt):
        """Ascending brute force: the smallest step that routes wins.

        The candidate steps are derived here from the registry directly, not
        by calling `graph._fallback_steps`. Asking the code under test for its
        own search space makes the differential blind by construction to every
        bug that lives inside it - the escalation admitting a model the
        download budget forbids, for one. The definition is "every distinct
        size above the cap, ascending", and that is a property of the
        registry, so the reference computes it from the registry.
        """
        index = graph._index
        found = graph._enumerate(src, tgt, 2, graph.prefer, False, index)
        if found:
            return found[0].model_ids
        steps = sorted({c.size_mb for c in self.REGISTRY
                        if c.size_mb > graph.max_model_mb})
        for step in steps:
            wider = graph._index_for(step, index.count_cached_as_free)
            found = graph._enumerate(src, tgt, 2, graph.prefer, False, wider)
            if found:
                return found[0].model_ids
        return None

    @pytest.mark.parametrize("prefer", ["dedicated", "fewest_hops"])
    def test_every_pair_agrees_with_the_reference(self, prefer):
        graph = TranslationGraph(self.REGISTRY, prefer=prefer,
                                 max_model_mb=500, oversize_fallback=True,
                                 count_cached_as_free=False)
        mismatches = []
        for src, tgt in self.PAIRS:
            try:
                got = graph.route(src, tgt).model_ids
            except NoRouteError:
                got = None
            want = self.reference(graph, src, tgt)
            if got != want:
                mismatches.append((src, tgt, got, want))
        assert mismatches == []


# --------------------------------------------------------------------------
# A cached model is not blocked by the download budget, so it is not blocked
# as a fallback step either.

class TestACachedModelIsNotRefusedByTheDownloadCeiling:
    """`ensure_model_files` budget-checks only when a file is missing.

    So a fully-cached model costs no download and the download budget never
    sees it. Dropping it from the fallback steps refuses a route the
    downloader would have served, and the error then advises prefetching a
    model that is already on disk - advice that provably cannot work.
    """

    @pytest.fixture
    def cached_madlad(self, monkeypatch):
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "2000")
        from linguonnx import model_manager
        monkeypatch.setattr(model_manager, "is_cached",
                            lambda model_id, kind="lid": model_id == "madlad")
        return TranslationGraph(REGISTRY, prefer="dedicated",
                                max_model_mb=500, oversize_fallback=True,
                                count_cached_as_free=False)

    def test_the_cached_oversized_model_is_still_a_fallback_step(self,
                                                                 cached_madlad):
        assert 4945 in cached_madlad._fallback_steps(cached_madlad._index,
                                                     "en", "mwl")

    def test_the_pair_routes_through_it(self, cached_madlad):
        route = cached_madlad.route("en", "mwl")
        assert route.model_ids == ("madlad",)
        assert route.waived_size_cap == 500

    def test_and_the_language_is_listed(self, cached_madlad):
        assert "mwl" in cached_madlad.languages

    def test_an_uncached_model_over_the_ceiling_is_still_refused(self,
                                                                monkeypatch):
        """The exemption is for cached models only; nothing else moved."""
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "2000")
        from linguonnx import model_manager
        monkeypatch.setattr(model_manager, "is_cached",
                            lambda model_id, kind="lid": False)
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True,
                                 count_cached_as_free=False)
        with pytest.raises(NoRouteError):
            graph.route("en", "mwl")

    def test_the_hint_never_names_a_model_that_is_already_prefetched(
            self, monkeypatch):
        """"Prefetch the model" has to be advice that can work.

        MADLAD is cached and so is not blocked; only an uncached over-ceiling
        model may be named. Here MADLAD is cached and `mwl` lives in MADLAD
        alone, so there is nothing left for the hint to blame - and it must
        not raise at all.
        """
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "2000")
        from linguonnx import model_manager
        monkeypatch.setattr(model_manager, "is_cached",
                            lambda model_id, kind="lid": model_id == "madlad")
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True,
                                 count_cached_as_free=False)
        assert graph.route("en", "mwl").model_ids == ("madlad",)


class TestOneCachedModelDoesNotWidenTheGateForUncachedOnes:
    """The exemption is per model; the escalation step it produces is a size.

    `_fallback_steps` correctly drops the uncached 1207 MB `m2m100` when the
    download budget is 500 MB, and correctly keeps the cached 4945 MB
    `madlad`. But the step it returns is the bare number 4945, and an index
    built at a 4945 MB cap admits *every* model at or below 4945 MB - which
    puts `m2m100` straight back in, one line after the filter removed it. The
    cost model then prefers it (smaller), `route` promises it, and
    `ensure_model_files` refuses it with `DownloadTooLargeError`. A route
    promised and then failed is worse than no route.

    The other cached-model test cannot see this: it sets the budget to
    2000 MB, which puts `m2m100` *under* the ceiling, so nothing was ever
    filtered. The case that distinguishes them is mixed - one cached model
    above the ceiling and one uncached model above it.
    """

    @pytest.fixture
    def mixed(self, monkeypatch):
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "500")
        from linguonnx import model_manager
        monkeypatch.setattr(model_manager, "is_cached",
                            lambda model_id, kind="lid": model_id == "madlad")
        return TranslationGraph(REGISTRY, prefer="dedicated",
                                max_model_mb=500, oversize_fallback=True,
                                count_cached_as_free=False)

    def test_the_uncached_model_is_not_a_step(self, mixed):
        """The premise: the filter itself is right. 1207 MB is over budget."""
        assert mixed._fallback_steps(mixed._index, "en", "cv") == [4945]

    def test_the_escalated_index_still_excludes_it(self, mixed):
        """And the index built at that step must not readmit it."""
        wider = mixed._index_for(4945, count_cached_as_free=False,
                                 download_ceiling=500)
        admitted = {c.model_id for c in wider.multilingual}
        assert "madlad" in admitted
        assert "m2m100" not in admitted

    def test_the_route_uses_the_cached_model_not_the_undownloadable_one(
            self, mixed):
        """`m2m100` is 1207 MB, uncached, and over the 500 MB budget."""
        assert mixed.route("en", "cv").model_ids == ("madlad",)

    def test_with_nothing_cached_the_pair_has_no_route_at_all(self,
                                                              monkeypatch):
        """Same budget, cold box: both models are undownloadable."""
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "500")
        from linguonnx import model_manager
        monkeypatch.setattr(model_manager, "is_cached",
                            lambda model_id, kind="lid": False)
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True,
                                 count_cached_as_free=False)
        with pytest.raises(NoRouteError) as excinfo:
            graph.route("en", "cv")
        message = str(excinfo.value)
        assert "madlad" in message and "m2m100" in message

    def test_languages_agrees_with_route(self, mixed):
        """`languages` is built from the same escalated index.

        It may not advertise a language that only `m2m100` reaches, because
        `route` cannot serve it.
        """
        assert "cv" in mixed.languages   # madlad is cached, so `cv` is real


# --------------------------------------------------------------------------
# Three hops: the endpoint filter stops being sound above two.

class TestTheFallbackFindsAMiddleHopAboveTwoHops:
    """A model in the middle of `src -> p1 -> p2 -> tgt` touches neither end.

    The endpoint filter is sound at one and two hops and only there. Above
    two, keeping it made `_fallback_steps(index, src, tgt)` blind to the
    middle model while the global `_fallback_steps(index)` that `languages`
    reads still saw it - so `languages` advertised a pair `route` refused,
    the exact disagreement `_Index` exists to prevent. `max_hops` above 2 is
    documented as allowed (docs/routing.md), so this is a supported shape.
    """

    THREE_HOP = [marian("en", "de", size=100, model_id="s1"),
                 marian("de", "fr", size=3000, model_id="big"),
                 marian("fr", "ja", size=100, model_id="s2")]

    @pytest.fixture
    def graph(self):
        return TranslationGraph(self.THREE_HOP, max_hops=3, max_model_mb=500,
                                oversize_fallback=True,
                                count_cached_as_free=False)

    def test_the_middle_model_is_a_fallback_step(self, graph):
        assert graph._fallback_steps(graph._index, "en", "ja") == [3000]

    def test_the_three_hop_route_is_found(self, graph):
        route = graph.route("en", "ja")
        assert route.model_ids == ("s1", "big", "s2")
        assert route.waived_size_cap == 500

    def test_route_and_languages_agree(self, graph):
        assert "ja" in graph.languages
        assert graph.can_translate("en", "ja")

    def test_the_same_answer_as_lifting_the_cap(self, graph):
        lifted = TranslationGraph(self.THREE_HOP, max_hops=3,
                                  max_model_mb=3000)
        assert graph.route("en", "ja").model_ids == \
            lifted.route("en", "ja").model_ids

    def test_two_hops_still_pays_only_two_searches_for_a_miss(self,
                                                              monkeypatch):
        """The widest-first optimisation must not regress on the common path.

        The endpoint filter is what keeps an unroutable pair at two searches
        instead of one per oversized size. It is kept for `max_hops <= 2`,
        which is the default and every deployed configuration.
        """
        graph = TranslationGraph(REGISTRY, prefer="dedicated",
                                 max_model_mb=500, oversize_fallback=True,
                                 count_cached_as_free=False)
        calls = []
        real = graph._enumerate

        def spy(src, tgt, *args, **kwargs):
            calls.append((src, tgt))
            return real(src, tgt, *args, **kwargs)

        monkeypatch.setattr(graph, "_enumerate", spy)
        assert graph.routes("en", "zz") == []
        assert len(calls) == 2
        # The filter is still doing the narrowing at two hops: three of the
        # five oversized sizes touch `en`, and only those are steps.
        assert len(graph._fallback_steps(graph._index, "en", "zz")) == 3
        assert len(graph._fallback_steps(graph._index)) == 5

    def test_a_per_call_max_hops_of_three_is_honoured_too(self):
        """The graph default is 2; the per-call override must widen it."""
        graph = TranslationGraph(self.THREE_HOP, max_model_mb=500,
                                 oversize_fallback=True,
                                 count_cached_as_free=False)
        with pytest.raises(NoRouteError):
            graph.route("en", "ja")
        assert graph.route("en", "ja", max_hops=3).model_ids == \
            ("s1", "big", "s2")


# --------------------------------------------------------------------------
# `tgt` alone is not silently ignored.

def test_the_endpoint_filter_honours_tgt_without_src():
    """`_fallback_steps(index, tgt=...)` used to fall through to the global
    list, because the guard asked only about `src`. No caller does this
    today, which is exactly why it would have gone unnoticed."""
    graph = TranslationGraph(REGISTRY, prefer="dedicated", max_model_mb=500,
                             oversize_fallback=True,
                             count_cached_as_free=False)
    everything = graph._fallback_steps(graph._index)
    # `mwl` is in MADLAD (4945 MB) alone.
    assert graph._fallback_steps(graph._index, tgt="mwl") == [4945]
    assert graph._fallback_steps(graph._index, src=None, tgt="zz") == []
    assert len(everything) == 5
