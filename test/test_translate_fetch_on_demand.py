"""Downloading a model mid-request is a policy the operator sets.

An absent model used to be fetched on the request path unconditionally: the
caller waited out the download, however long, and the disk grew by however
much. That is the right trade on a warm well-provisioned server and the wrong
one on an embedded box, a metered link, or anywhere predictable latency beats
coverage - and there was no way to say which one you were.

`fetch_on_demand=False` is the second stance: routing sees only what is
already cached. Two things it must NOT become:

* a coverage cliff - a pair a *cached* chain can serve is still served, even
  when the direct model for it is absent. Falling back to a cached route
  beats an error;
* a lie - when nothing cached covers the pair, the error says the model is
  not cached and names it, so the fix (prefetch it) is visible. "Unsupported
  language" would send the operator to the wrong place entirely.

And under either stance the failure is an exception. Returning the source text
as if it were a translation - a wrong answer wearing an HTTP 200 - is the one
outcome neither setting may produce.

Nothing here touches the network or the disk: `model_manager.is_cached` is the
single point every layer asks, so the tests state a cache instead of having
one.
"""

import pytest

from linguonnx import model_manager
from linguonnx.translate import load_translator
from linguonnx.translate.graph import NoRouteError

#: The two int8 models that chain `pt -> en -> eu`. `pt->eu` has no direct
#: model in the permissive int8 graph, so this pair is exactly the "a cached
#: chain still covers it" case.
PT_EN = "opus-mt-pt-en-int8"
EN_EU = "mt-hitz-en-eu-int8"


@pytest.fixture
def cache(monkeypatch):
    """State which model ids count as cached, for every layer that asks."""
    def _cache(*model_ids):
        cached = set(model_ids)
        monkeypatch.setattr(
            model_manager, "is_cached",
            lambda model_id, kind="lid": model_id in cached)
        return cached
    return _cache


class TestRefuseKeepsRoutingToWhatIsCached:
    """`fetch_on_demand=False` narrows the graph to the cache - it does not
    narrow it to nothing. A two-hop chain of cached models is a better answer
    than an error, so the fallback to it has to survive the filter.
    """

    def test_a_cached_chain_still_serves_a_pair_with_no_cached_direct_model(
            self, cache):
        cache(PT_EN, EN_EU)
        translator = load_translator(fetch_on_demand=False)
        route = translator.route("pt", "eu")
        assert [hop.model_id for hop in route.hops] == [PT_EN, EN_EU]

    def test_absent_models_are_not_routable_at_all(self, cache):
        cache(PT_EN, EN_EU)
        translator = load_translator(fetch_on_demand=False)
        assert set(translator.models) == {PT_EN, EN_EU}


class TestAnAbsentModelIsNamedRatherThanLookingUnsupported:
    """The difference between a constraint and a lie."""

    def test_the_error_says_not_cached_and_names_the_model(self, cache):
        """`en->eu` is served by `mt-hitz-en-eu` and nothing else permissive.

        The hint names the absent models that cover the *pair*. It does not
        chase the legs of routes that only absent models could have formed:
        those models are out of the graph, so there is no route to trace them
        through, and inventing one would mean re-running the search over a
        second graph on every miss.
        """
        cache(PT_EN)  # the `en->eu` model is absent
        translator = load_translator(fetch_on_demand=False)
        with pytest.raises(NoRouteError) as raised:
            translator.route("en", "eu")
        message = str(raised.value)
        assert "not cached" in message
        assert "fetch_on_demand" in message
        assert EN_EU in message

    def test_translate_raises_instead_of_returning_the_source_text(self, cache):
        """The failure mode this whole change exists to forbid.

        Handing the input back is indistinguishable from a translation that
        happened to be the identity, and it arrives with a success status. It
        must be an exception under either stance.
        """
        cache(PT_EN)
        translator = load_translator(fetch_on_demand=False)
        text = "Bom dia, o meu nome e Joao."
        with pytest.raises(NoRouteError):
            translator.translate(text, src="pt", tgt="eu")

    def test_nothing_cached_at_all_is_refused_at_build_time(self, cache):
        cache()
        with pytest.raises(ValueError, match="fetch_on_demand"):
            load_translator(fetch_on_demand=False)


class TestFetchOnDemandIsTheDefault:
    """The default preserves today's behaviour: an existing deployment must
    not silently lose coverage by upgrading.
    """

    def test_an_absent_model_stays_routable_by_default(self, cache):
        cache()  # nothing is cached anywhere
        translator = load_translator()
        assert translator.fetch_on_demand is True
        assert EN_EU in translator.models
        assert translator.can_translate("pt", "eu")
