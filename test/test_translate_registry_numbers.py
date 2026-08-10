"""The coverage numbers the docs quote, asserted against the real registry.

`docs/routing.md` and `linguonnx/translate/graph.py` both argue from concrete
counts: how many languages a 500 MB cap deletes, and how many the oversize
fallback keeps. Those numbers are the whole case for the feature, and they were
wrong - carried over from a 73-model registry into a 369-entry one, off by
more than a factor of four. A reader who checks a documented number and finds
it false stops trusting the paragraph around it, which is the paragraph that
explains why the knob exists.

So the numbers are pinned here. This test fails when the registry grows, and
that failure is the intended signal: update the docs and this file together.

No network. Building a graph parses one JSON file. `count_cached_as_free=False`
throughout, deliberately, and the autouse fixture below clears
`LINGUONNX_MAX_DOWNLOAD_MB` for the same reason: both make the answer depend on
what this host happens to have on disk.

That dependence is real, not an artifact of the test. Whenever the download
budget is below the largest runnable model, routing genuinely differs between a
warm box and a cold one - the same registry and the same config give `en->cv` a
`NoRouteError` cold and `madlad400-3b-mt-int8` warm, because a cached model
costs no download and the budget never sees it. `docs/routing.md` documents
that behaviour under "Routing is cache-dependent".

The numbers here are a different thing: they are the *registry's* coverage, the
figures the docs argue from, and those have to hold on every machine. So this
file pins them under the one configuration that removes the host from the
answer. A coverage number that changed per machine could not be documented;
a per-request route that changes per machine can be, and is.
"""

import pytest

from linguonnx.translate import load_translator

# Recount with:
#   load_translator(max_model_mb=CAP, count_cached_as_free=False)
LANGUAGES_UNCAPPED = 586
LANGUAGES_UNDER_500 = 249
FALLBACK_STEPS_UNDER_500 = 34


@pytest.fixture(autouse=True)
def _no_inherited_limits(monkeypatch):
    """These read the *registry*, not the operator's environment.

    `LINGUONNX_MAX_DOWNLOAD_MB` caps which oversized models the fallback may
    escalate to, so a sibling test that sets it - one in
    `test_translate_oversize_fallback.py` does - would change every number
    below. `os.environ` leaking between test modules has already produced a
    false pass in this repo.
    """
    monkeypatch.delenv("LINGUONNX_MAX_DOWNLOAD_MB", raising=False)
    monkeypatch.delenv("LINGUONNX_MAX_MODEL_MB", raising=False)


def capped(max_model_mb=None, **kwargs):
    return load_translator(max_model_mb=max_model_mb,
                           count_cached_as_free=False, **kwargs)


class TestTheDocumentedCoverageNumbers:

    def test_the_uncapped_registry_reaches_this_many_languages(self):
        assert len(capped().available_languages) == LANGUAGES_UNCAPPED

    def test_a_500_mb_cap_as_a_filter_deletes_most_of_them(self):
        assert len(capped(500).available_languages) == LANGUAGES_UNDER_500

    def test_the_fallback_gets_all_of_them_back(self):
        """The headline claim of `oversize_fallback`, on the real registry."""
        kept = capped(500, oversize_fallback=True)
        assert len(kept.available_languages) == LANGUAGES_UNCAPPED

    def test_the_cap_still_costs_something_without_the_fallback(self):
        """Guards against the numbers being equal for a boring reason."""
        assert LANGUAGES_UNDER_500 < LANGUAGES_UNCAPPED

    def test_the_documented_escalation_step_count(self):
        """`_search`'s docstring quotes this as the bound on the escalation."""
        graph = capped(500, oversize_fallback=True).graph
        got = len(graph._fallback_steps(graph._index))
        assert got == FALLBACK_STEPS_UNDER_500, (
            f"the registry now has {got} distinct oversized model sizes above "
            f"500 MB, not {FALLBACK_STEPS_UNDER_500}. This failure is the "
            f"intended signal that the registry grew, not a bug. To update: "
            f"recount with `load_translator(max_model_mb=500, "
            f"count_cached_as_free=False).graph`, then "
            f"`len(graph._fallback_steps(graph._index))`; set "
            f"FALLBACK_STEPS_UNDER_500 above to that number, and correct the "
            f"same figure where `docs/routing.md` and the `_search` docstring "
            f"in `linguonnx/translate/graph.py` quote it as the bound on the "
            f"escalation.")


class TestTheWorkedExamplesInTheDocs:
    """The `docs/routing.md` code blocks, run."""

    def test_a_small_model_still_serves_en_ca(self):
        tx = capped(500, oversize_fallback=True)
        assert tx.route("en", "ca").model_ids == ("opus-mt-en-ca-int8",)

    def test_chuvash_falls_back_to_madlad(self):
        tx = capped(500, oversize_fallback=True)
        route = tx.route("en", "cv")
        assert route.model_ids == ("madlad400-3b-mt-int8",)
        assert route.waived_size_cap == 500

    def test_chuvash_is_unroutable_without_the_fallback(self):
        assert not capped(500).can_translate("en", "cv")

    @pytest.mark.parametrize("src,tgt", [("pt", "ru"), ("nl", "fi"),
                                         ("pt", "eu")])
    def test_the_chains_the_docs_name_still_route_under_500_mb(self, src, tgt):
        assert capped(500).can_translate(src, tgt)
