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

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator

# Recount with:
#   load_translator(max_model_mb=CAP, count_cached_as_free=False)
LANGUAGES_UNCAPPED = 586
LANGUAGES_UNDER_500 = 249
FALLBACK_STEPS_UNDER_500 = 34

# README.md's "What ships" table and docs/models.md's opening paragraph quote
# these directly off `linguonnx/model_index/translate.json` and `lid.json`.
# Recount with:
#   len(list_models(kind="translate"))                        -> entries
#   {v["hf_repo"] for v in translate.json.values()}            -> models
#   Counter(v["precision"] for v in translate.json.values())   -> int8/fp32 split
TRANSLATE_ENTRIES = 369
TRANSLATE_INT8_ENTRIES = 184
TRANSLATE_FP32_ENTRIES = 185
TRANSLATE_MODELS = 185
LID_ENTRIES = 10
LID_MODELS = 5


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
        """`_search`'s docstring quotes this as the bound on the escalation.

        `aina-translator-es-ca-int8` (~1896 MB) was published to HuggingFace
        but is not yet in the committed registry. When `sync_registry.py`
        picks it up, it adds a new distinct oversized model size above 500 MB,
        and this count moves from 34 to 35 for that reason alone - not a bug,
        the expected next value of this constant.
        """
        graph = capped(500, oversize_fallback=True).graph
        got = len(graph._fallback_steps(graph._index))
        assert got == FALLBACK_STEPS_UNDER_500, (
            f"the registry now has {got} distinct oversized model sizes above "
            f"500 MB, not {FALLBACK_STEPS_UNDER_500}. This failure is the "
            f"intended signal that the registry grew, not a bug. If this "
            f"moved from 34 to 35 and `linguonnx/model_index/translate.json` "
            f"now has an `aina-translator-es-ca-int8` entry, that is the "
            f"expected consequence of syncing the registry after that model's "
            f"HuggingFace publication - not a regression to chase. To update: "
            f"recount with `load_translator(max_model_mb=500, "
            f"count_cached_as_free=False).graph`, then "
            f"`len(graph._fallback_steps(graph._index))`; set "
            f"FALLBACK_STEPS_UNDER_500 above to that number, and correct the "
            f"same figure where `docs/routing.md` and the `_search` docstring "
            f"in `linguonnx/translate/graph.py` quote it as the bound on the "
            f"escalation. Also update the registry-derived counts in "
            f"`README.md` and `docs/models.md` (entry/model/language totals) "
            f"if they changed.")


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


class TestTheDocumentedRegistrySizeNumbers:
    """`README.md`'s "What ships" table and `docs/models.md`'s opening
    paragraph both state the raw size of the registry - how many entries it
    holds, the int8/fp32 split, and how many distinct HuggingFace models that
    is. Those figures drift independently of the routing numbers above:
    someone can add a model (or a precision variant of one) without touching
    anything `_search` walks over below a size cap.

    A wrong count here is the same failure as a wrong reachable-languages
    count: a reader who checks it against `translate.json` and finds it false
    stops trusting the paragraph. So it is pinned the same way.
    """

    def test_the_translate_registry_entry_count(self):
        got = len(list_models(kind="translate"))
        assert got == TRANSLATE_ENTRIES, (
            f"translate.json now holds {got} entries, not "
            f"{TRANSLATE_ENTRIES}. This is the intended signal that the "
            f"registry grew, not a bug. Update TRANSLATE_ENTRIES above, and "
            f"correct the same count in README.md's \"What ships\" table and "
            f"the opening paragraphs of docs/models.md.")

    def test_the_lid_registry_entry_count(self):
        got = len(list_models(kind="lid"))
        assert got == LID_ENTRIES, (
            f"lid.json now holds {got} entries, not {LID_ENTRIES}. This is "
            f"the intended signal that the registry grew, not a bug. Update "
            f"LID_ENTRIES above, and correct the same count in README.md's "
            f"\"What ships\" table and docs/models.md's \"What is in the LID "
            f"registry\" section (including the Python example at the end of "
            f"that section).")

    def test_the_translate_precision_split(self):
        from collections import Counter

        entries = list_models(kind="translate")
        got = Counter(v["precision"] for v in entries.values())
        assert (got["int8"], got["fp32"]) == (
            TRANSLATE_INT8_ENTRIES, TRANSLATE_FP32_ENTRIES), (
            f"translate.json now has {got['int8']} int8 and {got['fp32']} "
            f"fp32 entries, not {TRANSLATE_INT8_ENTRIES} int8 and "
            f"{TRANSLATE_FP32_ENTRIES} fp32. This is the intended signal "
            f"that the registry grew, not a bug. Update "
            f"TRANSLATE_INT8_ENTRIES and TRANSLATE_FP32_ENTRIES above, and "
            f"correct the same split in README.md's \"What ships\" table and "
            f"docs/models.md's opening paragraph.")

    def test_the_distinct_translate_model_count(self):
        entries = list_models(kind="translate")
        got = len({v["hf_repo"] for v in entries.values()})
        assert got == TRANSLATE_MODELS, (
            f"translate.json now spans {got} distinct HuggingFace repos, not "
            f"{TRANSLATE_MODELS}. This is the intended signal that the "
            f"registry grew, not a bug. Update TRANSLATE_MODELS above, and "
            f"correct the same count in README.md's \"What ships\" table and "
            f"docs/models.md's opening paragraph (\"N entries is ... across N "
            f"models\").")

    def test_the_distinct_lid_model_count(self):
        entries = list_models(kind="lid")
        got = len({v["hf_repo"] for v in entries.values()})
        assert got == LID_MODELS, (
            f"lid.json now spans {got} distinct HuggingFace repos, not "
            f"{LID_MODELS}. This is the intended signal that the registry "
            f"grew, not a bug - most likely a new LID model, which also "
            f"needs a row in docs/models.md's \"What is in the LID registry\" "
            f"table. Update LID_MODELS above, and correct the model count in "
            f"README.md's \"What ships\" table (\"N models, ... labels, ... "
            f"MB to ... GB\").")
