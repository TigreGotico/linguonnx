"""A model the caller names by hand is not silently over budget.

`load_translator(model="aina-translator-ca-zh")` raised
`NoRouteError: no route from 'ca' to 'zh' within 2 hop(s)` for the one pair
that model exists to serve, because the fp32 export is 9294 MB and the
default budget is 8192 MB. The int8 twin (2345 MB) served the same pair, so
the failure looked like a missing language rather than a budget.

`models=`/`model=` already overrides every entry filter - precision, licence
tier, quality. The size budget has to follow, or naming a model is not enough
to use it. A budget the caller states explicitly still applies.
"""

import pytest

from linguonnx.translate import load_translator
from linguonnx.translate.graph import NoRouteError


# --------------------------------------------------------------------------

class TestAPinnedModelIsNotExcludedByTheDefaultSizeCap:
    """`aina-translator-ca-zh` (fp32, 9294 MB) is the only model for `ca->zh`
    in a graph built from it alone, and the default 8192 MB budget dropped it:
    `load_translator(model=...)` then raised `NoRouteError` for the one pair
    it exists to serve, while the int8 twin (2345 MB) worked. `models=` already
    overrides every entry filter; the budget has to follow.
    """

    @pytest.mark.parametrize("model_id,src,tgt", [
        ("aina-translator-ca-zh", "ca", "zh"),
        ("aina-translator-zh-ca", "zh", "ca"),
    ])
    def test_the_named_model_routes_its_own_pair(self, model_id, src, tgt):
        translator = load_translator(model=model_id)
        route = translator.route(src, tgt)
        assert [hop.model_id for hop in route.hops] == [model_id]

    def test_an_explicit_budget_still_applies(self):
        """Waiving the *default* is not the same as ignoring the caller."""
        translator = load_translator(model="aina-translator-ca-zh",
                                     max_model_mb=1024)
        with pytest.raises(NoRouteError):
            translator.route("ca", "zh")


class TestTheWaiverNeverDiscardsAnOperatorsBudget:
    """`UNSET` is not "the default".

    It reads `LINGUONNX_MAX_MODEL_MB` first and the cold-download budget
    second (`limits.py`, `TranslationGraph._check_max_model_mb`). Collapsing
    it to `None` whenever `models=` was passed threw away a budget set on a
    metered link or a small-disk device: with `LINGUONNX_MAX_MODEL_MB=500` and
    a `models=` list of every registry entry over 4 GB, nothing was excluded
    and 136 GB became routable.

    Naming a model overrides *library* policy. It does not override the
    operator's.
    """

    OVERSIZED = "aina-translator-ca-zh"

    @staticmethod
    def _operator_sets(monkeypatch, variable, value):
        """Set a budget the way an operator does - in the environment.

        `linguonnx.limits.MAX_MODEL_MB` is read once at import, so a process
        that starts with the variable set sees it as a module constant too;
        an in-process `setenv` alone would not reproduce that half.
        """
        monkeypatch.setenv(variable, value)
        if variable == "LINGUONNX_MAX_MODEL_MB":
            monkeypatch.setattr("linguonnx.translate.graph.MAX_MODEL_MB",
                                int(value))
            monkeypatch.setattr("linguonnx.limits.MAX_MODEL_MB", int(value))

    @pytest.mark.parametrize("variable", ["LINGUONNX_MAX_MODEL_MB",
                                          "LINGUONNX_MAX_DOWNLOAD_MB"])
    def test_an_env_budget_survives_a_pinned_model(self, monkeypatch, variable):
        self._operator_sets(monkeypatch, variable, "500")
        translator = load_translator(model=self.OVERSIZED)
        assert translator.graph.max_model_mb is not None, \
            f"{variable} was discarded: the graph runs with no budget at all"
        assert [c.model_id for c in translator.graph.oversized_capabilities] \
            == [self.OVERSIZED]
        with pytest.raises(NoRouteError):
            translator.route("ca", "zh")

    def test_the_blast_radius_is_the_whole_named_list(self, monkeypatch):
        """`models=` takes a list, so the waiver was never about one model."""
        from linguonnx.model_manager import list_models

        self._operator_sets(monkeypatch, "LINGUONNX_MAX_MODEL_MB", "500")
        big = [model_id for model_id, entry in list_models(kind="translate").items()
               if int(entry["size_mb"]) > 4000]
        assert len(big) > 5, "the registry no longer has a big-model tail to test"
        translator = load_translator(models=big)
        assert translator.graph.max_model_mb is not None
        # Every model named is over the budget, so the graph routes nothing.
        # Before the fix it routed all of them - 136 GB under a 500 MB budget.
        assert translator.graph.languages == frozenset(), (
            f"{sorted(translator.graph.languages)[:8]}... reachable under a "
            f"500 MB budget through "
            f"{sorted({c.model_id for c in translator.graph._index.multilingual})[:5]}")

    def test_with_no_operator_budget_the_library_default_is_waived(self,
                                                                   monkeypatch):
        monkeypatch.delenv("LINGUONNX_MAX_MODEL_MB", raising=False)
        monkeypatch.delenv("LINGUONNX_MAX_DOWNLOAD_MB", raising=False)
        translator = load_translator(model=self.OVERSIZED)
        assert translator.graph.max_model_mb is None
        assert translator.route("ca", "zh").hops

    def test_a_waived_route_is_one_the_download_path_will_also_serve(
            self, monkeypatch):
        """Routing and downloading have to agree.

        Waiving only the router's cap plans a hop that `ensure_model_files`
        then refuses with `DownloadTooLargeError` - `can_translate` lying,
        which is the failure the size cap exists to prevent in the first
        place.
        """
        monkeypatch.delenv("LINGUONNX_MAX_MODEL_MB", raising=False)
        monkeypatch.delenv("LINGUONNX_MAX_DOWNLOAD_MB", raising=False)
        assert load_translator(model=self.OVERSIZED)._enforce_download_budget \
            is False
        # ...and the download budget is never waived for anyone else.
        assert load_translator()._enforce_download_budget is True
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "500")
        assert load_translator(model=self.OVERSIZED)._enforce_download_budget \
            is True
