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
