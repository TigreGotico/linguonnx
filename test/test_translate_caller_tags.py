"""A model's private spelling means nothing in a caller's tag."""

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate.graph import NoRouteError, normalize_tag


def test_arn_latn_is_mapudungun():
    assert normalize_tag("arn_Latn") == "arn"
    assert normalize_tag("arn_Latn") == normalize_tag("arn")


def test_jap_is_not_japanese():
    assert normalize_tag("jap") == "jap"


@pytest.fixture(scope="module")
def translator():
    return load_translator(include_noncommercial=True)


def test_mapudungun_does_not_route_to_occitan(translator):
    route = translator.route("es", "arn_Latn")
    assert route.tgt == "arn"
    assert route.model_ids == ("madlad400-3b-mt-int8",)


def test_aranese_still_routes_as_occitan(translator):
    assert translator.route("es", "oc").tgt == "oc"
    assert translator.models["aina-es-oc-int8"]["native_codes"]["oc"] == "arn_Latn"


@pytest.mark.parametrize("tag", ["ns", "jap"])
def test_a_model_spelling_is_not_a_language_for_a_caller(translator, tag):
    with pytest.raises(NoRouteError):
        translator.route("en", tag)


def test_the_committed_registry_routes_no_model_spelling():
    routed = {tag for entry in list_models("translate").values()
              for key in ("languages", "src_languages", "tgt_languages", "pair")
              for tag in entry.get(key) or ()}
    assert len(routed) > 300
    assert not routed & {"ns", "jap", "arn_Latn"}
