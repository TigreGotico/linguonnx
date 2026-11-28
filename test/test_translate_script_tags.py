"""A subtag may be cut from a caller's tag only when the script stays the same."""

import pytest

from linguonnx.translate import load_translator
from linguonnx.translate.graph import NoRouteError, TranslationGraph
from linguonnx.translate.models import capability_from_entry


def graph(*languages):
    return TranslationGraph(
        [capability_from_entry({
            "model_id": "wide", "arch": "marian", "license": "MIT",
            "license_tier": "permissive", "size_mb": 100,
            "languages": ["en", *languages]})],
        pivot_ranking="table")


FULL = ("az", "pt", "sr", "zh", "zh-Hant")


@pytest.mark.parametrize("tag, node", [
    ("pt-BR", "pt"),
    ("pt-PT", "pt"),
    ("sr-Cyrl", "sr"),
    ("sr-RS", "sr"),
    ("az-AZ", "az"),
    ("az-Latn", "az"),
    ("zh-CN", "zh"),
    ("zh-SG", "zh"),
    ("zh-Hans", "zh"),
    ("zh-Hans-CN", "zh"),
    ("zh-TW", "zh-Hant"),
    ("zh-HK", "zh-Hant"),
    ("zh-MO", "zh-Hant"),
    ("zh-Hant", "zh-Hant"),
    ("zh-Hant-TW", "zh-Hant"),
])
def test_a_tag_reaches_the_node_that_writes_its_script(tag, node):
    built = graph(*FULL)
    assert built.route("en", tag).tgt == node
    assert built.route(tag, "en").src == node


@pytest.mark.parametrize("tag", ["az-Cyrl", "az-Arab", "az-IR", "sr-Latn",
                                 "sr-Latn-RS", "zh-Latn"])
def test_a_tag_in_another_script_has_no_route(tag):
    built = graph(*FULL)
    with pytest.raises(NoRouteError):
        built.route("en", tag)
    with pytest.raises(NoRouteError):
        built.route(tag, "en")


def test_traditional_chinese_never_falls_back_to_simplified():
    with pytest.raises(NoRouteError):
        graph("zh").route("en", "zh-TW")


@pytest.fixture(scope="module")
def translator():
    return load_translator(include_noncommercial=True)


@pytest.mark.parametrize("tag", ["az-Cyrl", "uz-Cyrl", "ku-Arab", "pa-Arab",
                                 "tg-Arab", "mn-Mong", "ha-Arab"])
def test_the_real_graph_does_not_drop_these_scripts(translator, tag):
    with pytest.raises(NoRouteError):
        translator.route("en", tag)
    assert not translator.can_translate("en", tag)


@pytest.mark.parametrize("tag", ["zh-TW", "zh-HK"])
def test_the_real_graph_serves_traditional_chinese(translator, tag):
    assert translator.route("en", tag).tgt == "zh-Hant"
