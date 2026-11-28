"""Rule-based translation: pt -> Mirandese through ``pt2mirandese``.

Offline against the committed registry, like ``test_translate_rules.py``.
``pt2mirandese`` comes from the ``rules`` extra, which the ``test`` extra pulls
in; it is imported unconditionally, never skipped.
"""

import sys

import pt2mirandese
import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate.graph import NoRouteError, _registry_runnability
from linguonnx.translate.models import capability_from_entry

LINES = ["Eu gosto muito de ti.", "", "Olá {name}, que horas são?"]


@pytest.fixture
def without_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "pt2mirandese", None)
    _registry_runnability.cache_clear()
    yield
    _registry_runnability.cache_clear()


def test_the_registry_declares_one_pt_to_mwl_edge():
    capability = capability_from_entry(list_models(kind="translate")["pt2mirandese"])
    assert capability.arch == "rules"
    assert capability.pair == ("pt", "mwl")
    assert capability.covers("pt", "mwl")
    assert not capability.covers("mwl", "pt")
    assert (capability.license, capability.license_tier, capability.size_mb) \
        == ("Apache-2.0 AND CC-BY-SA-4.0", "share-alike", 0)


def test_translation_is_the_package_output_line_by_line():
    text = "\n".join(LINES)
    expected = "\n".join(pt2mirandese.translate(line) if line else line
                         for line in LINES)
    out, route = load_translator().translate(text, "pt", "mwl", return_route=True)
    assert out == expected
    assert out != text
    assert "{name}" in out
    assert route.model_ids == ("pt2mirandese",)


@pytest.mark.parametrize("tgt", ["pt", "en"])
def test_nothing_routes_out_of_mirandese_through_the_rules(tgt):
    route = load_translator().route("mwl", tgt)
    assert "pt2mirandese" not in route.model_ids


def test_without_the_package_a_pinned_model_names_the_extra(without_package):
    with pytest.raises(NoRouteError, match=r"linguonnx\[rules\]"):
        load_translator(model="pt2mirandese").translate(
            LINES[0], model="pt2mirandese")


def test_without_the_package_mwl_still_routes_through_a_neural_model(without_package):
    route = load_translator().route("pt", "mwl")
    assert "pt2mirandese" not in route.model_ids
    assert route.hops[-1].model_id.startswith(("madlad400-3b-mt", "opus-mt-itc-itc"))
