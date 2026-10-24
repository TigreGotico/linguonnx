"""Rule-based translation: pt -> Barranquenho through ``pt2barranquenho``.

The entry has no ONNX graph and downloads nothing, so every test here runs
offline against the committed registry. ``pt2barranquenho`` comes from the
``rules`` extra, which the ``test`` extra pulls in; it is imported
unconditionally, never skipped.
"""

import sys

import pt2barranquenho
import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate.graph import (NoRouteError, _registry_runnability,
                                       entry_runnability, normalize_tag)
from linguonnx.translate.models import capability_from_entry

BARRANQUENHO = "ext-PT-x-barrancos"

LINES = [
    "Eu gosto muito de ti.",
    "",
    "Olá {name}, o João foi à NATO ontem.",
]


@pytest.fixture
def translator():
    return load_translator()


@pytest.fixture
def without_package(monkeypatch):
    """``import pt2barranquenho`` fails, as on a host without the extra.

    A ``None`` in ``sys.modules`` makes both ``find_spec`` and ``import``
    report the module as absent. Runnability is memoised per model id, so the
    memo is cleared on the way in and on the way out.
    """
    monkeypatch.setitem(sys.modules, "pt2barranquenho", None)
    _registry_runnability.cache_clear()
    yield
    _registry_runnability.cache_clear()


def test_the_barranquenho_tag_keeps_its_canonical_spelling_in_any_case():
    for spelling in (BARRANQUENHO, BARRANQUENHO.lower(), BARRANQUENHO.upper()):
        assert normalize_tag(spelling, strict=True) == "ext-PT-x-barrancos"


def test_the_registry_declares_one_pt_to_barranquenho_edge():
    capability = capability_from_entry(list_models(kind="translate")["pt2barranquenho"])
    assert capability.arch == "rules"
    assert capability.pair == ("pt", BARRANQUENHO)
    assert capability.covers("pt", BARRANQUENHO)
    assert not capability.covers(BARRANQUENHO, "pt")
    assert (capability.license, capability.license_tier, capability.size_mb) \
        == ("Apache-2.0", "permissive", 0)


def test_available_languages_list_the_canonical_tag(translator):
    assert BARRANQUENHO in translator.available_languages
    assert BARRANQUENHO.lower() not in translator.available_languages


@pytest.mark.parametrize("src", ["pt", "pt-PT"])
def test_portuguese_routes_to_barranquenho_in_one_hop(translator, src):
    route = translator.route(src, BARRANQUENHO)
    assert route.model_ids == ("pt2barranquenho",)
    assert (route.src, route.tgt) == ("pt", BARRANQUENHO)


def test_translation_is_the_package_output_line_by_line(translator):
    text = "\n".join(LINES)
    expected = "\n".join(pt2barranquenho.translate(line) if line else line
                         for line in LINES)
    out, route = translator.translate(text, "pt", BARRANQUENHO,
                                      return_route=True)
    assert out == expected
    assert out != text
    assert route.model_ids == ("pt2barranquenho",)


def test_a_pinned_model_gives_the_same_output():
    line = LINES[0]
    assert load_translator(model="pt2barranquenho").translate(
        line, model="pt2barranquenho") == pt2barranquenho.translate(line)


def test_english_reaches_barranquenho_through_a_portuguese_pivot(translator):
    route = translator.route("en", BARRANQUENHO)
    assert route.n_hops == 2
    assert route.pivots == ("pt",)
    assert route.model_ids[-1] == "pt2barranquenho"
    first = list_models(kind="translate")[route.model_ids[0]]
    assert route.licenses == (first["license"], "Apache-2.0")
    assert "pt2barranquenho" in str(route)


@pytest.mark.parametrize("tgt", ["pt", "en", "es"])
def test_nothing_routes_out_of_barranquenho(translator, tgt):
    assert not translator.can_translate(BARRANQUENHO, tgt)
    with pytest.raises(NoRouteError):
        translator.route(BARRANQUENHO, tgt)


def test_a_missing_callable_module_makes_an_entry_unrunnable():
    runnable, reason = entry_runnability({
        "model_id": "x", "arch": "rules", "pair": ["pt", BARRANQUENHO],
        "callable": "no_such_module_for_linguonnx:translate",
        "extra": "rules"})
    assert runnable is False
    assert "linguonnx[rules]" in reason


def test_without_the_package_the_pair_is_covered_but_not_routed(without_package):
    translator = load_translator()
    assert not translator.can_translate("pt", BARRANQUENHO)
    assert BARRANQUENHO not in translator.available_languages
    with pytest.raises(NoRouteError) as err:
        translator.route("pt", BARRANQUENHO)
    message = str(err.value)
    assert "cover this pair but cannot be run" in message
    assert "pt2barranquenho" in message
    assert "linguonnx[rules]" in message


def test_without_the_package_the_tag_never_falls_back_to_extremaduran(
        without_package):
    """``ext`` is Extremaduran, and opus-mt-tc-big-itc-itc covers it."""
    translator = load_translator(precision=None)
    assert translator.can_translate("pt", "ext")
    with pytest.raises(NoRouteError):
        translator.route("pt", BARRANQUENHO)


def test_without_the_package_a_pinned_model_names_the_extra(without_package):
    translator = load_translator(model="pt2barranquenho")
    with pytest.raises(NoRouteError, match=r"linguonnx\[rules\]"):
        translator.translate(LINES[0], model="pt2barranquenho")
