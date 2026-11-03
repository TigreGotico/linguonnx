"""Ugiat's MADLAD-400 fine-tunes in the committed registry, and region nodes.

Offline. The four exports keep every `<2xx>` piece of the base vocabulary, so
the registry states what each card says the model was trained for. `es-ES`
and `es-419` become graph nodes, and a request the size cap keeps away from
them routes as `es`.
"""
import sys
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate.graph import TranslationGraph
from linguonnx.translate.models import capability_from_entry

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sync_registry  # noqa: E402

TRANSLATE = list_models(kind="translate")
ARANESE_SOURCES = ("ca", "en", "es", "fr")


def _capability(model_id):
    return capability_from_entry(TRANSLATE[model_id])


@pytest.mark.parametrize("suffix", ["", "-int8"])
def test_the_aranese_model_covers_its_eight_trained_directions_only(suffix):
    capability = _capability(f"madlad400-3b-mt-oc_aranes{suffix}")
    expected = {(src, "oc") for src in ARANESE_SOURCES}
    expected |= {(tgt, src) for src, tgt in expected}
    assert capability.directions == frozenset(expected)
    assert not capability.covers("en", "de")
    assert not capability.covers("en", "ca")


@pytest.mark.parametrize("suffix", ["", "-int8"])
def test_the_three_language_model_covers_its_24_trained_directions_only(suffix):
    capability = _capability(f"madlad400-3b-mt-ast-an-oc_aranes{suffix}")
    expected = {(src, lang) for src in ARANESE_SOURCES for lang in ("an", "ast", "oc")}
    expected |= {(tgt, src) for src, tgt in expected}
    assert capability.directions == frozenset(expected)
    assert not capability.covers("ast", "an")
    assert not capability.covers("en", "es")


@pytest.mark.parametrize("model_id", ["madlad400-3b-mt-oc_aranes-int8",
                                      "madlad400-3b-mt-ast-an-oc_aranes-int8"])
def test_aranese_is_selected_with_the_new_piece(model_id):
    assert TRANSLATE[model_id]["native_codes"] == {"oc": "oc_aranes"}


@pytest.mark.parametrize("region,native", [("es-ES", "es_ES"), ("es-419", "es_419")])
def test_a_regional_spanish_model_writes_only_its_region(region, native):
    model_id = f"madlad400-3b-mt-{native}-int8"
    capability = _capability(model_id)
    base = _capability("madlad400-3b-mt-int8")
    assert capability.tgt_languages == frozenset({region})
    assert capability.src_languages == base.languages
    assert TRANSLATE[model_id]["native_codes"] == {region: native}
    assert capability.covers("de", region)
    assert not capability.covers("en", "es")
    assert not capability.covers(region, "en")


def test_the_base_madlad_entry_is_unchanged():
    assert len(TRANSLATE["madlad400-3b-mt-int8"]["languages"]) == 447
    assert "es-ES" not in TRANSLATE["madlad400-3b-mt-int8"]["languages"]


def test_the_registry_notes_are_the_override_text():
    for model_id, spec in sync_registry.MADLAD_FINETUNES.items():
        for suffix in ("", "-int8"):
            assert TRANSLATE[model_id + suffix]["notes"] == spec["notes"]


def test_the_target_token_pattern_reads_a_numeric_region():
    pattern = sync_registry._PREFIX_TARGET_TOKEN_RE
    assert pattern.match("<2es_419>").group(1) == "es_419"
    assert pattern.match("<2oc_aranes>").group(1) == "oc_aranes"
    assert pattern.match("<2es-419>") is None


def test_a_fine_tune_whose_vocabulary_lacks_the_piece_is_skipped():
    """`<2oc_aranes>` normalises to `oc-aranes`; a vocabulary without it is
    the base vocabulary, and the entry would send a piece the model lacks."""
    with pytest.raises(sync_registry.SkipRepo, match="oc-aranes"):
        sync_registry._madlad_coverage("madlad400-3b-mt-oc_aranes",
                                       ["ca", "en", "es", "fr", "oc"])
    coverage = sync_registry._madlad_coverage(
        "madlad400-3b-mt-oc_aranes", ["ca", "en", "es", "fr", "oc", "oc-aranes"])
    assert coverage["native_codes"] == {"oc": "oc_aranes"}


def test_a_regional_model_reads_every_language_but_its_target():
    coverage = sync_registry._madlad_coverage("madlad400-3b-mt-es_419",
                                              ["de", "en", "es", "es-419"])
    assert coverage["src_languages"] == ["de", "en", "es"]
    assert coverage["tgt_languages"] == ["es-419"]


def test_the_base_model_keeps_its_vocabulary():
    assert sync_registry._madlad_coverage("madlad400-3b-mt", ["de", "en"]) \
        == {"languages": ["de", "en"]}


def test_without_a_cap_regional_spanish_uses_the_fine_tunes():
    tx = load_translator(max_model_mb=None, count_cached_as_free=False)
    assert tx.route("en", "es-ES").model_ids == ("madlad400-3b-mt-es_ES-int8",)
    assert tx.route("en", "es-419").model_ids == ("madlad400-3b-mt-es_419-int8",)
    assert tx.route("en", "es").model_ids == ("opus-mt-en-es-int8",)


def test_under_a_cap_regional_spanish_routes_as_spanish():
    tx = load_translator(max_model_mb=500, count_cached_as_free=False)
    for region in ("es-ES", "es-419"):
        route = tx.route("en", region)
        assert route.model_ids == ("opus-mt-en-es-int8",)
        assert route.hops[-1].tgt == "es"
        assert tx.can_translate("en", region)


def test_a_regional_source_routes_as_its_language():
    """No model reads `es-ES` text, so it is read as `es`."""
    tx = load_translator(max_model_mb=None, count_cached_as_free=False)
    route = tx.route("es-ES", "en")
    assert route.hops[0].src == "es"
    assert route.model_ids == tx.route("es", "en").model_ids


def test_routes_falls_back_like_route():
    tx = load_translator(max_model_mb=500, count_cached_as_free=False)
    routes = tx.graph.routes("en", "es-ES")
    assert routes and routes[0].model_ids == tx.route("en", "es-ES").model_ids


def test_a_script_or_private_use_subtag_is_never_dropped():
    graph = load_translator(max_model_mb=None).graph
    assert ("es", "en") in graph._regional_fallbacks("es-ES", "en")
    for node, base in (("sr-Latn", "sr"), ("zh-Hant", "zh"),
                       ("ext-PT-x-barrancos", "ext")):
        assert base in graph._addressable
        assert graph._regional_fallbacks(node, "en") == []
        assert graph._regional_fallbacks("en", node) == []
