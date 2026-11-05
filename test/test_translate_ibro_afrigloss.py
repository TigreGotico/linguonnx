"""IbRo-nllb and the AfriGloss fine-tunes in the committed registry.

Offline. Each claim here was checked against the export ledger, where every
sentence was translated by ``transformers`` and by the ONNX graphs.
"""
import sys
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate.models import capability_from_entry

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sync_registry  # noqa: E402

TRANSLATE = list_models(kind="translate")
IBRO_EDGES = {("es", other) for other in ("an", "ast", "ca", "gl", "oc")}
IBRO_EDGES |= {(tgt, src) for src, tgt in IBRO_EDGES} | {("ca", "oc"), ("oc", "ca")}


@pytest.mark.parametrize("model_id", ["IbRo-nllb", "IbRo-nllb-int8"])
def test_ibro_covers_its_twelve_registered_directions_only(model_id):
    capability = capability_from_entry(TRANSLATE[model_id])
    assert capability.directions == frozenset(IBRO_EDGES)
    assert capability.covers("es", "oc")
    assert capability.covers("oc", "ca")
    assert not capability.covers("gl", "ast")
    assert not capability.covers("an", "ca")
    assert not capability.covers("es", "fr")


def test_ibro_sends_the_fine_tune_tokens():
    native = TRANSLATE["IbRo-nllb-int8"]["native_codes"]
    assert native["an"] == "arg_Latn"
    assert native["oc"] == "arn_Latn"
    assert native["ca"] == "cat_Latn"
    assert "val_Latn" not in native.values()


def test_aranese_spellings_route_alike():
    tx = load_translator()
    assert tx.route("es", "oc-aranes").model_ids == tx.route("es", "oc").model_ids


@pytest.mark.parametrize("budget", [None, 500])
@pytest.mark.parametrize("tgt", ["en", "es", "fr", "pt", "eu"])
def test_a_valencian_request_routes_as_catalan(budget, tgt):
    """A `ca-valencia` node served only by the 3 GB IbRo model took these
    requests off the small Catalan models and failed them under 500 MB."""
    tx = load_translator(max_model_mb=budget, count_cached_as_free=False)
    assert tx.route("ca-valencia", tgt).model_ids == tx.route("ca", tgt).model_ids


def test_an_nllb_export_without_the_clean_up_flag_gains_no_side_file():
    """nllb-600M's tokenizer_config.json leaves the flag false; recording it
    anyway would make every cache already on disk look incomplete."""
    for model_id in ("nllb-600M", "nllb-600M-int8"):
        assert "tokenizer_config" not in TRANSLATE[model_id]["side_files"]


def test_a_directions_list_becomes_the_capability_edges():
    entry = {"model_id": "x", "arch": "nllb", "license": "MIT",
             "license_tier": "permissive", "size_mb": 1,
             "languages": ["es", "ast", "an"],
             "directions": [["es", "ast"], ["ast", "es"]]}
    capability = capability_from_entry(entry)
    assert capability.covers("es", "ast") and capability.covers("ast", "es")
    assert not capability.covers("es", "an")


def test_afrigloss_marian_pairs_send_no_target_token():
    """en-yo keeps its base's >>yor<< token but was trained without it: the
    reference output matched on 6 of 6 sentences without it, 2 of 6 with it."""
    afrigloss = {k: e for k, e in TRANSLATE.items()
                 if k.startswith("afrigloss-opus-mt-")}
    assert len(afrigloss) == 28
    assert not [k for k, e in afrigloss.items() if e.get("target_token")]


def test_only_opus_mt_entries_carry_a_pair_target_token():
    assert not [k for k, e in TRANSLATE.items()
                if e.get("target_token") and not k.startswith("opus-mt-")]


@pytest.mark.parametrize("model_id", ["IbRo-nllb", "IbRo-nllb-int8"])
def test_ibro_notes_are_the_override_text(model_id):
    """`notes` is human-owned, so a committed note outlives a changed
    override; this keeps the published note in step with the coverage."""
    assert TRANSLATE[model_id]["notes"] \
        == sync_registry.MULTILINGUAL_LANGUAGE_OVERRIDES["IbRo-nllb"]["notes"]
