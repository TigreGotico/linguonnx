"""IbRo-nllb and the AfriGloss fine-tunes in the committed registry.

Offline. Each claim here was checked against the export ledger, where every
sentence was translated by ``transformers`` and by the ONNX graphs.
"""
import json
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
IBRO_EDGES = {("es", other) for other in ("an", "ast", "ca", "ca-valencia", "gl", "oc")}
IBRO_EDGES |= {(tgt, src) for src, tgt in IBRO_EDGES} | {("ca", "oc"), ("oc", "ca")}


@pytest.mark.parametrize("model_id", ["IbRo-nllb", "IbRo-nllb-int8"])
def test_ibro_covers_its_fourteen_trained_directions_only(model_id):
    capability = capability_from_entry(TRANSLATE[model_id])
    assert capability.directions == frozenset(IBRO_EDGES)
    assert capability.covers("es", "ca-valencia")
    assert capability.covers("oc", "ca")
    assert not capability.covers("gl", "ast")
    assert not capability.covers("an", "ca")
    assert not capability.covers("es", "fr")


def test_ibro_sends_the_fine_tune_tokens():
    native = TRANSLATE["IbRo-nllb-int8"]["native_codes"]
    assert native["an"] == "arg_Latn"
    assert native["oc"] == "arn_Latn"
    assert native["ca-valencia"] == "val_Latn"
    assert native["ca"] == "cat_Latn"


def test_valencian_routes_through_ibro_and_aranese_spellings_agree():
    tx = load_translator()
    assert tx.route("es", "ca-valencia").model_ids == ("IbRo-nllb-int8",)
    assert tx.route("es", "oc-aranes").model_ids == tx.route("es", "oc").model_ids


def test_a_directions_list_becomes_the_capability_edges():
    entry = {"model_id": "x", "arch": "nllb", "license": "MIT",
             "license_tier": "permissive", "size_mb": 1,
             "languages": ["es", "ast", "an"],
             "directions": [["es", "ast"], ["ast", "es"]]}
    capability = capability_from_entry(entry)
    assert capability.covers("es", "ast") and capability.covers("ast", "es")
    assert not capability.covers("es", "an")


@pytest.mark.parametrize("repo", ["TigreGotico/afrigloss-nllb-600m-onnx",
                                  "TigreGotico/afrigloss-nllb-1.3b-onnx"])
def test_the_afrigloss_nllb_exports_are_skipped_with_the_reason(repo):
    skipped = json.loads((ROOT / "linguonnx" / "model_index" / "skipped.json")
                         .read_text(encoding="utf-8"))
    assert "id 256204" in skipped[repo]
    assert not any(e.get("hf_repo") == repo for e in TRANSLATE.values())


def test_the_afrigloss_nllb_skip_is_allowlisted():
    assert set(sync_registry._UNSUPPORTED_TOKENIZATION) \
        == {"afrigloss-nllb-600m", "afrigloss-nllb-1.3b"}


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
