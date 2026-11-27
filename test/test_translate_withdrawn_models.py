"""A withdrawn model leaves the registry and stays out of it.

Each one must also leave a route behind: withdrawal is allowed only when
another model serves the pair.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sync():
    spec = importlib.util.spec_from_file_location(
        "sync_registry_withdrawn", ROOT / "scripts" / "sync_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opus_mt_en_jap_is_withdrawn_in_both_precisions():
    registry = list_models(kind="translate")
    assert not [model_id for model_id in registry if "en-jap" in model_id]


def test_english_to_japanese_goes_to_a_model_that_translates():
    route = load_translator().route("en", "ja")
    assert [hop.model_id for hop in route.hops] == ["m2m100-418M-int8"]


def test_the_sync_refuses_a_withdrawn_repo(sync):
    with pytest.raises(sync.SkipRepo) as refused:
        sync.translate_entries("TigreGotico/opus-mt-en-jap-onnx", {}, "")
    assert str(refused.value) == sync.WITHDRAWN["opus-mt-en-jap"]


def test_every_withdrawn_repo_is_recorded_with_its_reason(sync):
    skipped = json.loads(
        (ROOT / "linguonnx" / "model_index" / "skipped.json").read_text("utf-8"))
    for model_id, reason in sync.WITHDRAWN.items():
        assert skipped[f"TigreGotico/{model_id}-onnx"] == reason
