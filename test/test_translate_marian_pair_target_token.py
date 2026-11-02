"""A Marian pair whose one target language has several varieties.

`opus-mt-en-ar` is named as a pair, so the registry built it as a pair and
recorded no `>>xxx<<` target token. Its vocabulary carries five: `>>ara<<`,
`>>ara_Latn<<`, `>>arq<<`, `>>arq_Latn<<` and `>>arz<<`. Without one the model
is not told which variety to write, and it does not fail - it answers.

Measured on `TigreGotico/opus-mt-en-ar-onnx`, int8:

    no token    "cancel the alarm" -> "هذا"          ("this")
    no token    "tell me a joke"   -> "تُخبرُني a يَمْزحُ"  (a gloss, English "a" left in)
    >>ara<<     "cancel the alarm" -> "ألغِ المنبه"
    >>ara<<     "tell me a joke"   -> "قل لي نكتة"

So the token is recorded in the registry entry, where
`TranslationModel.translate` already reads it before the pair check. The
generator picks it from the export's own vocabulary; the vocabularies below are
the `>>xxx<<` keys of the real exports, and the expected tokens were measured,
not derived from the rule.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from linguonnx import model_manager
from linguonnx.detect.labels import to_bcp47

SYNC_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_registry.py"

#: The `>>xxx<<` keys of each real export's vocab.json, and the token that
#: measured correct for the pair. `None` means the model is correct with no
#: token and the generator must not pick one: `>>cmn<<` alone damages en-zh
#: ("cancel the alarm" -> "取消提醒@ info: whatsthis").
VOCABS = {
    ("opus-mt-en-ar", "ar"): (
        [">>ara<<", ">>ara_Latn<<", ">>arq<<", ">>arq_Latn<<", ">>arz<<"], ">>ara<<"),
    ("opus-mt-en-zh", "zh"): (
        [">>cmn<<", ">>cmn_Hans<<", ">>cmn_Hant<<", ">>nan<<", ">>wuu<<",
         ">>yue<<", ">>yue_Hans<<", ">>yue_Hant<<"], None),
    ("opus-mt-en-vi", "vi"): ([">>vie<<"], ">>vie<<"),
    ("opus-mt-en-bg", "bg"): ([">>bul<<", ">>bul_Latn<<"], ">>bul<<"),
    ("opus-mt-en-pt", "pt"): ([">>pob<<", ">>por<<"], ">>por<<"),
    ("opus-mt-en-de", "de"): ([], None),
}


@pytest.fixture(scope="module")
def sync_registry():
    spec = importlib.util.spec_from_file_location("sync_registry", SYNC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("repo,tgt", list(VOCABS))
def test_the_generator_picks_the_measured_token(sync_registry, monkeypatch, repo, tgt):
    keys, expected = VOCABS[(repo, tgt)]
    vocab = {key: index for index, key in enumerate(["<pad>", "▁the", *keys])}
    monkeypatch.setattr(sync_registry, "raw_file",
                        lambda repo_id, path: json.dumps(vocab))
    assert sync_registry._pair_target_token(
        f"TigreGotico/{repo}-onnx", ["vocab.json"], tgt) == expected


def test_an_export_without_a_vocab_gets_no_token(sync_registry):
    assert sync_registry._pair_target_token("x/y", ["config.json"], "ar") is None


@pytest.mark.parametrize("repo,tgt", list(VOCABS))
def test_the_committed_entries_record_the_measured_token(repo, tgt):
    entries = model_manager.list_models(kind="translate")
    expected = VOCABS[(repo, tgt)][1]
    for model_id in (repo, f"{repo}-int8"):
        assert entries[model_id].get("target_token") == expected, model_id


def test_a_recorded_target_token_is_one_the_model_could_carry():
    """Every recorded token keeps the `>>xxx<<` shape the vocabulary uses."""
    for model_id, entry in model_manager.list_models(kind="translate").items():
        token = entry.get("target_token")
        if token is None:
            continue
        assert token.startswith(">>") and token.endswith("<<"), (
            f"{model_id} records {token!r}, which is not a Marian target "
            f"token; a plain word is a word in the source text")


def test_a_pair_token_selects_the_target_not_the_source():
    for model_id, entry in model_manager.list_models(kind="translate").items():
        token = entry.get("target_token")
        pair = entry.get("pair")
        if token is None or not pair:
            continue
        raw = token[2:-2]
        # `to_bcp47` returns the tag unchanged when it cannot map it, so it
        # needs no fallback here.
        tag = to_bcp47(raw)
        assert tag == pair[1], (
            f"{model_id} records {token!r} for the pair {pair}; the token "
            f"must select the target, not the source")
