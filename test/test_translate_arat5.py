"""arat5-arabic-dialects-translation: a T5 with a tokenizer.json and no prefix.

The offline tests pin the registry entry, the architecture the crawler reads
off the export, and the two ways the `tokenizers` path differs from a bare
`Tokenizer.from_file`. The network test runs the published fp32 export against
token ids the upstream PyTorch checkpoint produced for the same sentences,
greedy, and committed in ``test/data/arat5_reference.json``.
"""
import json
import sys
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate.models import capability_from_entry, load_model
from linguonnx.translate.tokenizers import FastUnigramTokenizer, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sync_registry  # noqa: E402

MODEL_ID = "arat5-arabic-dialects-translation"
SOURCES = {"acm", "aeb", "afb", "apc", "apd", "arq", "ars", "ary", "arz",
           "ayl", "ayn", "ayp"}
REFERENCE = json.loads((ROOT / "test" / "data" / "arat5_reference.json")
                       .read_text(encoding="utf-8"))
TRANSLATE = list_models(kind="translate")


@pytest.mark.parametrize("model_id", [MODEL_ID, MODEL_ID + "-int8"])
def test_registered_as_dialects_into_standard_arabic_only(model_id):
    entry = TRANSLATE[model_id]
    assert entry["arch"] == "t5-fast"
    assert entry["extra"] == "fast-tokenizers"
    assert entry["hf_repo"] == "TigreGotico/arat5-arabic-dialects-translation-onnx"
    capability = capability_from_entry(entry)
    assert capability.src_languages == frozenset(SOURCES)
    assert capability.tgt_languages == frozenset({"ar"})
    assert capability.covers("arz", "ar")
    assert capability.covers("ayp", "ar")
    assert not capability.covers("ar", "arz")
    assert not capability.covers("arz", "apc")


def test_not_skipped_any_more():
    skipped = json.loads((ROOT / "linguonnx" / "model_index" / "skipped.json")
                         .read_text(encoding="utf-8"))
    assert "TigreGotico/arat5-arabic-dialects-translation-onnx" not in skipped
    assert MODEL_ID not in sync_registry._KNOWN_ARCH_MISMATCH


def test_arch_reads_t5_fast_only_for_a_curated_tokenizer_json_export():
    detail = {"config": {"model_type": "t5"}}
    files = {"tokenizer.json": 1, "config.json": 1}
    assert sync_registry._arch(detail, files, MODEL_ID) == "t5-fast"
    with pytest.raises(sync_registry.SkipRepo, match="names no languages"):
        sync_registry._arch(detail, files, "some-other-t5")
    madlad = {"spiece.model": 1, "config.json": 1}
    assert sync_registry._arch(detail, madlad, "madlad400-3b-mt") == "madlad"


def _word_level(tmp_path, truncate_at=None):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, ".": 3}
    vocab.update({word: 4 + i for i, word in enumerate("abcdefgh")})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if truncate_at:
        tokenizer.enable_truncation(truncate_at)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def test_a_truncation_setting_in_tokenizer_json_does_not_cut_the_input(tmp_path):
    """The AraT5 tokenizer.json truncates at 64 tokens; `transformers` does not
    apply that unless asked, so a long sentence reaches the encoder whole."""
    path = _word_level(tmp_path, truncate_at=4)
    tokenizer = FastUnigramTokenizer(path, pad_token="<pad>")
    assert tokenizer.encode("a b c d e f g h") == [4, 5, 6, 7, 8, 9, 10, 11, 1]


def test_t5_fast_decoding_applies_the_configured_clean_up(tmp_path):
    path = _word_level(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": True}))
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "</s>", "pad_token": "<pad>",
                    "unk_token": "<unk>"}))
    tokenizer = load_tokenizer("t5-fast", {
        "tokenizer_json": path,
        "tokenizer_config": tmp_path / "tokenizer_config.json",
        "special_tokens_map": tmp_path / "special_tokens_map.json"}, ())
    assert tokenizer.encode("a b .") == [4, 5, 3, 1]
    assert tokenizer.decode([0, 4, 5, 3, 1]) == "a b."


def test_reference_fixture_covers_every_registered_source():
    assert {row["src"] for row in REFERENCE} == SOURCES


@pytest.mark.network
def test_fp32_export_matches_the_pytorch_reference_token_for_token():
    model = load_model(MODEL_ID)
    translator = load_translator(models=[MODEL_ID], precision="fp32",
                                 max_model_mb=None)
    for row in REFERENCE:
        ids = model.pipeline.encode(model, row["text"], row["src"], "ar")
        assert ids == row["input_ids"], row["source"]
        assert list(model.decoder.generate(ids, config=_greedy())) \
            == row["ref_ids"], row["source"]
        assert translator.translate(row["text"], row["src"], "ar",
                                    model=MODEL_ID, num_beams=1) \
            == row["ref"], row["source"]


def _greedy():
    from linguonnx.translate.decode import GenerationConfig
    return GenerationConfig(num_beams=1)
