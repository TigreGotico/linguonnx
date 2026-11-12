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
from huggingface_hub import hf_hub_download

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

#: tokenizers 0.23.3 on the published tokenizer.json, for the first three
#: reference sentences joined by spaces: 66 tokens and ``</s>``.
THREE_SENTENCE_IDS = [
    23506, 568, 18890, 597, 2514, 109598, 151, 21852, 66514, 8755, 31, 15174,
    630, 8790, 109598, 4510, 12657, 109779, 62543, 35290, 14423, 109598, 548,
    66514, 396, 24407, 1743, 109566, 31183, 560, 109598, 6848, 8755, 31,
    15174, 630, 14649, 1157, 52512, 147, 2743, 245, 715, 678, 40, 109566,
    3316, 109590, 151, 21063, 8755, 31, 50063, 630, 8790, 410, 66371, 147,
    4429, 109598, 425, 13749, 3614, 66371, 729, 823, 1]


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
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, ".": 3}
    vocab.update({word: 4 + i for i, word in enumerate("abcdefgh")})
    truncation = truncate_at and {"direction": "Right", "max_length": truncate_at,
                                  "strategy": "LongestFirst", "stride": 0}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({
        "version": "1.0", "truncation": truncation, "padding": None,
        "added_tokens": [], "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}}))
    return path


def test_a_truncation_setting_in_tokenizer_json_does_not_cut_the_input(tmp_path):
    """The AraT5 tokenizer.json truncates at 64 tokens; `transformers` does not
    apply that unless asked, so a long sentence reaches the encoder whole."""
    path = _word_level(tmp_path, truncate_at=4)
    tokenizer = FastUnigramTokenizer(path, pad_token="<pad>")
    assert tokenizer.encode("a b c d e f g h") == [4, 5, 6, 7, 8, 9, 10, 11, 1]


def test_a_unigram_vocabulary_gives_each_special_token_its_list_position(tmp_path):
    """The Softcatala Pegasus exports hold a Unigram model, whose vocabulary is
    a list of ``[piece, score]``: a token's id is its position in that list."""
    pieces = ["<blank>", "<unk>", "</s>", "\u2581a", "\u2581b"]
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [], "normalizer": None,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581",
                          "prepend_scheme": "always", "split": True},
        "post_processor": None, "decoder": None,
        "model": {"type": "Unigram", "unk_id": 1, "byte_fallback": False,
                  "vocab": [[piece, -1.0] for piece in pieces]}}))
    tokenizer = FastUnigramTokenizer(path)
    assert (tokenizer.pad_id, tokenizer.unk_id, tokenizer.eos_id) == (0, 1, 2)
    assert tokenizer.encode("a b") == [3, 4, 2]


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
def test_the_published_tokenizer_encodes_past_its_64_token_truncation():
    """The tokenizers package reads the published tokenizer.json, on 0.x and
    1.x. Each reference sentence gets the reference ids, and a 67-id input
    reaches the encoder whole although the file truncates at 64."""
    repo = TRANSLATE[MODEL_ID]["hf_repo"]
    tokenizer = load_tokenizer("t5-fast", {
        name: hf_hub_download(repo, filename) for name, filename in (
            ("tokenizer_json", "tokenizer.json"),
            ("tokenizer_config", "tokenizer_config.json"),
            ("special_tokens_map", "special_tokens_map.json"))}, ())
    assert [tokenizer.encode(row["text"]) for row in REFERENCE] \
        == [row["input_ids"] for row in REFERENCE]
    text = " ".join(row["text"] for row in REFERENCE[:3])
    assert tokenizer.encode(text) == THREE_SENTENCE_IDS


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
