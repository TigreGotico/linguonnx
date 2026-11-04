"""mBART: the source tag follows ``</s>``, the target tag starts the decoder.

``transformers``' ``MBartTokenizer`` encodes ``pieces + [</s>, src_lang]``,
with SentencePiece ids shifted by the fairseq offset of 1 and one language
token per code after the vocabulary. VinAI's usage selects English with
``decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"]``: the target
tag is the decoder's first input, and the config names no start token.

The offline tests below derive every expected id from SentencePiece itself.
The network tests compare against ids and text that ``transformers`` produced
from the upstream PyTorch checkpoint, committed in
``test/data/mbart_vinai_vi2en_reference.json``.
"""
import json
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate.decode import GenerationConfig
from linguonnx.translate.models import TranslationModel
from linguonnx.translate.preprocess import pipeline_for
from linguonnx.translate.tokenizers import load_tokenizer

#: mBART-cc25's language block, in the export's own order.
CC25 = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI",
        "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR",
        "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU",
        "si_LK", "tr_TR", "vi_VN", "zh_CN"]

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "mbart_vinai_vi2en_reference.json")
    .read_text(encoding="utf-8"))

MODEL_ID = "vinai-translate-vi2en-v2"


@pytest.fixture(scope="module")
def tiny_spm(tmp_path_factory):
    spm = pytest.importorskip("sentencepiece")
    path = tmp_path_factory.mktemp("spm-mbart")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join(["xin chào thế giới", "hello world",
                                 "chúng tôi đi học", "we go to school"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(path / "tiny"), vocab_size=48,
        hard_vocab_limit=False)
    return path / "tiny.model"


def _tokenizer(tiny_spm, tmp_path):
    special = tmp_path / "special_tokens_map.json"
    special.write_text(json.dumps({"additional_special_tokens": CC25}),
                       encoding="utf-8")
    return load_tokenizer("mbart", {"spm": tiny_spm,
                                    "special_tokens_map": special}, [])


def test_the_source_tag_follows_the_end_of_sentence_token(tiny_spm, tmp_path):
    import sentencepiece as spm

    tokenizer = _tokenizer(tiny_spm, tmp_path)
    sp = spm.SentencePieceProcessor(model_file=str(tiny_spm))
    text = "chúng tôi đi học"
    pieces = [sp.piece_to_id(p) + 1 for p in sp.encode(text, out_type=str)]
    vi_vn = sp.get_piece_size() + 1 + CC25.index("vi_VN")
    assert tokenizer.encode(text, "vi_VN") == pieces + [2, vi_vn]


def test_language_ids_follow_the_offset_vocabulary(tiny_spm, tmp_path):
    import sentencepiece as spm

    tokenizer = _tokenizer(tiny_spm, tmp_path)
    size = spm.SentencePieceProcessor(model_file=str(tiny_spm)).get_piece_size()
    assert tokenizer.lang_id("ar_AR") == size + 1
    assert tokenizer.lang_id("en_XX") == size + 1 + 3
    assert tokenizer.lang_id("zh_CN") == size + 1 + 24


def test_the_target_tag_starts_the_decoder_and_is_not_forced(tiny_spm,
                                                            tmp_path):
    tokenizer = _tokenizer(tiny_spm, tmp_path)

    class Model:
        def native_code(self, tag):
            return {"vi": "vi_VN", "en": "en_XX"}[tag]

    model = Model()
    model.tokenizer = tokenizer
    pipeline = pipeline_for("mbart")
    assert pipeline.decoder_start(model, "en") == tokenizer.lang_id("en_XX")
    assert pipeline.forced_bos(model, "en") is None


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_the_registry_sends_mbart_spellings(model_id):
    entry = list_models(kind="translate")[model_id]
    assert entry["arch"] == "mbart"
    assert entry["pair"] == ["vi", "en"]
    assert entry["native_codes"] == {"en": "en_XX", "vi": "vi_VN"}


@pytest.fixture(scope="module")
def vinai():
    return TranslationModel(MODEL_ID)


@pytest.mark.network
@pytest.mark.parametrize("row", REFERENCE["rows"], ids=lambda r: r["src"][:24])
def test_input_ids_match_transformers(vinai, row):
    assert vinai.pipeline.encode(vinai, row["src"], "vi", "en") == \
        row["input_ids"]


@pytest.mark.network
@pytest.mark.parametrize("row", REFERENCE["rows"], ids=lambda r: r["src"][:24])
def test_greedy_output_matches_transformers(vinai, row):
    config = GenerationConfig(num_beams=1, max_new_tokens=128)
    ids = vinai.pipeline.encode(vinai, row["src"], "vi", "en")
    start = vinai.pipeline.decoder_start(vinai, "en")
    generated = vinai.decoder.generate(ids, decoder_start_token_id=start,
                                       config=config)
    assert [start] + generated + [2] == row["greedy_ids"]
    assert vinai.translate(row["src"], "vi", "en", config=config) == \
        row["greedy"]
