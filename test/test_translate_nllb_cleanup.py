"""An NLLB export that sets ``clean_up_tokenization_spaces`` gets it applied.

IbRo-nllb emits a separate ``▁.`` piece in Asturian, so SentencePiece decodes
"pal próximu añu ." with a space before the full stop. The export's
``tokenizer_config.json`` sets ``clean_up_tokenization_spaces: true``, and
``transformers`` then removes that space. linguonnx did not read the file, and
5 of 35 reference sentences differed by those spaces alone.
"""
import json

import pytest
import sentencepiece
from transformers import PreTrainedTokenizerBase

from linguonnx.model_manager import list_models
from linguonnx.translate.tokenizers import clean_up_tokenization, load_tokenizer

LANGS = ["spa_Latn", "ast_Latn"]
SENTENCE = "el presupuestu pal próximu añu ."


@pytest.fixture(scope="module")
def tiny_spm(tmp_path_factory):
    path = tmp_path_factory.mktemp("spm-cleanup")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join([SENTENCE, "llovió tola nueche , pero ?"] * 40),
                      encoding="utf-8")
    sentencepiece.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(path / "tiny"), vocab_size=60,
        hard_vocab_limit=False, pad_id=1, eos_id=2, unk_id=3, bos_id=0)
    return path / "tiny.model"


def _tokenizer(tiny_spm, tmp_path, config):
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"additional_special_tokens": LANGS}), encoding="utf-8")
    files = {"spm": tiny_spm,
             "special_tokens_map": tmp_path / "special_tokens_map.json"}
    if config is not None:
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps(config), encoding="utf-8")
        files["tokenizer_config"] = tmp_path / "tokenizer_config.json"
    return load_tokenizer("nllb", files, [])


def _round_trip(tokenizer):
    ids = tokenizer.encode(SENTENCE, "spa_Latn")
    return tokenizer.decode(ids)


def test_the_export_flag_removes_the_space_before_punctuation(tiny_spm, tmp_path):
    tokenizer = _tokenizer(tiny_spm, tmp_path,
                           {"clean_up_tokenization_spaces": True})
    assert _round_trip(tokenizer) == "el presupuestu pal próximu añu."


@pytest.mark.parametrize("config", [None, {"clean_up_tokenization_spaces": False}])
def test_without_the_flag_the_decoded_text_is_unchanged(tiny_spm, tmp_path, config):
    assert _round_trip(_tokenizer(tiny_spm, tmp_path, config)) == SENTENCE


@pytest.mark.parametrize("text", [
    "pal próximu añu .", "llovió tola nueche , pero ?", "isn 't it !",
    "I 'm here , you 're there and we 've gone .", "l ' home", "a.b , c",
])
def test_the_rule_is_the_transformers_rule(text):
    assert clean_up_tokenization(text) \
        == PreTrainedTokenizerBase.clean_up_tokenization(text)


@pytest.mark.parametrize("model_id", ["IbRo-nllb", "IbRo-nllb-int8",
                                      "aina-translator-es-ast-int8"])
def test_the_registry_ships_the_tokenizer_config(model_id):
    side_files = list_models(kind="translate")[model_id]["side_files"]
    assert side_files["tokenizer_config"].endswith("tokenizer_config.json")
