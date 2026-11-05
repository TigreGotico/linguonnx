"""NLLB addresses a language by the export's own token order.

NLLB ships no ``added_tokens.json``, so the language block is positional:
``lang_block_start + i`` over the list the tokenizer is given. The registry's
``languages`` list is normalised BCP-47 in its own sort order, which is NOT
the export's order, and handing it to the tokenizer put every language after
the first divergence on another language's token.

Measured on ``TigreGotico/nllb-200-distilled-600M-onnx`` before this fix: a
request for ``kab`` forced id 256085, which that export spells ``kas_Deva``,
Kashmiri in the Devanagari script. 148 of its 202 languages were affected.
``en`` forced ``epo_Latn`` and ``fr`` forced ``gla_Latn``.

So the export's order is authoritative. The registry entry's
``native_codes`` table maps each BCP-47 tag to the export's own code, and the
tokenizer knows only those codes. A name that matches no token raises at
load: nothing is repaired quietly.
"""
import json

import pytest

from linguonnx.translate.tokenizers import (SpmSeq2SeqTokenizer,
                                            artifact_lang_codes,
                                            load_tokenizer)

#: The head of the real export's block, in its own order, with the two codes
#: the Kabyle defect turns on. `kab_Latn` sits at index 79 of 202; the
#: registry's sorted BCP-47 list puts `kab` at index 84, where the export has
#: `kas_Deva`.
EXPORT_HEAD = ["ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab",
               "afr_Latn", "epo_Latn", "eng_Latn", "fra_Latn", "gla_Latn",
               "kab_Latn", "kac_Latn", "kas_Deva", "kas_Arab", "por_Latn",
               "quy_Latn", "deu_Latn", "cat_Latn"]

#: What the registry advertises for the same model: normalised BCP-47 of the
#: same tokens, sorted, so in a different order from the block above. This is
#: how the real entry is built (``scripts/sync_registry._multilingual_languages``).
REGISTRY_TAGS = sorted(["ace-Arab", "ace", "acm", "acq", "aeb", "af", "eo",
                        "en", "fr", "gd", "kab", "kac", "ks-Deva", "ks",
                        "pt", "quy", "de", "ca"])

#: The entry's ``native_codes`` for those tags, written out by hand.
NATIVE = {"ace-Arab": "ace_Arab", "ace": "ace_Latn", "acm": "acm_Arab",
          "acq": "acq_Arab", "aeb": "aeb_Arab", "af": "afr_Latn",
          "eo": "epo_Latn", "en": "eng_Latn", "fr": "fra_Latn",
          "gd": "gla_Latn", "kab": "kab_Latn", "kac": "kac_Latn",
          "ks-Deva": "kas_Deva", "ks": "kas_Arab", "pt": "por_Latn",
          "quy": "quy_Latn", "de": "deu_Latn", "ca": "cat_Latn"}

#: What the model passes to the tokenizer: the native codes, in registry order.
DECLARED = [NATIVE[tag] for tag in REGISTRY_TAGS]


@pytest.fixture(scope="module")
def tiny_spm(tmp_path_factory):
    """A real SentencePiece model. Its content is irrelevant; ids are."""
    spm = pytest.importorskip("sentencepiece")
    path = tmp_path_factory.mktemp("spm-nllb")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join(["hello world", "bonjour le monde",
                                 "azul fellawen", "ola mundo"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(path / "tiny"), vocab_size=48,
        hard_vocab_limit=False, pad_id=1, eos_id=2, unk_id=3, bos_id=0)
    return path / "tiny.model"


def _fake_nllb_export(tmp_path, codes=EXPORT_HEAD):
    """``special_tokens_map.json`` shaped like a real NLLB export."""
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"additional_special_tokens": list(codes)}),
        encoding="utf-8")
    return tmp_path / "special_tokens_map.json"


def _files(tiny_spm, tmp_path, codes=EXPORT_HEAD):
    return {"spm": tiny_spm, "special_tokens_map":
            _fake_nllb_export(tmp_path, codes)}


def _position(tokenizer, code):
    """The id the export gives *code*: its index in the export's block."""
    return tokenizer.lang_id(EXPORT_HEAD[0]) + EXPORT_HEAD.index(code)


def test_a_request_addresses_the_token_that_carries_its_own_code(tiny_spm,
                                                                 tmp_path):
    """The defect, in miniature: `kab` must not reach `kas_Deva`."""
    tokenizer = load_tokenizer("nllb", _files(tiny_spm, tmp_path), DECLARED)
    for tag in ("kab", "en", "fr", "pt", "de", "ca"):
        assert tokenizer.lang_id(NATIVE[tag]) == \
            _position(tokenizer, NATIVE[tag]), tag
    assert tokenizer.lang_id(NATIVE["kab"]) != _position(tokenizer, "kas_Deva")


def test_the_ids_are_the_export_s_own_block(tiny_spm, tmp_path):
    """Order comes from the artifact, never from the registry."""
    files = _files(tiny_spm, tmp_path)
    tokenizer = load_tokenizer("nllb", files, DECLARED)
    artifact = artifact_lang_codes(files)
    base = tokenizer.lang_id(artifact[0])
    for offset, code in enumerate(artifact):
        assert tokenizer.lang_id(code) == base + offset


def test_the_registry_order_alone_no_longer_decides_anything(tiny_spm,
                                                             tmp_path):
    """Shuffling the registry list must not move a single id."""
    files = _files(tiny_spm, tmp_path)
    straight = load_tokenizer("nllb", files, DECLARED)
    shuffled = load_tokenizer("nllb", files, list(reversed(DECLARED)))
    for code in DECLARED:
        assert straight.lang_id(code) == shuffled.lang_id(code)


def test_every_registry_row_names_the_token_of_its_own_language(tiny_spm,
                                                               tmp_path):
    """Each row of the table names a token of this export, and that token's
    own code normalises back to the row's tag."""
    from linguonnx.detect.labels import to_bcp47

    tokenizer = load_tokenizer("nllb", _files(tiny_spm, tmp_path), DECLARED)
    for tag, code in NATIVE.items():
        assert code in tokenizer.lang_code_to_id, tag
        assert to_bcp47(code) == tag, f"{tag} names {code}"


def test_a_claim_the_export_cannot_carry_raises(tiny_spm, tmp_path):
    """A name that matches no token is refused, never repaired."""
    files = _files(tiny_spm, tmp_path)
    with pytest.raises(ValueError) as excinfo:
        load_tokenizer("nllb", files, DECLARED + ["xx-Zzzz"])
    assert "xx-Zzzz" in str(excinfo.value)


def test_a_bcp47_tag_is_not_a_token_name(tiny_spm, tmp_path):
    """The tokenizer knows the export's codes only. A tag reaches a token
    through the registry's ``native_codes`` table, not through the
    tokenizer."""
    files = _files(tiny_spm, tmp_path)
    tokenizer = load_tokenizer("nllb", files, DECLARED)
    assert set(tokenizer.lang_code_to_id) == set(EXPORT_HEAD)
    with pytest.raises(KeyError):
        tokenizer.lang_id("kab")
    with pytest.raises(ValueError):
        load_tokenizer("nllb", files, REGISTRY_TAGS)


def test_the_declared_check_is_off_when_nothing_is_declared(tiny_spm,
                                                            tmp_path):
    """A bilingual entry declares no languages and must still load."""
    tokenizer = load_tokenizer("nllb", _files(tiny_spm, tmp_path), ())
    assert tokenizer.lang_id("kab_Latn") == _position(tokenizer, "kab_Latn")


def test_the_m2m100_path_is_untouched(tiny_spm, tmp_path):
    """m2m100 reads added_tokens.json, which stays authoritative."""
    vocab = {f"piece{i}": i for i in range(100)}
    added = {"__fr__": 100, "__pt__": 101}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "added_tokens.json").write_text(json.dumps(added),
                                                encoding="utf-8")
    files = {"spm": tiny_spm, "vocab": tmp_path / "vocab.json",
             "added_tokens": tmp_path / "added_tokens.json"}
    tokenizer = load_tokenizer("m2m100", files, ["fr", "pt"])
    assert tokenizer.lang_id("fr") == 100
    assert tokenizer.lang_id("pt") == 101
    assert isinstance(tokenizer, SpmSeq2SeqTokenizer)
