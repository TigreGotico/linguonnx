"""The registry's routing spelling is not the model's own spelling.

Two published models were unusable at once for the same reason, and neither
failure was visible from the registry alone:

* every IndicTrans2 entry raised ``'en' is not an IndicTrans2 language tag``
  on every call, because the entries were rewritten into BCP-47 without
  keeping the FLORES tags the vocabulary is written in;
* ``aina-translator-ca-zh`` raised ``language code 'ca' is not supported by
  this model``, because a bilingual entry carries no ``languages`` list and
  the language-token block was therefore built from an empty sequence.

Both are checked here without a network and without ``transformers``. The
integration suite already had a guard for the first one, but it sits behind
``pytest.importorskip("optimum.onnxruntime")`` and so was skipped in every
environment that did not happen to have torch installed - which is how a
broken build shipped with a green suite. Nothing in this module is skippable.
"""

import json

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate._indic_processor import LANGUAGE_TAGS
from linguonnx.translate.models import TranslationModel
from linguonnx.translate.tokenizers import artifact_lang_codes

REGISTRY = list_models("translate")


def _entries(predicate):
    return sorted(model_id for model_id, entry in REGISTRY.items()
                  if predicate(entry))


INDICTRANS2 = _entries(lambda e: e.get("arch") == "indictrans2")
SPM_LANG_TOKEN = _entries(lambda e: e.get("arch") in ("m2m100", "nllb"))


def test_the_registry_still_has_indictrans2_entries():
    """Guards the two tests below from passing because they found nothing."""
    assert INDICTRANS2


@pytest.mark.parametrize("model_id", INDICTRANS2)
def test_indictrans2_entries_map_every_language_to_a_flores_tag(model_id):
    """`native_code` must answer with a tag `IndicProcessor` accepts.

    ``preprocess`` puts the two tags straight onto the model input and rejects
    anything outside ``LANGUAGE_TAGS``; a BCP-47 ``en`` there is not a
    degraded translation, it is a ``ValueError`` on every single call.
    """
    entry = REGISTRY[model_id]
    model = TranslationModel(model_id, entry=entry)
    advertised = set(entry.get("src_languages") or ()) \
        | set(entry.get("tgt_languages") or ()) \
        | set(entry.get("languages") or ())
    assert advertised, f"{model_id} advertises no languages at all"
    for tag in sorted(advertised):
        native = model.native_code(tag)
        assert native in LANGUAGE_TAGS, (
            f"{model_id} routes {tag!r} to {native!r}, which IndicTrans2's "
            f"vocabulary does not carry; preprocess() raises on it")


@pytest.mark.parametrize("model_id", SPM_LANG_TOKEN)
def test_language_token_models_can_build_a_language_block(model_id):
    """A language-token model must have codes to build its block from.

    M2M100 and NLLB select the source language with a token on the input and
    force the target token on the output. With no codes there is no block, and
    ``lang_id`` raises for every language the entry claims. A bilingual entry
    has no ``languages`` list by design, so the export itself has to be
    readable - which means ``special_tokens_map.json`` must be fetched.
    """
    entry = REGISTRY[model_id]
    if entry.get("languages"):
        return
    assert "special_tokens_map" in entry.get("side_files", {}), (
        f"{model_id} lists no languages and does not fetch "
        f"special_tokens_map.json, so its language-token block would be "
        f"empty and every call would raise")


class _FakeSpm:
    """Just enough SentencePiece for the language-token block arithmetic."""

    def get_piece_size(self):
        return 256_000

    def encode(self, text, out_type=str):
        return list(text.split())

    def PieceToId(self, piece):
        return 7

    def IdToPiece(self, token_id):
        return "x"


@pytest.mark.parametrize("arch,tokens,wanted", [
    ("nllb", ["spa_Latn", "cat_Latn", "zho_Hans"], "cat_Latn"),
    ("m2m100", ["__ca__", "__zh__"], "ca"),
])
def test_a_bilingual_entry_still_gets_a_language_block(tmp_path, monkeypatch,
                                                       arch, tokens, wanted):
    """`load_tokenizer` with no registry ``languages`` must read the export.

    Regression test for ``aina-translator-ca-zh``: a bilingual entry carries a
    ``pair`` and no ``languages``, so ``lang_codes`` arrives empty. Before the
    fix the language-token block was then built from that empty sequence and
    every call raised ``KeyError: language code 'ca' is not supported by this
    model`` - on a model whose own ``special_tokens_map.json`` lists it.
    """
    from linguonnx.translate import tokenizers as mod

    special = tmp_path / "special_tokens_map.json"
    special.write_text(json.dumps({"additional_special_tokens": tokens}),
                       encoding="utf-8")
    files = {"spm": tmp_path / "sentencepiece.bpe.model",
             "special_tokens_map": special}
    if arch == "m2m100":
        vocab = tmp_path / "vocab.json"
        vocab.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
        files["vocab"] = vocab
    monkeypatch.setattr(mod, "_load_spm", lambda path: _FakeSpm())

    tokenizer = mod.load_tokenizer(arch, files, (), pair=("ca", "zh"))
    assert tokenizer.lang_id(wanted) >= 0


def test_artifact_lang_codes_unwraps_m2m100_tokens(tmp_path):
    path = tmp_path / "special_tokens_map.json"
    path.write_text(json.dumps({
        "additional_special_tokens": ["__af__", "__ca__", "__zh__"],
        "eos_token": "</s>",
    }), encoding="utf-8")
    assert artifact_lang_codes({"special_tokens_map": path}) == \
        ["af", "ca", "zh"]


def test_artifact_lang_codes_leaves_flores_tags_alone(tmp_path):
    path = tmp_path / "special_tokens_map.json"
    path.write_text(json.dumps({
        "additional_special_tokens": ["spa_Latn", "ast_Latn"],
    }), encoding="utf-8")
    assert artifact_lang_codes({"special_tokens_map": path}) == \
        ["spa_Latn", "ast_Latn"]


def test_artifact_lang_codes_refuses_to_invent_an_empty_block(tmp_path):
    path = tmp_path / "special_tokens_map.json"
    path.write_text(json.dumps({"eos_token": "</s>"}), encoding="utf-8")
    with pytest.raises(ValueError):
        artifact_lang_codes({"special_tokens_map": path})
    with pytest.raises(ValueError):
        artifact_lang_codes({})
