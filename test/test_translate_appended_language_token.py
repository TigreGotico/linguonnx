"""An appended language token has an id that cannot be counted to.

NLLB ships no ``added_tokens.json``, so the language block was derived by
counting through ``special_tokens_map.json``: ``lang_block_start + i``. That
holds only while the names and the ids agree.

A Projecte Aina fine-tune **appends** its new language to the base NLLB
tokenizer, after the base tokenizer's own ``<mask>``. Measured over the
network on ``TigreGotico/aina-translator-es-oc-onnx`` and
``TigreGotico/aina-translator-es-an-onnx``, 2026-09-28:

    additional_special_tokens           203, the new code last
    block start (256000 + 1)            256001
    counted id for arn_Latn/arg_Latn    256203
    what 256203 really is               <mask>
    real id, stated in tokenizer.json   256204

So the model was told to begin its answer in ``<mask>``. It answered with a
fragment: ``cancelar la alarma`` -> ``alarma``, the verb gone. And when the
Aragonese model emitted 256204 itself, that id was in no special set, so the
decoder asked SentencePiece for piece 256203 and raised
``IndexError: OUT_OF_RANGE: piece id is out of range``.

``tokenizer.json`` states every id, so it is read rather than counted.
"""
import json

import pytest

from linguonnx.translate.tokenizers import (
    SpmSeq2SeqTokenizer,
    load_tokenizer,
    tokenizer_json_added_tokens,
)

#: The shape of the real export, in miniature: a base block, the base
#: tokenizer's own `<mask>` after it, and the fine-tune's language last.
BASE_BLOCK = ["ace_Arab", "spa_Latn", "cat_Latn", "zul_Latn"]
APPENDED = "arn_Latn"


@pytest.fixture(scope="module")
def tiny_spm(tmp_path_factory):
    spm = pytest.importorskip("sentencepiece")
    path = tmp_path_factory.mktemp("spm-aina")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join(["hola mon", "bon dia", "adiu lo monde"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(path / "tiny"), vocab_size=40,
        hard_vocab_limit=False, pad_id=1, eos_id=2, unk_id=3, bos_id=0)
    return path / "tiny.model"


def _block_start(tiny_spm):
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(tiny_spm))
    return sp.get_piece_size() + 1


def _files(tiny_spm, tmp_path, with_tokenizer_json=True):
    """A fine-tune's own file shape: the appended code last in the names,
    and one id past ``<mask>`` in ``tokenizer.json``."""
    names = BASE_BLOCK + [APPENDED]
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"additional_special_tokens": names}), encoding="utf-8")
    files = {"spm": tiny_spm,
             "special_tokens_map": tmp_path / "special_tokens_map.json"}
    if not with_tokenizer_json:
        return files
    start = _block_start(tiny_spm)
    added = [{"id": 0, "content": "<s>"}, {"id": 1, "content": "<pad>"},
             {"id": 2, "content": "</s>"}, {"id": 3, "content": "<unk>"}]
    added += [{"id": start + i, "content": code}
              for i, code in enumerate(BASE_BLOCK)]
    mask_id = start + len(BASE_BLOCK)
    added.append({"id": mask_id, "content": "<mask>"})
    added.append({"id": mask_id + 1, "content": APPENDED})
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"added_tokens": added}), encoding="utf-8")
    files["tokenizer_json"] = tmp_path / "tokenizer.json"
    return files


def test_the_appended_language_gets_the_id_the_export_states(tiny_spm,
                                                             tmp_path):
    """The defect, in miniature: `arn_Latn` must not reach `<mask>`."""
    files = _files(tiny_spm, tmp_path)
    tokenizer = load_tokenizer("nllb", files, [])
    start = _block_start(tiny_spm)
    counted = start + len(BASE_BLOCK)          # what counting gives it
    assert tokenizer.lang_id(APPENDED) == counted + 1
    assert tokenizer.lang_id(APPENDED) != counted


def test_the_base_block_does_not_move(tiny_spm, tmp_path):
    """A control: reading the ids must not shift the languages that were
    already right, which is every one before the appended token."""
    files = _files(tiny_spm, tmp_path)
    tokenizer = load_tokenizer("nllb", files, [])
    start = _block_start(tiny_spm)
    for offset, code in enumerate(BASE_BLOCK):
        assert tokenizer.lang_id(code) == start + offset


def test_the_mask_token_is_never_handed_out_as_a_language(tiny_spm, tmp_path):
    """`<mask>` sits inside the block and is not a language."""
    files = _files(tiny_spm, tmp_path)
    tokenizer = load_tokenizer("nllb", files, [])
    assert "<mask>" not in tokenizer.lang_code_to_id
    mask_id = _block_start(tiny_spm) + len(BASE_BLOCK)
    assert mask_id not in set(tokenizer.lang_code_to_id.values())


def test_a_token_past_the_spm_table_is_stripped_not_decoded(tiny_spm,
                                                            tmp_path):
    """The `an` crash: the model emits its own language token, and decoding
    it asked SentencePiece for a piece far past its table."""
    files = _files(tiny_spm, tmp_path)
    tokenizer = load_tokenizer("nllb", files, [])
    # The id the export itself states for the appended language - which is
    # what the Aragonese model emits, whatever the loader mapped it to.
    stated, _ = tokenizer_json_added_tokens(files["tokenizer_json"])
    assert tokenizer.decode([stated[APPENDED], tokenizer.eos_id]) == ""
    mask_id = _block_start(tiny_spm) + len(BASE_BLOCK)
    assert tokenizer.decode([mask_id, tokenizer.eos_id]) == ""


def test_the_declared_check_still_reads_the_stated_ids(tiny_spm, tmp_path):
    """A registry claim the export cannot carry still raises, and one it can
    still passes."""
    files = _files(tiny_spm, tmp_path)
    load_tokenizer("nllb", files, ["arn"])       # an alias of arn_Latn
    with pytest.raises(ValueError, match="no language token"):
        load_tokenizer("nllb", files, ["zzz_Zzzz"])


def test_names_and_ids_that_disagree_raise(tiny_spm, tmp_path):
    """Two files that disagree about what the export carries is a defect in
    the export, not something to repair by counting."""
    files = _files(tiny_spm, tmp_path)
    data = json.loads((tmp_path / "tokenizer.json").read_text())
    data["added_tokens"] = [t for t in data["added_tokens"]
                            if t["content"] != APPENDED]
    (tmp_path / "tokenizer.json").write_text(json.dumps(data),
                                             encoding="utf-8")
    with pytest.raises(ValueError, match="states no token"):
        load_tokenizer("nllb", files, [])


def test_an_export_without_a_tokenizer_json_is_unchanged(tiny_spm, tmp_path):
    """The control for every NLLB export that ships no `tokenizer.json`:
    the positional block is still what it was."""
    files = _files(tiny_spm, tmp_path, with_tokenizer_json=False)
    tokenizer = load_tokenizer("nllb", files, [])
    start = _block_start(tiny_spm)
    for offset, code in enumerate(BASE_BLOCK + [APPENDED]):
        assert tokenizer.lang_id(code) == start + offset


def test_added_tokens_json_still_outranks_tokenizer_json(tiny_spm, tmp_path):
    """M2M100's path must not move: `added_tokens.json` stays authoritative."""
    # Called for the `tokenizer.json` it writes into `tmp_path`, which the
    # tokenizer below reads; the returned mapping is not used here.
    _files(tiny_spm, tmp_path)
    (tmp_path / "added_tokens.json").write_text(
        json.dumps({code: 900 + i for i, code in enumerate(BASE_BLOCK)}),
        encoding="utf-8")
    tokenizer = SpmSeq2SeqTokenizer(
        tiny_spm, BASE_BLOCK, fairseq_offset=1,
        added_tokens_path=tmp_path / "added_tokens.json",
        tokenizer_json_path=tmp_path / "tokenizer.json")
    assert tokenizer.lang_id(BASE_BLOCK[0]) == 900


def test_the_reader_refuses_a_file_with_no_language_at_all(tmp_path):
    """A `tokenizer.json` that states only bracketed specials must raise
    rather than let the caller fall back to counting."""
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 0, "content": "<s>"},
                                     {"id": 1, "content": "<pad>"}]}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="could be a language"):
        tokenizer_json_added_tokens(tmp_path / "tokenizer.json")
