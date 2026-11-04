"""An NLLB export whose ``tokenizer.json`` adds a text token.

The AfriGloss NLLB fine-tunes add ``"\\n"`` as id 256204, marked
``"special": false`` and ``normalized: true``. The normaliser maps that
content to a space, so the `tokenizers` library inserts the id at every space
of the input, and the models were trained on that encoding. SentencePiece
alone never inserts it, and the id is past the SentencePiece table, so
decoding it raised ``IndexError: OUT_OF_RANGE``.

The offline tests build the same shape in miniature. The expected ids come
from SentencePiece, one word at a time, with the text token between words.
The network tests compare the real exports with ids and text that
`transformers` produced from the upstream adapter (``afrigloss_nllb_reference.json``).
"""
import json
import re
import sys
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate.models import capability_from_entry
from linguonnx.translate.tokenizers import (
    load_tokenizer,
    tokenizer_json_added_tokens,
)

ROOT = Path(__file__).resolve().parent.parent
LANGS = ["eng_Latn", "yor_Latn"]
WORDS = ["hola", "mon", "bon", "dia"]


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    """A SentencePiece model plus a `tokenizers` file with NLLB's id layout:
    ``<s> <pad> </s> <unk>``, the pieces at ``sp_id + 1``, the languages, and
    the text token ``"\\n"`` last."""
    spm = pytest.importorskip("sentencepiece")
    tokenizers = pytest.importorskip("tokenizers")
    path = tmp_path_factory.mktemp("nllb-text-token")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join(["hola mon", "bon dia", "adiu lo monde"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(path / "tiny"), vocab_size=40,
        hard_vocab_limit=False, unk_id=0, bos_id=1, eos_id=2, pad_id=-1)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(path / "tiny.model"))

    vocab = [("<s>", 0.0), ("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0)]
    vocab += [(sp.IdToPiece(i), sp.GetScore(i))
              for i in range(3, sp.GetPieceSize())]
    fast = tokenizers.Tokenizer(tokenizers.models.Unigram(vocab, unk_id=3))
    fast.normalizer = tokenizers.normalizers.Replace("\n", " ")
    fast.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always", split=True)
    fast.decoder = tokenizers.decoders.Metaspace(
        replacement="▁", prepend_scheme="always", split=True)
    fast.add_special_tokens(LANGS)
    fast.add_tokens([tokenizers.AddedToken("\n", normalized=True,
                                           special=False)])
    fast.save(str(path / "tokenizer.json"))
    (path / "special_tokens_map.json").write_text(
        json.dumps({"additional_special_tokens": LANGS}), encoding="utf-8")
    files = {"spm": path / "tiny.model", "tokenizer_json": path / "tokenizer.json",
             "special_tokens_map": path / "special_tokens_map.json"}
    ids = {word: [i + 1 for i in sp.encode(word)] for word in WORDS}
    return files, ids, fast.token_to_id("\n"), fast.token_to_id("eng_Latn")


@pytest.fixture(scope="module")
def tokenizer(tiny_export):
    files, _, _, _ = tiny_export
    return load_tokenizer("nllb", files, LANGS)


def test_the_text_token_is_inserted_at_every_space(tiny_export, tokenizer):
    _, ids, text_id, eng = tiny_export
    expected = [eng]
    for n, word in enumerate(WORDS):
        expected += ([text_id] if n else []) + ids[word]
    expected.append(tokenizer.eos_id)
    assert tokenizer.encode(" ".join(WORDS), "eng_Latn") == expected


def test_the_text_token_decodes_as_one_space(tiny_export, tokenizer):
    _, ids, text_id, _ = tiny_export
    yor = tokenizer.lang_id("yor_Latn")
    out = [yor] + ids["hola"] + [text_id] + ids["mon"] + [text_id] \
        + ids["bon"] + ids["dia"] + [tokenizer.eos_id]
    assert tokenizer.decode(out) == "hola mon bon dia"


def test_a_text_token_at_the_edges_or_twice_adds_no_space(tiny_export,
                                                         tokenizer):
    _, ids, text_id, _ = tiny_export
    out = [text_id] + ids["hola"] + [text_id, text_id] + ids["dia"] \
        + [text_id, tokenizer.eos_id]
    assert tokenizer.decode(out) == "hola dia"


def test_a_round_trip_gives_the_input_back(tokenizer):
    text = "bon dia hola mon"
    assert tokenizer.decode(tokenizer.encode(text, "eng_Latn")) == text


def test_the_text_token_is_not_a_language(tiny_export):
    files, _, text_id, _ = tiny_export
    languages, _ = tokenizer_json_added_tokens(files["tokenizer_json"])
    assert set(languages) == set(LANGS)
    assert text_id not in languages.values()


# --------------------------------------------------------------------------
# The registered exports
# --------------------------------------------------------------------------

TRANSLATE = list_models(kind="translate")
AFRIGLOSS = ("afrigloss-nllb-600m", "afrigloss-nllb-600m-int8",
             "afrigloss-nllb-1.3b", "afrigloss-nllb-1.3b-int8")
TARGETS = {"ee", "ha", "ig", "sn", "sw", "tw", "yo"}


@pytest.mark.parametrize("model_id", AFRIGLOSS)
def test_afrigloss_nllb_covers_english_and_french_into_seven(model_id):
    entry = TRANSLATE[model_id]
    capability = capability_from_entry(entry)
    assert capability.directions == frozenset(
        (src, tgt) for src in ("en", "fr") for tgt in TARGETS)
    assert entry["native_codes"]["tw"] == "twi_Latn"
    assert entry["native_codes"]["sw"] == "swh_Latn"
    assert entry["extra"] == "fast-tokenizers"
    assert entry["side_files"]["tokenizer_json"] == "tokenizer.json"


@pytest.mark.parametrize("model_id", AFRIGLOSS)
def test_afrigloss_nllb_notes_are_the_override_text(model_id):
    sys.path.insert(0, str(ROOT / "scripts"))
    import sync_registry
    base = model_id.removesuffix("-int8")
    assert TRANSLATE[model_id]["notes"] \
        == sync_registry.MULTILINGUAL_LANGUAGE_OVERRIDES[base]["notes"]


def test_afrigloss_nllb_is_not_skipped():
    skipped = json.loads((ROOT / "linguonnx" / "model_index" / "skipped.json")
                         .read_text(encoding="utf-8"))
    assert not [repo for repo in skipped if "afrigloss-nllb" in repo]


REFERENCE = json.loads((ROOT / "test" / "data" / "afrigloss_nllb_reference.json")
                       .read_text(encoding="utf-8"))


def _export_tokenizer(model_id):
    from huggingface_hub import hf_hub_download
    entry = TRANSLATE[model_id]
    files = {key: Path(hf_hub_download(entry["hf_repo"], name))
             for key, name in entry["side_files"].items()}
    native = [entry["native_codes"][tag] for tag in entry["languages"]]
    return load_tokenizer("nllb", files, native)


@pytest.mark.network
@pytest.mark.parametrize("model_id", ["afrigloss-nllb-600m", "afrigloss-nllb-1.3b"])
def test_afrigloss_nllb_encodes_and_decodes_as_transformers(model_id):
    """Input ids equal the upstream tokenizer's. Decoded text equals the
    upstream decode with the doubled space at each text token made one."""
    tokenizer = _export_tokenizer(model_id)
    rows = REFERENCE[model_id]
    assert rows
    for row in rows:
        assert tokenizer.encode(row["text"], row["src"]) == row["input_ids"]
        expected = re.sub(" {2,}", " ", row["decoded"]).strip()
        assert tokenizer.decode(row["output_ids"]) == expected
