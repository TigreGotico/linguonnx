"""Tokenizer parity against `transformers`, plus the Moses port.

The tokenizers are the other silent-failure surface: a wrong id offset or a
missing language token does not raise, it just degrades the translation. So the
ids are compared to the reference tokenizer directly, token for token.
"""

import json
import pytest

from linguonnx.translate.tokenizers import normalize_punctuation

CASES = [
    "Hello, how are you today?",
    "The library opens at nine in the morning.",
    "Bom dia, o tempo está muito bom hoje.",
    "Кошка сидит на окне.",
    "أعلنت وزارة الخارجية اليوم عن اتفاقية جديدة.",
    "Kaixo, zer moduz zaude gaur?",
]


# --- Moses punctuation normalisation (no download needed) -----------------

def test_moses_port_matches_sacremoses():
    sacremoses = pytest.importorskip("sacremoses")
    reference = sacremoses.MosesPunctNormalizer()
    samples = CASES + [
        "“Quoted” text — with dashes – here.",
        "L’homme dit « bonjour ».",
        "50 % of it ( yes ) ;",
        "a‘b and don’t",
        "Text with  extra   spaces .",
    ]
    for text in samples:
        assert normalize_punctuation(text) == reference.normalize(text).strip(), text


def test_moses_port_is_identity_on_plain_ascii():
    assert normalize_punctuation("Hello, how are you?") == "Hello, how are you?"


# --- id-level parity (needs the tokenizer files) --------------------------

pytestmark_network = pytest.mark.network


@pytest.mark.network
def test_marian_ids_match_transformers():
    transformers = pytest.importorskip("transformers")
    from linguonnx.translate.models import TranslationModel

    model = TranslationModel("opus-mt-en-pt-int8")
    reference = transformers.AutoTokenizer.from_pretrained(
        "TigreGotico/opus-mt-en-pt-onnx", subfolder="int8")
    for text in CASES:
        assert model.tokenizer.encode(text) == reference(text)["input_ids"], text


@pytest.mark.network
def test_m2m100_ids_match_transformers():
    transformers = pytest.importorskip("transformers")
    from linguonnx.translate.models import TranslationModel

    model = TranslationModel("m2m100-418M-int8")
    reference = transformers.AutoTokenizer.from_pretrained(
        "TigreGotico/m2m100-418M-onnx", subfolder="int8")
    for source in ("en", "pt"):
        reference.src_lang = source
        for text in CASES:
            assert model.tokenizer.encode(text, source) == \
                reference(text)["input_ids"], (source, text)
    for code in ("pt", "gl", "ca", "ar", "zh"):
        assert model.tokenizer.lang_id(code) == reference.get_lang_id(code)


@pytest.mark.network
def test_nllb_ids_match_transformers():
    transformers = pytest.importorskip("transformers")
    from linguonnx.translate.models import TranslationModel

    model = TranslationModel("nllb-600M-int8")
    reference = transformers.AutoTokenizer.from_pretrained(
        "TigreGotico/nllb-200-distilled-600M-onnx", use_fast=False)
    for source in ("eng_Latn", "por_Latn"):
        reference.src_lang = source
        for text in CASES:
            assert model.tokenizer.encode(text, source) == \
                reference(text)["input_ids"], (source, text)
    for code in ("por_Latn", "glg_Latn", "eus_Latn", "arb_Arab"):
        assert model.tokenizer.lang_id(code) == \
            reference.convert_tokens_to_ids(code)


@pytest.mark.network
def test_round_trip_decoding_is_lossless_enough():
    """encode -> decode gives the sentence back, specials stripped."""
    from linguonnx.translate.models import TranslationModel

    for model_id, kwargs in (("m2m100-418M-int8", {"src_lang": "en"}),
                             ("nllb-600M-int8", {"src_lang": "eng_Latn"})):
        model = TranslationModel(model_id)
        text = "The library opens at nine in the morning."
        ids = model.tokenizer.encode(text, kwargs["src_lang"])
        assert model.tokenizer.decode(ids) == text


@pytest.mark.network
def test_bcp47_to_native_code_conversion():
    from linguonnx.translate.models import TranslationModel

    assert TranslationModel("nllb-600M-int8").native_code("pt") == "por_Latn"
    assert TranslationModel("nllb-600M-int8").native_code("eu") == "eus_Latn"
    assert TranslationModel("m2m100-418M-int8").native_code("pt") == "pt"
    with pytest.raises(KeyError):
        TranslationModel("m2m100-418M-int8").native_code("eu")


# --- The instruction-prefixed T5 tokenizer --------------------------------

def _tiny_prefix_tokenizer(tmp_path, added=None, legacy=True):
    """A real SentencePiece model over an English/instruction corpus.

    Ids follow T5's own layout (``pad=0``, ``eos=1``, ``unk=2``), which is
    deliberately *not* MADLAD's - a class that quietly inherited MADLAD's
    would strip the wrong ids on decode.
    """
    from linguonnx.translate.tokenizers import T5TextPrefixTokenizer
    spm = pytest.importorskip("sentencepiece")
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([
        "Translate Akkadian cuneiform to English",
        "Translate English to Akkadian cuneiform",
        "Transliterate Akkadian cuneiform to simple Latin Characters",
        "the king built a great temple",
        "hello world",
        # the instruction separator has to be in the alphabet, or a 64-piece
        # toy vocabulary shreds it to <unk> and the test measures the fixture
        "Translate Akkadian cuneiform to English: hello world",
    ] * 60), encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(tmp_path / "tiny"),
        vocab_size=64, hard_vocab_limit=False,
        pad_id=0, eos_id=1, unk_id=2, bos_id=-1)
    added_path = tmp_path / "added_tokens.json"
    added_path.write_text(json.dumps(added or {}), encoding="utf-8")
    return T5TextPrefixTokenizer(tmp_path / "tiny.model", added_path,
                                 legacy=legacy)


def test_prefix_tokenizer_takes_its_unk_from_the_sentencepiece_model(tmp_path):
    """`config.json` carries no unk id, so the SPM is the only authority."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path)
    assert tokenizer.unk_id == 2
    assert (tokenizer.eos_id, tokenizer.pad_id) == (1, 0)


def test_the_instruction_is_joined_to_the_text_the_way_the_card_joins_it(tmp_path):
    """The card's snippet is ``prompt + input_text`` with the prompt ending
    in ": ", so that separator is part of the surface the model was trained
    on and belongs to the tokenizer, not the caller."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path)
    prefixed = tokenizer.encode("hello world",
                                prefix="Translate Akkadian cuneiform to English")
    joined = tokenizer.encode(
        "Translate Akkadian cuneiform to English: hello world")
    assert prefixed == joined


def test_a_multi_piece_instruction_is_accepted(tmp_path):
    """MADLAD asserts its prefix is one piece and survives whole. An
    instruction is a sentence; inheriting that assertion would reject every
    valid prefix here."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path)
    ids = tokenizer.encode("hello world",
                           prefix="Translate Akkadian cuneiform to English")
    assert len(ids) > 5
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.unk_id not in ids


def test_an_instruction_the_vocabulary_cannot_represent_raises(tmp_path):
    """An instruction that reaches the encoder as <unk> is no instruction;
    the model answers anyway, in a direction of its own."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path)
    with pytest.raises(ValueError):
        tokenizer.encode("hello world", prefix="😀😀😀")


def test_an_empty_instruction_is_treated_as_no_instruction(tmp_path):
    tokenizer = _tiny_prefix_tokenizer(tmp_path)
    assert tokenizer.encode("hello world", prefix="") == \
        tokenizer.encode("hello world")


def test_decode_drops_the_specials_and_round_trips(tmp_path):
    tokenizer = _tiny_prefix_tokenizer(tmp_path)
    ids = tokenizer.encode("hello world")
    assert tokenizer.decode(ids) == "hello world"


# --- added tokens: the vocabulary the SentencePiece model does not hold ----

#: Two cuneiform signs and a transliteration bracket pair, numbered past the
#: toy vocabulary the way a real export numbers them past 32000.
_ADDED = {"\U00012157": 900, "\U00012000": 901, "⌈": 902, "⌉": 903}


def test_added_tokens_encode_to_their_own_ids_not_unk(tmp_path):
    """The signs are not in the SentencePiece model. Encoding without the
    added-token map does not raise - it sends the encoder a sentence of
    <unk> and the model answers it fluently."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path, added=_ADDED)
    ids = tokenizer.encode("\U00012157 \U00012000")
    assert 900 in ids and 901 in ids
    assert tokenizer.unk_id not in ids


def test_without_the_added_map_the_same_input_is_destroyed(tmp_path):
    """The failure this guards, stated as a test: same input, no map."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path, added={})
    ids = tokenizer.encode("\U00012157 \U00012000")
    assert tokenizer.unk_id in ids


def test_an_added_token_id_survives_decoding(tmp_path):
    """An id past the SentencePiece model's range makes `IdToPiece` raise
    `IndexError`, which is the loud half of the same defect."""
    tokenizer = _tiny_prefix_tokenizer(tmp_path, added=_ADDED)
    assert tokenizer.decode(tokenizer.encode("\U00012157 \U00012000")) \
        == "\U00012157 \U00012000"


def test_added_tokens_are_matched_longest_first(tmp_path):
    """A literal that begins with another literal must match whole."""
    added = dict(_ADDED, **{"⌈⌉": 904})
    tokenizer = _tiny_prefix_tokenizer(tmp_path, added=added)
    assert 904 in tokenizer.encode("⌈⌉")


def test_legacy_false_drops_the_sentencepiece_dummy_prefix(tmp_path):
    """`legacy=False` changes the first piece of every span. It never raises;
    it just feeds the model a slightly different input on every call."""
    legacy = _tiny_prefix_tokenizer(tmp_path / "a", legacy=True)
    modern = _tiny_prefix_tokenizer(tmp_path / "b", legacy=False)
    assert legacy.encode("hello world") != modern.encode("hello world")
    assert legacy.sp.IdToPiece(legacy.encode("hello world")[0]).startswith("\u2581")
    assert not modern.sp.IdToPiece(modern.encode("hello world")[0]).startswith("\u2581")


# --- the vendored Unigram reader (no `tokenizers` package) -----------------

def _unigram_json(tmp_path, normalizer=None, byte_fallback=False,
                  model_type="Unigram"):
    """A tiny Unigram tokenizer.json in the shape the real exports use.

    Scores are log probabilities, so a *less negative* piece wins. `▁king`
    is deliberately cheaper than `▁kin` + `g` so the lattice has something to
    get wrong.
    """
    vocab = [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0],
             ["▁", -6.0], ["▁king", -1.0], ["▁kin", -4.0],
             ["g", -4.0], ["▁the", -1.0], ["a", -5.0], ["-", -5.0],
             ["▁a", -2.0], [":", -3.0]]
    spec = {
        "version": "1.0",
        "normalizer": normalizer,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": "▁",
             "prepend_scheme": "always", "split": True}]},
        "decoder": {"type": "Metaspace", "replacement": "▁"},
        "model": {"type": model_type, "unk_id": 2, "byte_fallback": byte_fallback,
                  "vocab": vocab},
        # The space is an added token in the real export, and so are the signs.
        "added_tokens": [{"id": 12, "content": " "},
                         {"id": 13, "content": "\U00012157"},
                         {"id": 14, "content": "\U00012000"}],
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _unigram(tmp_path, **kwargs):
    from linguonnx.translate.tokenizers import UnigramTextPrefixTokenizer
    return UnigramTextPrefixTokenizer(_unigram_json(tmp_path, **kwargs))


def test_the_lattice_takes_the_best_scoring_path(tmp_path):
    """`▁king` scores better than `▁kin` + `g`, and a greedy longest-match
    or a left-to-right walk would not necessarily find it."""
    tokenizer = _unigram(tmp_path)
    assert tokenizer.encode("king") == [4, 1]


def test_added_tokens_are_matched_before_the_lattice(tmp_path):
    """The signs are not pieces. Segmenting them would emit <unk>."""
    tokenizer = _unigram(tmp_path)
    ids = tokenizer.encode("\U00012157 \U00012000")
    assert ids == [13, 12, 14, 1]
    assert tokenizer.unk_id not in ids


def test_the_space_survives_but_other_whitespace_does_not(tmp_path):
    """The pre-tokenizer splits on whitespace and drops it. That the plain
    space survives is only because it is an added token."""
    tokenizer = _unigram(tmp_path)
    assert 12 in tokenizer.encode("the king")
    assert 12 not in tokenizer.encode("the\tking")
    assert tokenizer.unk_id not in tokenizer.encode("the\tking")


def test_a_character_no_piece_covers_becomes_unk_rather_than_failing(tmp_path):
    tokenizer = _unigram(tmp_path)
    assert tokenizer.unk_id in tokenizer.encode("¡")


def test_decode_puts_the_words_back(tmp_path):
    """The separator is carried twice - as the added space token and as the
    next word's mark - so an uncollapsed decode returns "the  king"."""
    tokenizer = _unigram(tmp_path)
    assert tokenizer.decode(tokenizer.encode("the king")) == "the king"


def test_decoded_text_re_encodes_to_the_same_ids(tmp_path):
    """The reason the space run is collapsed: without it every round trip
    adds one space token and the ids drift."""
    tokenizer = _unigram(tmp_path)
    ids = tokenizer.encode("the king")
    assert tokenizer.encode(tokenizer.decode(ids)) == ids


def test_a_declared_normaliser_is_refused_rather_than_ignored(tmp_path):
    """A normaliser rewrites the text before the lattice sees it. Ignoring
    one produces a plausible tokenisation of a different string."""
    with pytest.raises(ValueError) as err:
        _unigram(tmp_path, normalizer={"type": "Precompiled",
                                       "precompiled_charsmap": "..."})
    assert "normalis" in str(err.value).lower()


def test_byte_fallback_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _unigram(tmp_path, byte_fallback=True)


def test_a_non_unigram_tokenizer_json_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _unigram(tmp_path, model_type="BPE")


def test_an_instruction_the_vocabulary_cannot_spell_is_refused(tmp_path):
    tokenizer = _unigram(tmp_path)
    with pytest.raises(ValueError):
        tokenizer.encode("the king", prefix="¡¡¡")
