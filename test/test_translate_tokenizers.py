"""Tokenizer parity against `transformers`, plus the Moses port.

The tokenizers are the other silent-failure surface: a wrong id offset or a
missing language token does not raise, it just degrades the translation. So the
ids are compared to the reference tokenizer directly, token for token.
"""

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
