import pytest

from lingonnx.detect.labels import LabelMapper, parse_label, to_bcp47


def test_parse_label_strips_fasttext_prefix():
    assert parse_label("__label__eng_Latn") == ("eng", "Latn")
    assert parse_label("eng_Latn") == ("eng", "Latn")


def test_parse_label_rejects_malformed_input():
    with pytest.raises(ValueError):
        parse_label("notalabel")
    with pytest.raises(ValueError):
        parse_label("_Latn")
    with pytest.raises(ValueError):
        parse_label("eng_")


@pytest.mark.parametrize(
    "label,expected",
    [
        ("eng_Latn", "en"),        # has 639-1, default script -> dropped
        ("por_Latn", "pt"),
        ("glg_Latn", "gl"),
        ("eus_Latn", "eu"),
        ("cat_Latn", "ca"),
        ("spa_Latn", "es"),
        ("rus_Cyrl", "ru"),        # default script for ru is Cyrl -> dropped
        ("arb_Arab", "ar"),        # macrolanguage override, default script -> dropped
        ("cmn_Hans", "zh-Hans"),   # macrolanguage override, always-keep-script lang
        ("zho_Hans", "zh-Hans"),
        ("ast_Latn", "ast"),       # no 639-1 code, keeps iso3, drops default script
        ("srp_Cyrl", "sr-Cyrl"),   # always-keep-script lang even though Cyrl is default
    ],
)
def test_to_bcp47(label, expected):
    assert to_bcp47(label) == expected


def test_to_bcp47_accepts_fasttext_prefix():
    assert to_bcp47("__label__eng_Latn") == "en"


def test_to_bcp47_unknown_language_falls_back_to_iso3():
    # xyz has no 639-1 mapping and langcodes doesn't recognize it either
    assert to_bcp47("xyz_Latn") == "xyz"


LABELS = [
    "__label__eng_Latn",
    "__label__por_Latn",
    "__label__glg_Latn",
    "__label__eus_Latn",
    "__label__cat_Latn",
    "__label__spa_Latn",
    "__label__rus_Cyrl",
    "__label__arb_Arab",
    "__label__cmn_Hans",
    "__label__ast_Latn",
    "__label__srp_Cyrl",
]


def test_label_mapper_forward_and_reverse_round_trip():
    mapper = LabelMapper(LABELS)
    assert mapper.to_bcp47("eng_Latn") == "en"
    assert mapper.to_glotlid("en") == "eng_Latn"
    assert mapper.to_glotlid("zh-Hans") == "cmn_Hans"


def test_label_mapper_accepts_prefixed_label():
    mapper = LabelMapper(LABELS)
    assert mapper.to_bcp47("__label__glg_Latn") == "gl"


def test_label_mapper_unknown_label_raises_keyerror():
    mapper = LabelMapper(LABELS)
    with pytest.raises(KeyError):
        mapper.to_bcp47("fra_Latn")
    with pytest.raises(KeyError):
        mapper.to_glotlid("fr")


def test_label_mapper_available_languages():
    mapper = LabelMapper(LABELS)
    assert mapper.available_languages == {
        "en", "pt", "gl", "eu", "ca", "es", "ru", "ar", "zh-Hans", "ast", "sr-Cyrl",
    }


def test_label_mapper_first_label_wins_on_collision():
    # two labels collapsing onto the same bcp47 tag: first in list order wins
    # the reverse mapping (documented, deterministic behavior).
    mapper = LabelMapper(["__label__eng_Latn", "__label__eng_Latn"])
    assert mapper.to_glotlid("en") == "eng_Latn"
