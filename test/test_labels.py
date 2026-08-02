import pytest

from linguonnx.detect.labels import LabelMapper, parse_label, to_bcp47


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


class TestCollapseVariety:
    """Varieties fold onto the macrolanguage a caller can act on."""

    def test_arabic_varieties_collapse(self):
        from linguonnx.detect.labels import collapse_variety
        for tag in ("ajp-Arab", "ars-Arab", "arz-Arab", "ary-Arab", "arb-Arab"):
            assert collapse_variety(tag) == "ar", tag

    def test_chinese_keeps_script_when_collapsed(self):
        from linguonnx.detect.labels import collapse_variety
        assert collapse_variety("yue-Hani") == "zh-Hani"
        assert collapse_variety("cmn-Hans") == "zh-Hans"

    def test_non_variety_unchanged(self):
        from linguonnx.detect.labels import collapse_variety
        for tag in ("pt", "gl", "eu", "ast", "sr-Cyrl"):
            assert collapse_variety(tag) == tag

    def test_empty_and_unknown(self):
        from linguonnx.detect.labels import collapse_variety
        assert collapse_variety("") == ""
        assert collapse_variety("xyz") == "xyz"


class TestBareLabelShape:
    """lid.176 emits bare ISO codes with no script subtag."""

    def test_split_label_accepts_both_shapes(self):
        from linguonnx.detect.labels import split_label
        assert split_label("__label__pt") == ("pt", None)
        assert split_label("glg") == ("glg", None)
        assert split_label("__label__glg_Latn") == ("glg", "Latn")

    def test_split_label_rejects_junk(self):
        from linguonnx.detect.labels import split_label
        for bad in ("notalabel", "", "e", "_Latn", "eng_"):
            with pytest.raises(ValueError):
                split_label(bad)

    def test_parse_label_still_requires_a_script(self):
        with pytest.raises(ValueError):
            parse_label("pt")

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("__label__pt", "pt"),   # lid.176 Portuguese
            ("pt", "pt"),
            ("gl", "gl"),            # lid.176 Galician
            ("en", "en"),
            ("eu", "eu"),
            ("ca", "ca"),
            ("ru", "ru"),
            ("zh", "zh"),
            ("ceb", "ceb"),          # lid.176 3-letter code, no 639-1
            ("nds", "nds"),
        ],
    )
    def test_bare_codes_pass_through(self, label, expected):
        assert to_bcp47(label) == expected

    def test_openlid_labels_use_the_glotlid_shape(self):
        # OpenLID/OpenLID-v2 share GlotLID's iso3_Script labels
        assert to_bcp47("glg_Latn") == "gl"
        assert to_bcp47("por_Latn") == "pt"
        assert to_bcp47("eus_Latn") == "eu"

    def test_mixed_shapes_in_one_mapper(self):
        mapper = LabelMapper(["__label__pt", "__label__gl", "__label__zh"])
        assert mapper.to_bcp47("pt") == "pt"
        assert mapper.to_glotlid("gl") == "gl"
        assert mapper.available_languages == {"pt", "gl", "zh"}


class TestStandardizeTagBehaviour:
    """Regression guards on what langcodes.standardize_tag gives us."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("srp_Latn", "sr-Latn"),   # second script kept, not just Cyrl
            ("nno_Latn", "nn"),        # canonical 639-1 for Nynorsk
            ("nob_Latn", "nb"),        # ... and Bokmal, kept distinct
            ("tgl_Latn", "fil"),       # canonical replacement tgl -> fil
            ("iw", "he"),              # deprecated code repaired
            ("in", "id"),
            ("PT", "pt"),              # case normalised
        ],
    )
    def test_standardization(self, label, expected):
        assert to_bcp47(label) == expected

    def test_uncollapsed_path_stays_faithful(self):
        # standardize_tag does not fold varieties; only collapse_variety does
        from linguonnx.detect.labels import collapse_variety
        assert to_bcp47("ars_Arab") == "ars"
        assert to_bcp47("yue_Hani") == "yue-Hani"
        assert collapse_variety(to_bcp47("ars_Arab")) == "ar"
        assert collapse_variety(to_bcp47("cmn_Hant")) == "zh-Hant"
        assert collapse_variety(to_bcp47("nno_Latn")) == "no"

    def test_invalid_label_warns_but_does_not_raise(self, caplog):
        import logging
        from linguonnx.detect import labels as labels_mod
        labels_mod._WARNED_UNKNOWN.clear()
        with caplog.at_level(logging.WARNING, logger=labels_mod.__name__):
            assert to_bcp47("xyz_Latn") == "xyz"
        assert any("not a valid language tag" in r.message for r in caplog.records)
