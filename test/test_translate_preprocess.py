"""The IndicTrans2 and OpenNMT-BPE preprocessing pipelines.

These two architectures do work on both sides of the model, and every way of
getting it wrong is silent: the model still produces fluent text, just in the
wrong script, or of a sentence that is not the one it was given. So the tests
here are about *identities* rather than about output looking plausible -
transliterate in and out and get the original back, BPE-merge and unmerge and
get the original back, tokenise and find the tags where the model expects them.

Nothing here mocks the processing. The pure-function tests use the real
`sacremoses`, `subword-nmt` and IndicNLP where the extras are installed and
skip where they are not; the tests marked ``network`` use the real published
models.
"""

import pytest

from linguonnx.limits import InputTooLongError
from linguonnx.translate.preprocess import (IndicTrans2Pipeline,
                                            MarianPipeline, Pipeline,
                                            pipeline_for)


# --------------------------------------------------------------------------
# The registry itself. No downloads, no optional dependencies.
# --------------------------------------------------------------------------

class TestPipelineRegistry:

    @pytest.mark.parametrize("arch", ["marian", "m2m100", "nllb", "madlad",
                                      "indictrans2", "opennmt-bpe"])
    def test_every_registered_architecture_has_a_pipeline(self, arch):
        assert isinstance(pipeline_for(arch), Pipeline)

    def test_an_unknown_architecture_raises_and_lists_what_is_known(self):
        with pytest.raises(ValueError) as err:
            pipeline_for("no-such-arch")
        assert "indictrans2" in str(err.value)

    def test_pre_and_post_processing_are_the_same_object(self):
        """The pairing is what stops one half being changed without the other."""
        assert pipeline_for("indictrans2") is pipeline_for("indictrans2")

    def test_only_indictrans2_declares_a_source_ceiling(self):
        assert IndicTrans2Pipeline.max_source_tokens == 256
        assert MarianPipeline.max_source_tokens is None


class TestSourceLengthCeiling:
    """A frozen position table is a hard limit, not a quality threshold."""

    class _FakeModel:
        model_id = "indictrans2-test"

    def test_an_input_over_the_ceiling_is_refused(self):
        pipeline = pipeline_for("indictrans2")
        with pytest.raises(InputTooLongError) as err:
            pipeline.check_length(list(range(257)), self._FakeModel())
        assert "256" in str(err.value)
        # The caller has to be told what to do about it, not just that it broke.
        assert "split" in str(err.value).lower()

    def test_an_input_at_the_ceiling_is_accepted(self):
        pipeline = pipeline_for("indictrans2")
        assert len(pipeline.check_length(list(range(256)), self._FakeModel())) == 256

    def test_an_architecture_without_a_ceiling_accepts_anything(self):
        assert len(pipeline_for("marian").check_length(
            list(range(5000)), self._FakeModel())) == 5000


# --------------------------------------------------------------------------
# IndicProcessor. Needs `linguonnx[indic]`, needs no model.
# --------------------------------------------------------------------------

def _processor():
    pytest.importorskip("regex")
    pytest.importorskip("sacremoses")
    pytest.importorskip("indicnlp")
    from linguonnx.translate._indic_processor import IndicProcessor
    return IndicProcessor()


HINDI = "जब मैं छोटा था, तब मैं हर दिन पार्क जाता था।"
TAMIL = "நான் சிறுவனாக இருந்தபோது தினமும் பூங்காவிற்குச் செல்வேன்."
BENGALI = "আমি ছোট থাকতে প্রতিদিন পার্কে যেতাম।"
MALAYALAM = "ഞാൻ ചെറുപ്പത്തിൽ എല്ലാ ദിവസവും പാർക്കിൽ പോകുമായിരുന്നു."
MARATHI = "मी लहान असताना दररोज बागेत जायचो."


class TestLanguageTagPrefix:
    """The tags are what select the pair; without them the model guesses."""

    def test_the_prefix_is_the_two_tags_in_order(self):
        text, _ = _processor().preprocess("Hello there.", "eng_Latn", "hin_Deva")
        assert text.startswith("eng_Latn hin_Deva ")

    def test_the_target_side_carries_no_tags(self):
        text, _ = _processor().preprocess(HINDI, "hin_Deva", is_target=True)
        assert not text.startswith("hin_Deva")

    @pytest.mark.parametrize("src,tgt", [("hin_Deva", "xyz_Abcd"),
                                         ("xyz_Abcd", "hin_Deva"),
                                         ("hi", "ta")])
    def test_a_tag_the_model_does_not_know_raises(self, src, tgt):
        """An unknown tag would be tokenised as ordinary text, and silently."""
        with pytest.raises(ValueError):
            _processor().preprocess("test", src, tgt)


class TestTransliterationRoundTrip:
    """Transliterate to Devanagari and back, and the script has to survive.

    This is the step that fails silently: the output of a `tam_Taml` request
    that skips the return leg is fluent Tamil written in Devanagari.
    """

    @pytest.mark.parametrize("tag,text", [
        ("tam_Taml", TAMIL), ("ben_Beng", BENGALI),
        ("mal_Mlym", MALAYALAM), ("mar_Deva", MARATHI), ("hin_Deva", HINDI),
    ])
    def test_a_sentence_survives_the_round_trip(self, tag, text):
        processor = _processor()
        forward, entity_map = processor.preprocess(text, tag, "eng_Latn")
        stripped = forward.split(" ", 2)[2]
        assert processor.postprocess(stripped, tag, entity_map) == text

    @pytest.mark.parametrize("tag,text", [("tam_Taml", TAMIL),
                                          ("ben_Beng", BENGALI),
                                          ("mal_Mlym", MALAYALAM)])
    def test_a_non_devanagari_script_really_is_transliterated_on_the_way_in(
            self, tag, text):
        forward, _ = _processor().preprocess(text, tag, "eng_Latn")
        body = forward.split(" ", 2)[2]
        assert any("ऀ" <= ch <= "ॿ" for ch in body), \
            "expected Devanagari in the model's input"
        assert body != text

    @pytest.mark.parametrize("tag,text", [
        ("urd_Arab", "میں بچپن میں روز پارک جاتا تھا۔"),
        ("eng_Latn", "I used to go to the park every day."),
    ])
    def test_perso_arabic_and_latin_are_left_in_their_own_script(self, tag, text):
        """Only some scripts are transliterated; over-applying it is a bug too."""
        forward, _ = _processor().preprocess(text, tag, "hin_Deva")
        body = forward.split(" ", 2)[2]
        assert not any("ऀ" <= ch <= "ॿ" for ch in body)


class TestPlaceholders:

    def test_an_email_is_hidden_behind_a_placeholder(self):
        text, entity_map = _processor().preprocess(
            "Write to a.b@example.com today.", "eng_Latn", "hin_Deva")
        assert "a.b@example.com" not in text
        # Moses tokenisation then splits the marker, which is why the map
        # carries every spelling the model has been seen to emit.
        assert "< ID1 >" in text
        assert entity_map["<ID1>"] == "a.b@example.com"
        assert entity_map["< ID1 >"] == "a.b@example.com"

    def test_the_placeholder_is_put_back_on_the_way_out(self):
        processor = _processor()
        text, entity_map = processor.preprocess(
            "Write to a.b@example.com today.", "eng_Latn", "hin_Deva")
        restored = processor.postprocess("Send to <ID1> .", "eng_Latn", entity_map)
        assert "a.b@example.com" in restored

    def test_the_map_belongs_to_the_call_that_made_it(self):
        """Upstream parks the map in a queue; a failed call desynchronises it."""
        processor = _processor()
        _, first = processor.preprocess("mail a@b.com", "eng_Latn", "hin_Deva")
        _, second = processor.preprocess("no entities here", "eng_Latn", "hin_Deva")
        assert first["<ID1>"] == "a@b.com"
        assert second == {}

    def test_native_digits_become_ascii(self):
        text, _ = _processor().preprocess("मेरे पास १२३ किताबें हैं।",
                                          "hin_Deva", "eng_Latn")
        assert "123" in text
        assert "१२३" not in text


class TestAgainstTheRealIndicProcessor:
    """Differential against `IndicTransToolkit`, where it happens to be installed.

    linguonnx does not depend on it - it pulls `transformers` in - but when it
    is present the vendored copy has to agree with it exactly. This is the test
    that would catch the copy drifting from upstream.
    """

    SAMPLES = [
        ("eng_Latn", "hin_Deva", "When I was young, I used to go to the park."),
        ("eng_Latn", "tam_Taml", "Send the report to a.b@example.com by 12/03/2024."),
        ("hin_Deva", "eng_Latn", HINDI),
        ("tam_Taml", "eng_Latn", TAMIL),
        ("ben_Beng", "mal_Mlym", BENGALI),
        ("mar_Deva", "ben_Beng", MARATHI),
        ("mal_Mlym", "mar_Deva", MALAYALAM),
        ("urd_Arab", "eng_Latn", "میں بچپن میں روز پارک جاتا تھا۔"),
    ]

    @pytest.mark.parametrize("src,tgt,text", SAMPLES)
    def test_preprocessing_matches_upstream(self, src, tgt, text):
        toolkit = pytest.importorskip("IndicTransToolkit")
        reference = toolkit.IndicProcessor(inference=True)
        ours, _ = _processor().preprocess(text, src, tgt)
        assert ours == reference.preprocess_batch([text], src_lang=src,
                                                  tgt_lang=tgt)[0]


# --------------------------------------------------------------------------
# OpenNMT BPE. Needs `linguonnx[opennmt]`.
# --------------------------------------------------------------------------

class TestBpeMergeRemoval:
    """`@@` removal has to follow the upstream rule, not the obvious one."""

    @staticmethod
    def _strip(text):
        import re
        return re.sub(r"@\s*", "", text)

    def test_a_word_is_rejoined(self):
        assert self._strip("un@@ import@@ ant") == "unimportant"

    def test_a_marker_before_punctuation_does_not_glue_two_words(self):
        """The naive `replace("@@ ", "")` gets this one wrong."""
        assert self._strip("xard@@ ín .") == "xardín ."
        assert self._strip("cas@@ a@@ , ho@@ xe") == "casa, hoxe"

    def test_plain_text_is_untouched(self):
        assert self._strip("Os nenos xogan no xardín.") == "Os nenos xogan no xardín."


class TestMissingExtrasAreNamed:
    """A missing extra must say which one, and must never be worked around."""

    def test_the_indic_helper_names_the_extra(self, monkeypatch):
        import importlib

        from linguonnx.translate import _indic_processor

        def refuse(name):
            raise ImportError(f"No module named {name!r}")

        monkeypatch.setattr(importlib, "import_module", refuse)
        with pytest.raises(ImportError) as err:
            _indic_processor._require("indicnlp")
        assert "linguonnx[indic]" in str(err.value)

    def test_the_opennmt_helper_names_the_extra(self, monkeypatch):
        import importlib

        from linguonnx.translate import tokenizers

        def refuse(name):
            raise ImportError(f"No module named {name!r}")

        monkeypatch.setattr(importlib, "import_module", refuse)
        with pytest.raises(ImportError) as err:
            tokenizers._require_opennmt("subword_nmt.apply_bpe")
        assert "linguonnx[opennmt]" in str(err.value)


# --------------------------------------------------------------------------
# Against the published models.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indictrans2_model():
    pytest.importorskip("indicnlp")
    from linguonnx.translate.models import TranslationModel
    return TranslationModel("indictrans2-en-indic-dist-200M-int8")


@pytest.fixture(scope="module")
def opennmt_model():
    pytest.importorskip("subword_nmt")
    from linguonnx.translate.models import TranslationModel
    return TranslationModel("nos-coda_iacobus-en-gl-int8")


@pytest.mark.network
@pytest.mark.usefixtures("indictrans2_model")
class TestIndicTrans2OnTheRealModel:

    @pytest.fixture
    def model(self, indictrans2_model):
        return indictrans2_model

    def test_the_tags_are_single_vocabulary_entries(self, model):
        """Split into pieces they would mean nothing in that position."""
        ids = model.tokenizer.encode("eng_Latn hin_Deva Hello there .")
        assert ids[0] == model.tokenizer.src_encoder["eng_Latn"]
        assert ids[1] == model.tokenizer.src_encoder["hin_Deva"]
        assert ids[-1] == model.tokenizer.eos_id

    def test_input_without_tags_raises(self, model):
        with pytest.raises(ValueError):
            model.tokenizer.encode("Hello")

    def test_the_output_is_in_the_requested_script(self, model):
        """Devanagari here would be the silent failure this pipeline prevents."""
        out = model.translate("The children are playing in the garden.",
                              "en", "ta")
        assert any("஀" <= ch <= "௿" for ch in out), out
        assert not any("ऀ" <= ch <= "ॿ" for ch in out), out

    def test_a_long_input_is_refused_rather_than_truncated(self, model):
        with pytest.raises(InputTooLongError):
            model.translate("The children are playing in the garden. " * 200,
                            "en", "hi")


@pytest.mark.network
class TestOpenNmtOnTheRealModel:

    @pytest.fixture
    def model(self, opennmt_model):
        return opennmt_model

    def test_encoder_ids_carry_the_source_offset(self, model):
        ids = model.tokenizer.encode("The children are playing in the garden.")
        assert ids
        assert min(ids) >= model.tokenizer.source_offset

    def test_no_eos_is_appended_to_the_source(self, model):
        ids = model.tokenizer.encode("The children are playing in the garden.")
        assert ids[-1] != model.tokenizer.eos_id + model.tokenizer.source_offset

    def test_it_translates_into_galician(self, model):
        assert model.translate("The children are playing in the garden.",
                               "en", "gl") == "Os nenos xogan no xardín."

    def test_unk_is_not_a_special_token(self, model):
        """Dropping `<unk>` would hide where the model failed."""
        assert model.tokenizer.unk_id not in model.tokenizer._specials
