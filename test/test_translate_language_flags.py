"""Per-language quality flags: a model advertises a language and cannot write it.

``madlad400-3b-mt`` covers Chuvash in its tokenizer and answers ``en -> cv``
in Russian. The whole-model ``quality`` block cannot express that - one chrF
number stands for ~400 languages - and flagging the whole model would cost
the ~399 it does fine. These tests pin the per-language field, its
target/source semantics, and the precision rule.

Everything here is offline and reads either the committed registry or a
hand-built dict. Nothing loads a graph, so there is no ``importorskip`` and
no test in this file can be silently skipped.
"""

import json
from pathlib import Path

import pytest

from linguonnx.translate.quality import (LANGUAGE_FLAG_SIDES,
                                         entry_target_languages,
                                         flagged_languages,
                                         flagged_target_languages,
                                         is_language_flagged,
                                         language_flag_reason,
                                         language_flag_reason_for)

REGISTRY = json.loads(
    (Path(__file__).resolve().parent.parent
     / "linguonnx" / "model_index" / "translate.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def translator():
    """A graph over MADLAD int8 alone. Nothing is downloaded - routing and
    coverage read the registry entry, never the weights."""
    from linguonnx.translate import load_translator
    return load_translator(models=["madlad400-3b-mt-int8"], max_model_mb=None)


def _entry(flags, **kw):
    """A minimal registry entry carrying ``language_flags``."""
    entry = {"model_id": "demo", "arch": "madlad", "language_flags": flags}
    entry.update(kw)
    return entry


class TestAFlagReadsBackAsAReason:

    def test_a_flagged_language_returns_its_curated_reason_verbatim(self):
        entry = _entry({"cv": {"reason": "answers in ru", "side": "target"}})
        assert language_flag_reason(entry, "cv") == "answers in ru"

    def test_the_reason_composes_into_the_message_routing_will_print(self):
        """The clause is written to be read inside a sentence, not alone.

        Phase 2's failure message is "<model> covers <lang> but is flagged:
        <reason>", so a reason that reads as a sentence of its own would
        produce nonsense there.
        """
        entry = _entry({"cv": {"reason": "answers in ru", "side": "target"}})
        message = (f"madlad400-3b-mt covers cv but is flagged: "
                   f"{language_flag_reason(entry, 'cv')}")
        assert message == \
            "madlad400-3b-mt covers cv but is flagged: answers in ru"

    def test_an_unflagged_language_of_a_flagged_model_is_none(self):
        """The whole point: a flag costs one language, not the model."""
        entry = _entry({"cv": {"reason": "answers in ru", "side": "target"}})
        assert language_flag_reason(entry, "de") is None

    def test_an_entry_with_no_flags_at_all_is_none(self):
        assert language_flag_reason({"model_id": "demo"}, "cv") is None

    def test_a_flag_with_no_reason_still_flags(self):
        """Absence of prose must not read as absence of a flag - the entry
        exists because somebody saw the language fail."""
        entry = _entry({"cv": {"evidence": "en->cv gave Russian"}})
        assert language_flag_reason(entry, "cv") == \
            "flagged, no reason recorded"

    def test_flagged_languages_lists_exactly_the_flagged_ones(self):
        entry = _entry({"cv": {"reason": "answers in ru"},
                        "gn": {"reason": "answers in es"}})
        assert flagged_languages(entry) == frozenset({"cv", "gn"})


class TestASideIsAClaimAboutOneDirectionOnly:
    """The MADLAD observation is ``en -> cv``. It says nothing about
    ``cv -> en``, and inferring one from the other would delete coverage
    nobody measured."""

    TARGET = _entry({"cv": {"reason": "answers in ru", "side": "target"}})
    SOURCE = _entry({"cv": {"reason": "reads it as ru", "side": "source"}})
    BOTH = _entry({"cv": {"reason": "unusable", "side": "both"}})

    def test_a_target_flag_does_not_answer_a_source_question(self):
        assert language_flag_reason(self.TARGET, "cv", "target") == \
            "answers in ru"
        assert language_flag_reason(self.TARGET, "cv", "source") is None

    def test_a_source_flag_does_not_answer_a_target_question(self):
        assert language_flag_reason(self.SOURCE, "cv", "source") == \
            "reads it as ru"
        assert language_flag_reason(self.SOURCE, "cv", "target") is None

    def test_both_answers_either_question(self):
        assert language_flag_reason(self.BOTH, "cv", "target") == "unusable"
        assert language_flag_reason(self.BOTH, "cv", "source") == "unusable"

    def test_an_absent_side_means_target(self):
        entry = _entry({"cv": {"reason": "answers in ru"}})
        assert language_flag_reason(entry, "cv", "target") == "answers in ru"
        assert language_flag_reason(entry, "cv", "source") is None

    def test_an_unknown_side_in_the_data_raises_rather_than_reads_as_unflagged(
            self):
        """A typo'd side must not silently unflag a language. Returning
        ``None`` here would turn a curation mistake into restored coverage
        for a language that does not work."""
        entry = _entry({"cv": {"reason": "answers in ru", "side": "targets"}})
        with pytest.raises(ValueError):
            language_flag_reason(entry, "cv", "target")

    def test_an_unknown_side_in_the_question_raises(self):
        with pytest.raises(ValueError):
            language_flag_reason(self.TARGET, "cv", "output")

    def test_the_declared_sides_are_the_three_documented_ones(self):
        assert LANGUAGE_FLAG_SIDES == ("target", "source", "both")


class TestAFlagCrossesPrecisions:
    """int8 is the same weights at a lower bit depth. Quantising a model
    that cannot write Chuvash does not teach it any."""

    REGISTRY = {
        "demo": _entry({"cv": {"reason": "answers in ru", "side": "target"}},
                       model_id="demo", precision="fp32"),
        "demo-int8": _entry({}, model_id="demo-int8", precision="int8"),
    }

    def test_a_flag_on_fp32_holds_for_its_int8_counterpart(self):
        assert language_flag_reason_for("demo-int8", "cv",
                                        registry=self.REGISTRY) == \
            "answers in ru"

    def test_a_flag_on_int8_holds_for_its_fp32_counterpart(self):
        registry = {"demo": _entry({}, model_id="demo"),
                    "demo-int8": _entry(
                        {"cv": {"reason": "answers in ru"}},
                        model_id="demo-int8")}
        assert language_flag_reason_for("demo", "cv",
                                        registry=registry) == "answers in ru"

    def test_the_counterpart_rule_does_not_invent_flags(self):
        assert language_flag_reason_for("demo-int8", "de",
                                        registry=self.REGISTRY) is None

    def test_a_missing_counterpart_is_not_an_error(self):
        assert language_flag_reason_for("demo-int8", "cv",
                                        registry={}) is None

    def test_is_language_flagged_agrees_with_the_reason(self):
        assert is_language_flagged("demo-int8", "cv",
                                   registry=self.REGISTRY) is True
        assert is_language_flagged("demo-int8", "de",
                                   registry=self.REGISTRY) is False


class TestTargetCoverageIsDerivedTheWayTheGraphDerivesIt:

    def test_a_bilingual_entry_writes_only_the_second_half_of_its_pair(self):
        assert entry_target_languages({"pair": ["en", "gl"]}) == \
            frozenset({"gl"})

    def test_a_directional_multilingual_entry_writes_its_target_set(self):
        entry = {"src_languages": ["en"], "tgt_languages": ["hi", "bn"],
                 "languages": ["en", "hi", "bn"]}
        assert entry_target_languages(entry) == frozenset({"hi", "bn"})

    def test_an_any_to_any_entry_writes_everything_it_covers(self):
        assert entry_target_languages({"languages": ["en", "cv"]}) == \
            frozenset({"en", "cv"})

    def test_native_codes_are_normalised_like_the_graph_normalises_them(self):
        """The registry stores the model's own codes; the flags and the
        graph both speak BCP-47, so the two have to meet."""
        assert entry_target_languages({"languages": ["por_Latn"]}) == \
            frozenset({"pt"})


class TestAnHonestCoverageCount:

    FLAGGED = _entry({"cv": {"reason": "answers in ru", "side": "target"}},
                     model_id="big", languages=["en", "cv", "ru"])
    OTHER = {"model_id": "small", "pair": ["en", "cv"]}

    def test_a_language_only_a_flagged_model_covers_is_counted_out(self):
        counted = flagged_target_languages({"big": self.FLAGGED})
        assert set(counted) == {"cv"}
        assert counted["cv"] == ("big: answers in ru",)

    def test_a_second_unflagged_provider_keeps_the_language(self):
        """A flag is about one model, not about the language. If anything
        else can write it, coverage is intact."""
        assert flagged_target_languages(
            {"big": self.FLAGGED, "small": self.OTHER}) == {}

    def test_a_source_only_flag_does_not_reduce_target_coverage(self):
        entry = _entry({"cv": {"reason": "reads it as ru", "side": "source"}},
                       model_id="big", languages=["en", "cv"])
        assert flagged_target_languages({"big": entry}) == {}

    def test_the_unflagged_languages_of_a_selection_are_the_rest(self):
        counted = flagged_target_languages({"big": self.FLAGGED})
        covered = entry_target_languages(self.FLAGGED)
        assert covered - frozenset(counted) == frozenset({"en", "ru"})


class TestTheCommittedRegistry:
    """The one populated flag, and the shape rules every future one obeys."""

    MADLAD = ("madlad400-3b-mt", "madlad400-3b-mt-int8")

    @pytest.mark.parametrize("model_id", MADLAD)
    def test_madlad_is_flagged_for_chuvash_in_both_precisions(self, model_id):
        assert language_flag_reason_for(model_id, "cv",
                                        registry=REGISTRY) == "answers in ru"

    @pytest.mark.parametrize("model_id", MADLAD)
    def test_madlad_still_advertises_chuvash(self, model_id):
        """The flag records that a claim is wrong; it must not quietly
        rewrite the claim. Deleting `cv` from `languages` would hide the
        problem and lose the ability to say why the language is refused."""
        assert "cv" in REGISTRY[model_id]["languages"]

    @pytest.mark.parametrize("model_id", MADLAD)
    def test_the_flag_is_not_a_whole_model_flag(self, model_id):
        """MADLAD is a good model for ~400 languages. One bad language must
        not remove it, or the size of the loss is 400x the size of the bug."""
        assert not REGISTRY[model_id].get("quality")
        assert language_flag_reason_for(model_id, "de",
                                        registry=REGISTRY) is None

    @pytest.mark.parametrize("model_id", MADLAD)
    def test_the_flag_carries_the_observation_it_rests_on(self, model_id):
        """A flag with no evidence is an opinion, and an opinion silently
        deletes a language from the coverage count."""
        flag = REGISTRY[model_id]["language_flags"]["cv"]
        assert flag["evidence"] == \
            "en→cv 'Good day, my friend.' → 'Добрый день, мой друг.'"
        assert flag["detector"] == "glotlid=ru"
        assert flag["date"] == "2026-08-10"
        assert flag["method"] == "int8-sweep"

    def test_the_two_precisions_carry_the_identical_flag(self):
        """The counterpart rule makes them equivalent at read time; they are
        both written out anyway, so a human reading either entry sees it."""
        assert REGISTRY["madlad400-3b-mt"]["language_flags"] == \
            REGISTRY["madlad400-3b-mt-int8"]["language_flags"]

    def test_no_other_entry_carries_an_invented_flag(self):
        """Every flag cites an observation someone actually made. Today
        that is exactly one."""
        flagged = {model_id for model_id, entry in REGISTRY.items()
                   if entry.get("language_flags")}
        assert flagged == set(self.MADLAD)

    def test_every_committed_flag_states_its_side_explicitly(self):
        """The reader defaults to `target`, but a curated flag must not lean
        on a default for something this consequential - the difference
        decides whether a language is refused as a source too."""
        for model_id, entry in REGISTRY.items():
            for lang, flag in (entry.get("language_flags") or {}).items():
                assert flag.get("side") in LANGUAGE_FLAG_SIDES, \
                    f"{model_id}/{lang} does not state a valid side"

    def test_every_committed_flag_names_a_language_its_model_covers(self):
        """A flag on a language the model never claimed is dead text; the
        likely cause is a typo, and it would silently protect nothing."""
        for model_id, entry in REGISTRY.items():
            covered = frozenset(entry.get("languages") or ()) \
                | frozenset(entry.get("tgt_languages") or ()) \
                | frozenset(entry.get("pair") or ())
            for lang in (entry.get("language_flags") or {}):
                assert lang in covered, f"{model_id} does not cover {lang}"

    def test_every_committed_flag_carries_reason_evidence_and_date(self):
        for model_id, entry in REGISTRY.items():
            for lang, flag in (entry.get("language_flags") or {}).items():
                for key in ("reason", "evidence", "detector", "date",
                            "method"):
                    assert flag.get(key), f"{model_id}/{lang} lacks {key}"


class TestATranslatorReportsBothNumbers:
    """`available_languages` is a count of *claims*. `unflagged_languages` is
    the count worth quoting."""

    def test_chuvash_is_routable_and_not_usable(self, translator):
        assert "cv" in translator.available_languages
        assert "cv" not in translator.unflagged_languages

    def test_the_flag_is_reported_with_its_reason_and_its_model(
            self, translator):
        assert translator.flagged_languages["cv"] == \
            ("madlad400-3b-mt-int8: answers in ru",)

    def test_only_the_flagged_language_is_deducted(self, translator):
        available = translator.available_languages
        assert available - translator.unflagged_languages == frozenset({"cv"})
        assert "ru" in translator.unflagged_languages
        assert "de" in translator.unflagged_languages

    def test_the_fp32_flag_would_reach_an_int8_only_selection(self, translator):
        """The selection holds the int8 entry only. The rule that a flag
        crosses precisions is what keeps that from being an escape hatch."""
        assert list(translator.models) == ["madlad400-3b-mt-int8"]
        assert translator.language_flag_reason(
            "madlad400-3b-mt", "cv") == "answers in ru"

    def test_a_selection_with_no_flags_loses_nothing(self):
        from linguonnx.translate import load_translator
        other = load_translator(models=["nos-mt-en-gl"], max_model_mb=None)
        assert other.flagged_languages == {}
        assert other.unflagged_languages == other.available_languages
