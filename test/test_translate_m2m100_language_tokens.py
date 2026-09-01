"""Does an M2M100 model get told the right language - on both ends?

M2M100 names a language twice per call: a source tag prepended to the encoder
input, and ``forced_bos_token_id`` for the target. Both are ids out of the
export's own ``added_tokens.json``, whose keys are spelled ``__fr__``.

The defect guarded here is that those ids were derived from the registry's
``languages`` list *by position* whenever the list did not match the file
exactly. Two things made that silent and large:

* The registry's list is normalised BCP-47 and the export's is not.
  ``normalize_tag('tl')`` is ``'fil'``, so ``m2m100-418M``'s 100-entry list
  carried ``fil`` where the export has ``tl``. 99 of 100 matched, one did not,
  so the count check failed and the positional rebuild ran. The substitution
  sits at index 24, so every entry from there to index 87 slid onto the *next*
  language's token: 64 of 100 misrouted. French returned Frisian, Portuguese
  returned Romanian, Russian returned Sindhi. Nothing raised.
* ``m2m100-418M-smugri`` declares 8 languages against a 104-token block, and
  spells Northern Sami ``se`` where the export says ``sme``. All 8 went
  through the same rebuild: Finnish landed on ``__ar__`` and came back in
  Arabic script.

``test_translate_native_codes.py`` covers the neighbouring defect - a
bilingual entry with no ``languages`` list at all - and its fix made the
Masakhane fine-tunes work on their French side. It did not make them work on
the other side: ``native_code('bam')`` still answers ``'bam'``, which M2M100
has no token for. Masakhane reused Swahili's slot for those languages, and
only the registry can say so.

Every assertion here is about *identity* - which id, for which language,
against the export's own file. "Translation returns a fluent sentence" was
true throughout the defect.
"""

import json
from pathlib import Path

import pytest

import linguonnx
from linguonnx.model_manager import list_models
from linguonnx.translate.models import TranslationModel
from linguonnx.translate.tokenizers import SpmSeq2SeqTokenizer

REGISTRY = list_models("translate")


# --------------------------------------------------------------------------
# The published M2M100 language inventory
# --------------------------------------------------------------------------

#: The 100 codes M2M100 ships, in the id order of its own
#: ``added_tokens.json`` (``__af__`` = 128004 ... ``__zu__`` = 128103). A
#: fixed, published property of the checkpoint family - identical in
#: ``facebook/m2m100_418M``, ``facebook/m2m100_1.2B`` and every Masakhane
#: fine-tune of them - so it can be asserted against with no network.
#:
#: Note ``tl``, not ``fil``. Note that none of Bambara, Ghomala, Ewe, Fon or
#: Mossi is here. Those two facts are the whole defect.
M2M100_CODES = (
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca",
    "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi",
    "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu",
    "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn",
    "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr",
    "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt",
    "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv",
    "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh",
    "yi", "yo", "zh", "zu",
)

#: The four Finno-Ugric tokens TartuNLP appended for ``m2m100-418M-smugri``.
SMUGRI_EXTRA_CODES = ("liv", "sma", "sme", "vro")

#: ``aina-translator-ca-zh`` / ``-zh-ca`` are registered ``arch: "m2m100"``
#: while their own notes say Marian. That mismatch is a separate defect with
#: its own fix; pinning it here would make this file red for a reason it is
#: not about.
_DISPUTED_ARCH = {"aina-translator-ca-zh", "aina-translator-zh-ca"}


def _m2m100_entries():
    return [(model_id, entry) for model_id, entry in sorted(REGISTRY.items())
            if entry.get("arch") == "m2m100"
            and model_id.replace("-int8", "") not in _DISPUTED_ARCH]


def test_the_registry_still_has_m2m100_entries():
    """Guards the parametrised tests from passing because they found nothing."""
    assert len(_m2m100_entries()) > 20


# --------------------------------------------------------------------------
# Offline, registry-wide: every advertised language can actually be named
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,entry", _m2m100_entries(),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_every_advertised_language_has_a_real_m2m100_token(model_id, entry):
    """The blast-radius guard.

    ``native_code`` is what the source tag and the forced BOS are both built
    from, so whatever it answers has to be a token M2M100's vocabulary really
    contains. Pre-fix this fails for ``m2m100-418M`` and ``m2m100-1.2B``
    (which answer ``fil``, a code M2M100 does not have), for
    ``m2m100-418M-smugri`` (``se``, where the export says ``sme``), and for
    the nine Masakhane fine-tunes (``bam``/``bbj``/``ewe``/``fon``/``mos``).
    """
    model = TranslationModel(model_id, entry=entry)
    advertised = (set(entry.get("languages") or ())
                  | set(entry.get("src_languages") or ())
                  | set(entry.get("tgt_languages") or ())
                  | set(model.capability.pair or ()))
    assert advertised, f"{model_id} advertises no languages at all"

    allowed = set(M2M100_CODES)
    if "smugri" in model_id:
        allowed |= set(SMUGRI_EXTRA_CODES)

    wrong = {}
    for tag in sorted(advertised):
        native = model.native_code(tag)
        if native not in allowed:
            wrong[tag] = native
    assert not wrong, (
        f"{model_id} resolves {wrong!r} to codes M2M100's vocabulary does not "
        f"carry. Either the entry's `native_codes` must map them onto a token "
        f"the export really has, or the coverage claim must be dropped.")


def test_every_native_codes_key_is_a_key_native_code_will_look_up():
    """A `native_codes` key that is not the normalised tag is dead weight.

    ``native_code`` normalises its argument before the lookup
    (``normalize_tag('bam')`` is ``'bm'``, ``normalize_tag('ewe')`` is
    ``'ee'``), but ``_to_native`` is populated with whatever key the entry
    wrote. So an override keyed on the raw ISO 639-3 code is never consulted:
    the entry looks fixed, the registry test that reads the entry passes, and
    the model still resolves the language to a token it does not have.

    This is not hypothetical. linguonnx#58 added the Masakhane ``__sw__``
    overrides keyed on ``bam`` and ``ewe``; ``bbj``/``fon``/``mos`` normalise
    to themselves and worked, so 6 of 9 entries were fixed and
    ``bam_fr``/``fr_bam``/``fr_ewe`` silently stayed broken behind a green
    suite. The check is registry-wide because the trap is not m2m100's.
    """
    from linguonnx.translate.graph import normalize_tag

    dead = {}
    for model_id, entry in sorted(REGISTRY.items()):
        for key in (entry.get("native_codes") or {}):
            normalised = normalize_tag(key)
            if normalised != key:
                dead.setdefault(model_id, []).append((key, normalised))
    assert not dead, (
        f"these native_codes keys are never looked up, because native_code() "
        f"normalises first - key them on the normalised tag instead: {dead}")


def test_the_registry_never_loses_the_export_s_spelling_of_tagalog():
    """``fil`` is correct BCP-47 and wrong as an M2M100 token.

    Both spellings have to survive: the graph routes on ``fil``, the tokenizer
    addresses ``__tl__``. Losing the second is what shifted 64 languages.
    """
    entry = REGISTRY["m2m100-418M-int8"]
    assert "fil" in entry["languages"], "the router's spelling"
    assert entry["native_codes"]["fil"] == "tl", "the model's spelling"

    model = TranslationModel("m2m100-418M-int8", entry=entry)
    assert model.native_code("fil") == "tl"
    assert "fil" not in model.native_codes
    assert "tl" in model.native_codes


def test_the_masakhane_low_resource_side_uses_the_slot_upstream_chose():
    """Bambara has no M2M100 token; Masakhane reused Swahili's.

    ``__sw__`` is id 128088, the exact value every ``fr_<lang>`` repo bakes
    into its own ``config.json`` as ``forced_bos_token_id``. A registry that
    instead resolved Bambara to some other plausible-looking code would
    translate into that language and report nothing.

    linguonnx#58 established the mapping; this asserts it end to end, through
    ``native_codes``, which is what the earlier check of the *entry* alone
    could not see.
    """
    for model_id in ("m2m100_418M_bam_fr_rel_news_ft-int8",
                     "m2m100_418M_fr_bam_rel_news_ft-int8",
                     "m2m100_418M_fr_ewe_rel_news_ft-int8",
                     "m2m100_418M_fr_mos_rel_news_ft-int8",
                     "m2m100_418M_mos_fr_rel_news_ft-int8"):
        model = TranslationModel(model_id, entry=REGISTRY[model_id])
        assert set(model.native_codes) == {"fr", "sw"}, \
            (model_id, model.native_codes)


def test_hausa_keeps_its_own_token_rather_than_the_swahili_slot():
    """The control case: ``ha`` *is* an M2M100 code, so it must not be remapped.

    Applying the Swahili workaround family-wide would silently move Hausa off
    its own token - the same defect, introduced by the fix for it.
    """
    model_id = "m2m100_418M_en_hau_rel_news_ft-int8"
    model = TranslationModel(model_id, entry=REGISTRY[model_id])
    assert set(model.native_codes) == {"en", "ha"}, model.native_codes


# --------------------------------------------------------------------------
# Offline, unit: ids come from the export, never from list position
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_spm(tmp_path_factory):
    """A real SentencePiece model. Its content is irrelevant; ids are."""
    spm = pytest.importorskip("sentencepiece")
    path = tmp_path_factory.mktemp("spm")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join(["hello world", "bonjour le monde",
                                 "ola mundo", "hei maailma"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(path / "tiny"), vocab_size=48,
        hard_vocab_limit=False, pad_id=1, eos_id=2, unk_id=3, bos_id=0)
    return path / "tiny.model"


def _fake_m2m100_export(tmp_path, codes=M2M100_CODES, vocab_size=1000):
    """``vocab.json`` + ``added_tokens.json`` shaped like a real export."""
    vocab = {f"piece{i}": i for i in range(vocab_size)}
    added = {f"__{code}__": vocab_size + i for i, code in enumerate(codes)}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "added_tokens.json").write_text(json.dumps(added),
                                                encoding="utf-8")
    return tmp_path / "vocab.json", tmp_path / "added_tokens.json", added


def test_one_substituted_code_no_longer_shifts_every_later_language(tiny_spm,
                                                                    tmp_path):
    """The exact shape of the m2m100-418M defect, in miniature.

    One code in the middle is spelled the way the router spells it (``fil``)
    rather than the way the export does (``tl``). Lengths still match, so the
    old code fell through to the positional rebuild and returned an id for
    every one of the 100 - each one off by one from ``fil`` onwards. It now
    refuses instead.
    """
    vocab_path, added_path, _ = _fake_m2m100_export(tmp_path)
    router_spelling = ["fil" if code == "tl" else code for code in M2M100_CODES]

    with pytest.raises(ValueError) as err:
        SpmSeq2SeqTokenizer(tiny_spm, router_spelling, vocab_path=vocab_path,
                            added_tokens_path=added_path)
    message = str(err.value)
    assert "fil" in message
    assert "position" in message, "the error has to say why guessing is refused"


def test_a_code_the_export_lacks_is_refused_rather_than_guessed(tiny_spm,
                                                                tmp_path):
    """The Masakhane shape: a real ISO code that is simply not in the model."""
    vocab_path, added_path, _ = _fake_m2m100_export(tmp_path)
    with pytest.raises(ValueError) as err:
        SpmSeq2SeqTokenizer(tiny_spm, ("bam", "fr"), vocab_path=vocab_path,
                            added_tokens_path=added_path)
    assert "bam" in str(err.value)


def test_ids_come_from_the_export_not_from_list_order(tiny_spm, tmp_path):
    """Same inventory, reversed input order: every id must stay put."""
    vocab_path, added_path, added = _fake_m2m100_export(tmp_path)
    ordered = SpmSeq2SeqTokenizer(tiny_spm, M2M100_CODES, vocab_path=vocab_path,
                                  added_tokens_path=added_path)
    shuffled = SpmSeq2SeqTokenizer(tiny_spm, tuple(reversed(M2M100_CODES)),
                                   vocab_path=vocab_path,
                                   added_tokens_path=added_path)
    for code in M2M100_CODES:
        assert ordered.lang_id(code) == added[f"__{code}__"], code
        assert shuffled.lang_id(code) == added[f"__{code}__"], code


def test_a_partial_inventory_still_addresses_the_right_tokens(tiny_spm,
                                                              tmp_path):
    """``m2m100-418M-smugri``: 8 declared codes, 104 tokens in the block.

    Pre-fix this was the worst-affected model - the 8 codes were counted from
    the start of a 104-token block, so Finnish resolved to ``__ar__`` and the
    model answered in Arabic script.
    """
    codes = M2M100_CODES + SMUGRI_EXTRA_CODES
    vocab_path, added_path, added = _fake_m2m100_export(tmp_path, codes=codes)
    tokenizer = SpmSeq2SeqTokenizer(
        tiny_spm, ("en", "et", "fi", "lv", "liv", "vro", "sma", "sme"),
        vocab_path=vocab_path, added_tokens_path=added_path)
    assert tokenizer.lang_id("fi") == added["__fi__"]
    assert tokenizer.lang_id("liv") == added["__liv__"]
    assert tokenizer.lang_id("sme") == added["__sme__"]


def test_both_spellings_address_the_same_token(tiny_spm, tmp_path):
    vocab_path, added_path, added = _fake_m2m100_export(tmp_path)
    tokenizer = SpmSeq2SeqTokenizer(tiny_spm, M2M100_CODES,
                                    vocab_path=vocab_path,
                                    added_tokens_path=added_path)
    assert tokenizer.lang_id("fr") == tokenizer.lang_id("__fr__") \
        == added["__fr__"]


def test_language_tokens_are_stripped_from_the_output(tiny_spm, tmp_path):
    """A language id outside the block decodes as SentencePiece's ``⁇``.

    With the block built from an empty list - the Masakhane case before #56 -
    the forced BOS survived into ``decode()`` and every translation came back
    with ``⁇`` at the front.
    """
    vocab_path, added_path, added = _fake_m2m100_export(tmp_path)
    tokenizer = SpmSeq2SeqTokenizer(tiny_spm, M2M100_CODES,
                                    vocab_path=vocab_path,
                                    added_tokens_path=added_path)
    assert "⁇" not in tokenizer.decode([added["__fr__"], 5, 6])


def test_bare_lang_code_only_matches_wrapped_tokens():
    # Imported here, not at module scope: this helper is part of the fix, and
    # a module-level import of it would turn every other test in this file
    # into a collection error on an unfixed tree instead of a real failure.
    from linguonnx.translate.tokenizers import bare_lang_code

    assert bare_lang_code("__fr__") == "fr"
    assert bare_lang_code("__ceb__") == "ceb"
    assert bare_lang_code("<mask>") is None
    assert bare_lang_code("piece42") is None
    assert bare_lang_code("cat_Latn") is None


# --------------------------------------------------------------------------
# A bilingual export names its own target; a multi-target one must not
# --------------------------------------------------------------------------

def test_a_multi_target_model_is_not_pinned_to_one_target():
    """``declared_forced_bos_token_id`` must stay ``None`` for these.

    Honouring a baked-in target on a 100-language model would answer every
    request in one language - the same defect from the opposite direction.
    """
    for model_id in ("m2m100-418M-int8", "m2m100-418M-smugri-int8"):
        model = TranslationModel(model_id, entry=REGISTRY[model_id])
        assert model.capability.pair is None
        assert model.declared_forced_bos_token_id is None


# --------------------------------------------------------------------------
# End to end: one source, several targets, each in the language asked for
# --------------------------------------------------------------------------

#: Afrikaans in, with targets on both sides of the misrouted range: ``de`` and
#: ``es`` sit before it and were always right, ``fr``/``it``/``pt`` sit inside
#: it. Pre-fix all five outputs still differ and all five are fluent - only
#: the *language* of three of them is wrong, which is why a difference-only
#: assertion is not enough and GlotLID has to be consulted.
SENTENCE_AF = "Die weer was hierdie week ongewoon warm."
TARGETS = ("de", "es", "fr", "it", "pt")


@pytest.fixture(scope="module")
def m2m100_outputs():
    from linguonnx.translate.decode import GenerationConfig

    model = TranslationModel("m2m100-418M-int8")
    config = GenerationConfig(max_new_tokens=64, num_beams=1)
    return {tgt: model.translate(SENTENCE_AF, "af", tgt, config=config)
            for tgt in TARGETS}


@pytest.mark.network
class TestM2M100TranslatesToTheRequestedLanguage:

    def test_every_target_gets_a_different_sentence(self, m2m100_outputs):
        assert len(set(m2m100_outputs.values())) == len(TARGETS), m2m100_outputs

    def test_the_output_is_in_the_language_that_was_asked_for(self,
                                                              m2m100_outputs):
        """The assertion the defect failed.

        Pre-fix ``fr`` comes back Frisian ("De wetter is yn de oanhâlding
        warm."), ``it`` Japanese and ``pt`` Romanian, each being the next
        language along in the token block.
        """
        detector = linguonnx.load_detector()
        wrong = {}
        for tgt, text in m2m100_outputs.items():
            found = detector.detect(text, collapse_varieties=True).split("-")[0]
            if found != tgt:
                wrong[tgt] = (found, text)
        assert not wrong, wrong

    def test_no_output_carries_the_unknown_token_marker(self, m2m100_outputs):
        for tgt, text in m2m100_outputs.items():
            assert "⁇" not in text, (tgt, text)

    def test_no_output_loops(self, m2m100_outputs):
        for tgt, text in m2m100_outputs.items():
            words = text.split()
            assert len(set(words)) > len(words) / 2, (tgt, text)


@pytest.mark.network
class TestSmugriTranslatesToTheRequestedLanguage:
    """TartuNLP's Finno-Ugric fine-tune: 7 of 8 codes misrouted pre-fix."""

    SENTENCE_EN = "The weather has been unusually warm this week."
    SMUGRI_TARGETS = ("et", "fi", "lv")

    @pytest.fixture(scope="class")
    def outputs(self):
        from linguonnx.translate.decode import GenerationConfig

        model = TranslationModel("m2m100-418M-smugri-int8")
        config = GenerationConfig(max_new_tokens=64, num_beams=1)
        return {tgt: model.translate(self.SENTENCE_EN, "en", tgt, config=config)
                for tgt in self.SMUGRI_TARGETS}

    def test_every_target_gets_a_different_sentence(self, outputs):
        assert len(set(outputs.values())) == len(self.SMUGRI_TARGETS), outputs

    def test_the_output_is_in_the_language_that_was_asked_for(self, outputs):
        """Pre-fix ``fi`` resolved to ``__ar__`` and returned Arabic script."""
        detector = linguonnx.load_detector()
        wrong = {}
        for tgt, text in outputs.items():
            found = detector.detect(text, collapse_varieties=True).split("-")[0]
            if found != tgt:
                wrong[tgt] = (found, text)
        assert not wrong, wrong

    def test_no_output_loops(self, outputs):
        for tgt, text in outputs.items():
            words = text.split()
            assert len(set(words)) > len(words) / 2, (tgt, text)


@pytest.mark.network
class TestTheMasakhaneFinetunesServeTheirOwnPair:
    """Every direction of the family raised ``KeyError`` through this API."""

    CASES = (
        ("m2m100_418M_bam_fr_rel_news_ft-int8", "bm", "fr",
         "Mali jamana kɔnɔ, mɔgɔw caman bɛ baara kɛ sɛnɛkɛ la."),
        ("m2m100_418M_mos_fr_rel_news_ft-int8", "mos", "fr",
         "Burkina Faso soolem pʋgẽ, neb wʋsg tʋmda koodo zĩigẽ."),
    )

    @pytest.mark.parametrize("model_id,src,tgt,text", CASES,
                             ids=[case[0] for case in CASES])
    def test_the_declared_pair_translates(self, model_id, src, tgt, text):
        from linguonnx.translate.decode import GenerationConfig

        model = TranslationModel(model_id)
        out = model.translate(
            text, src, tgt,
            config=GenerationConfig(max_new_tokens=64, num_beams=1))
        assert "⁇" not in out, out
        detector = linguonnx.load_detector()
        found = detector.detect(out, collapse_varieties=True).split("-")[0]
        assert found == tgt, (found, out)
