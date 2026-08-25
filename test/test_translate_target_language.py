"""Does the model actually get told where to translate *to*?

This file exists because of a defect that no other test in the suite could
see. MADLAD-400 selects its target language with a ``<2xx>`` piece on the
encoder input; the registry entry carried no ``target_token_template``, so no
piece was ever prefixed, and the model was asked to translate with no target
at all. It did not fail. `pt->eu`, `pt->fr` and `pt->ca` all returned the same
fluent English sentence, byte for byte.

Every test here therefore asserts a *difference between targets*, never that
some output exists. "Returns a non-empty string" is exactly the assertion the
defect passed.

The cheap checks are offline: the target token has to be built for the
architecture, it has to reach the tokenizer, and it has to survive
tokenisation as its own piece. Only the end-to-end differential needs weights.
"""

import pytest

from linguonnx.translate.models import TranslationModel
from linguonnx.translate.preprocess import pipeline_for
from linguonnx.translate.tokenizers import T5SpmTokenizer

SENTENCE = "Ola, o meu nome e Joao e vivo em Lisboa, Portugal."


# --------------------------------------------------------------------------
# Offline: the token is built, and it reaches the encoder input
# --------------------------------------------------------------------------

class _RecordingTokenizer:
    """Stands in for the SentencePiece tokenizer and records what it was given."""

    def __init__(self):
        self.prefixes = []

    def encode(self, text, prefix=None):
        self.prefixes.append(prefix)
        return [1, 2, 3]

    def decode(self, ids):
        return "out"


class _StubDecoder:
    def generate(self, input_ids, forced_bos_token_id=None, config=None):
        return [1, 2, 3]


def _madlad_model(entry_extra=None):
    """A MADLAD model with the real registry entry, stubbed graphs."""
    model = TranslationModel.__new__(TranslationModel)
    model.model_id = "madlad400-3b-mt-int8"
    model.arch = "madlad"
    model.entry = {"arch": "madlad", "model_id": model.model_id,
                   "languages": ["eu", "fr", "ca", "pt"],
                   **(entry_extra or {})}
    from linguonnx.translate.models import capability_from_entry
    model.entry.update({"license": "Apache-2.0", "license_tier": "permissive",
                        "size_mb": 1})
    model.capability = capability_from_entry(model.entry)
    model._to_native = {c: c for c in ("eu", "fr", "ca", "pt")}
    model._tokenizer = _RecordingTokenizer()
    model._decoder = _StubDecoder()
    model._files = model._config = None
    model._banned_token_ids = frozenset()
    return model


class TestMadladBuildsATargetToken:

    def test_the_architecture_supplies_the_template_the_registry_omits(self):
        """The `<2xx>` spelling is a MADLAD fact, not a per-export fact.

        Reading it only from the registry is what broke: an entry without the
        key was silently read as "this model needs no target token".
        """
        assert pipeline_for("madlad").default_target_token_template == "<2{code}>"

    def test_each_target_reaches_the_tokenizer_as_its_own_token(self):
        model = _madlad_model()
        for tgt in ("eu", "fr", "ca"):
            model.translate(SENTENCE, "pt", tgt)
        assert model.tokenizer.prefixes == ["<2eu>", "<2fr>", "<2ca>"]

    def test_two_targets_never_produce_the_same_prefix(self):
        model = _madlad_model()
        model.translate(SENTENCE, "pt", "eu")
        model.translate(SENTENCE, "pt", "fr")
        assert model.tokenizer.prefixes[0] != model.tokenizer.prefixes[1]

    def test_an_explicit_registry_template_still_wins(self):
        model = _madlad_model({"target_token_template": ">>{code}<<"})
        model.translate(SENTENCE, "pt", "fr")
        assert model.tokenizer.prefixes == [">>fr<<"]

    def test_encoding_without_a_target_token_raises_instead_of_translating(self):
        """There is no safe default. English is not a safe default."""
        with pytest.raises(ValueError) as err:
            pipeline_for("madlad").encode(_madlad_model(), SENTENCE, "pt", "eu",
                                          target_token=None)
        assert "<2xx>" in str(err.value)


class TestTheTokenSurvivesTokenisation:
    """A `<2xx>` that is not a vocabulary piece is shredded, not rejected."""

    def test_a_prefix_outside_the_vocabulary_raises(self, tmp_path):
        tokenizer = _tiny_t5_tokenizer(tmp_path)
        with pytest.raises(ValueError) as err:
            tokenizer.encode("hello", prefix="<2zz>")
        assert "vocabulary" in str(err.value)

    def test_a_prefix_in_the_vocabulary_is_present_in_the_ids(self, tmp_path):
        tokenizer = _tiny_t5_tokenizer(tmp_path)
        ids = tokenizer.encode("hello", prefix="<2fr>")
        assert tokenizer.sp.PieceToId("<2fr>") in ids


def _tiny_t5_tokenizer(tmp_path):
    """A real SentencePiece model with two `<2xx>` pieces and nothing else."""
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["hello world", "bonjour le monde",
                                 "ola mundo", "hola mundo",
                                 "kaixo mundua", "buongiorno mondo"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(tmp_path / "tiny"),
        vocab_size=32, hard_vocab_limit=False,
        user_defined_symbols="<2fr>,<2eu>",
        pad_id=1, eos_id=2, unk_id=0, bos_id=-1)
    return T5SpmTokenizer(tmp_path / "tiny.model")


# --------------------------------------------------------------------------
# The registry as a whole: every multi-target entry can name its target
# --------------------------------------------------------------------------

def _multi_target_entries():
    import json
    from pathlib import Path

    import linguonnx
    path = Path(linguonnx.__file__).parent / "model_index" / "translate.json"
    with open(path, encoding="utf-8") as handle:
        registry = json.load(handle)
    entries = []
    for model_id, entry in sorted(registry.items()):
        if entry.get("pair"):
            continue
        # `opus-mt-tc-big-cat_oci_spa-en`: many sources, one fixed target, no
        # `>>xxx<<` token in its vocabulary at all - there is genuinely
        # nothing to disambiguate, so it is not a "multi-target" entry in the
        # sense this guard cares about. Same exemption as
        # `entry_runnability` in graph.py.
        targets = entry.get("tgt_languages") if entry.get("tgt_languages") is not None \
            else entry.get("languages")
        if isinstance(targets, list) and len(targets) == 1:
            continue
        entries.append((model_id, entry))
    return entries


#: Architectures that put the target language on the *encoder input*. For
#: these there is no forced decoder token to fall back on, so a missing
#: mechanism is silent: the model translates into whatever it likes.
_INPUT_SIGNALLED = {"madlad", "marian"}


@pytest.mark.parametrize("model_id,entry", _multi_target_entries(),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_every_multi_target_model_has_a_way_to_name_its_target(model_id, entry):
    """The blast-radius guard.

    MADLAD shipped with no target mechanism at all and routed 269 languages
    that no other model in the registry covers. Nothing failed; the answers
    were just in the wrong language. This asserts the mechanism exists for
    every multi-target entry, per architecture, from the registry alone.
    """
    arch = entry["arch"]
    if arch in _INPUT_SIGNALLED:
        template = (entry.get("target_token_template")
                    or pipeline_for(arch).default_target_token_template)
        assert entry.get("target_token") or template, (
            f"{model_id} is a multi-target {arch} model and selects its "
            f"target with a token on the input, but neither the entry nor "
            f"the pipeline supplies one")
    elif arch == "t5-prefix":
        # The instruction is the whole mechanism, and it is per direction
        # rather than per target: the card spells "Akkadian simple
        # transliteration to English" and "English to simple Akkadian
        # transliteration" differently, so one template cannot serve both.
        templates = entry.get("prefix_templates")
        assert templates, (
            f"{model_id} is a multi-target {arch} model and selects its task "
            f"with an instruction on the input, but the entry registers none")
        assert all(instruction.strip() for instruction in templates.values()), \
            f"{model_id} has an empty instruction, which is no instruction"
        # Every routable edge must have one. `Capability.directions` is built
        # from these keys, so a language named in `languages` with no
        # instruction reaching it is a node the router can enter and the
        # pipeline cannot serve.
        reachable = {code for key in templates for code in key.split(">", 1)}
        assert reachable == set(entry["languages"]), (
            f"{model_id} lists {sorted(entry['languages'])} but its "
            f"instructions only reach {sorted(reachable)}")
    elif arch in ("m2m100", "nllb"):
        assert pipeline_for(arch).forced_bos.__qualname__.startswith(
            "SpmLangTokenPipeline"), \
            f"{model_id} relies on a forced decoder token and has none"
    else:
        assert arch == "indictrans2", (
            f"{model_id} has architecture {arch!r}, which this test does not "
            f"know how to check for target selection; add it deliberately "
            f"rather than letting it default to 'fine'")


# --------------------------------------------------------------------------
# End to end: different targets, different sentences, right languages
# --------------------------------------------------------------------------

#: One target per family, and none of them English: the failure mode was
#: "English for everything", so an English target would have passed it.
TARGETS = ("eu", "fr", "ca", "es")


@pytest.fixture(scope="module")
def madlad_outputs():
    """One MADLAD load, four targets, the same source sentence."""
    from linguonnx.translate.decode import GenerationConfig

    model = TranslationModel("madlad400-3b-mt-int8")
    config = GenerationConfig(max_new_tokens=64, num_beams=1)
    return {tgt: model.translate(SENTENCE, "pt", tgt, config=config)
            for tgt in TARGETS}


@pytest.mark.network
class TestMadladTranslatesToTheRequestedLanguage:
    """The defect end to end. Needs the 4.9 GB int8 export."""

    def test_every_target_gets_a_different_sentence(self, madlad_outputs):
        assert len(set(madlad_outputs.values())) == len(TARGETS), madlad_outputs

    def test_the_output_is_in_the_language_that_was_asked_for(self, madlad_outputs):
        from linguonnx import load_detector

        detector = load_detector()
        for tgt, text in madlad_outputs.items():
            found = detector.detect(text, collapse_varieties=True).split("-")[0]
            assert found == tgt, (tgt, found, text)

    def test_the_output_does_not_loop(self, madlad_outputs):
        """Degenerate repetition was the visible half of the same defect."""
        for tgt, text in madlad_outputs.items():
            words = text.split()
            assert len(set(words)) > len(words) / 2, (tgt, text)


# --------------------------------------------------------------------------
# One language -> a family: the `>>xxx<<` tokens are the target side only
# --------------------------------------------------------------------------
#
# `opus-mt-tc-big-en-zle` (English -> East Slavic) shipped with its six
# `>>xxx<<` target codes as a flat `languages` list, which the graph reads as
# "any-to-any across these six". English, the one language the encoder can
# actually read, was not in the list. So `en->ru` did not route at all, while
# `ru->uk` did - on an English-only encoder, which tokenises Cyrillic into
# `<unk>` and answers `',       .'` for every target. That is the
# same-output-for-different-targets signature, arriving through the registry
# shape rather than through a missing token.
#
# The rule is structural, not a list of three model ids: for an
# `opus-mt[-tc-big]-<source>-<family>` export the token set describes the
# target side, and the source side of the *name* is the only source there is.

#: model id -> the single source language its name declares.
DIRECTIONAL_GROUP_MODELS = {
    "opus-mt-tc-big-en-zle": "en",
    "opus-mt-tc-big-en-zle-int8": "en",
    "opus-mt-en-gmq": "en",
    "opus-mt-en-gmq-int8": "en",
    "opus-mt-tc-big-en-cat_oci_spa": "en",
    "opus-mt-tc-big-en-cat_oci_spa-int8": "en",
}


def _registry():
    import json
    from pathlib import Path

    import linguonnx
    path = Path(linguonnx.__file__).parent / "model_index" / "translate.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("model_id,source", sorted(DIRECTIONAL_GROUP_MODELS.items()))
class TestAOneSourceGroupModelIsDirectional:

    def test_the_entry_declares_the_source_its_encoder_reads(self, model_id, source):
        entry = _registry()[model_id]
        assert entry.get("src_languages") == [source], (
            f"{model_id} translates *from* {source!r} only; a flat "
            f"`languages` list makes the router offer target->target hops "
            f"its encoder cannot read")
        assert entry.get("languages") is None, (
            f"{model_id} carries both a flat `languages` list and a "
            f"direction; `languages` means any-to-any and would win")
        assert entry.get("tgt_languages"), f"{model_id} declares no targets"

    def test_it_routes_from_that_source_into_every_target(self, model_id, source):
        from linguonnx.translate.models import capability_from_entry

        entry = _registry()[model_id]
        capability = capability_from_entry(entry)
        for tgt in entry["tgt_languages"]:
            assert capability.covers(source, tgt), (
                f"{model_id} cannot route {source}->{tgt}, the direction it "
                f"was trained for")

    def test_it_never_offers_a_hop_between_two_of_its_targets(self, model_id, source):
        """The defect, stated as a route the registry must refuse."""
        from linguonnx.translate.models import capability_from_entry

        entry = _registry()[model_id]
        capability = capability_from_entry(entry)
        targets = entry["tgt_languages"]
        for src in targets:
            for tgt in targets:
                if src == tgt:
                    continue
                assert not capability.covers(src, tgt), (
                    f"{model_id} offers {src}->{tgt}, but its encoder only "
                    f"reads {source}: every non-{source} input tokenises to "
                    f"<unk> and every target gets the same answer")


def test_no_group_model_publishes_its_target_tokens_as_a_flat_language_set():
    """The blast-radius guard, over the whole registry.

    Any `opus-mt[-tc-big]-<lang>-<family>` entry that carries a `>>xxx<<`
    template and a flat `languages` list is the same defect in a new export.
    The source side of the name is read with the generator's own helper, so a
    newly synced export is held to the rule the generator applies.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "sync_registry_flat_check", root / "scripts" / "sync_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    offenders = []
    for model_id, entry in sorted(_registry().items()):
        if entry.get("target_token_template") != ">>{code}<<":
            continue
        if entry.get("languages") is None:
            continue
        source = module._marian_group_source(model_id)
        if source is not None and source not in entry["languages"]:
            offenders.append((model_id, source, entry["languages"]))
    assert not offenders, (
        "these entries publish their >>xxx<< target tokens as an any-to-any "
        "`languages` set, but their name says they read one source language "
        "only: " + repr(offenders))


def test_the_generator_reads_direction_off_the_export_name():
    """`scripts/sync_registry.py` must not regenerate the broken shape."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "sync_registry_direction", root / "scripts" / "sync_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._marian_group_source("opus-mt-tc-big-en-zle-onnx") == "en"
    assert module._marian_group_source("opus-mt-en-gmq-onnx") == "en"
    assert module._marian_group_source("opus-mt-tc-big-en-cat_oci_spa-onnx") == "en"
    # A family on both sides stays any-to-any.
    assert module._marian_group_source("opus-mt-tc-big-itc-itc-onnx") is None
    assert module._marian_group_source("opus-mt-tc-big-gmw-gmw-onnx") is None
    assert module._marian_group_source("opus-mt-itc-itc-onnx") is None


def test_a_token_that_does_not_produce_its_language_is_not_claimed():
    """Vocabulary presence is necessary evidence, never sufficient.

    `opus-mt-tc-big-en-zle` carries `>>orv<<` and `>>orv_Cyrl<<` (Old East
    Slavic). Both produce modern Russian: over five English sources the two
    tokens returned byte-identical output to each other 5/5, output
    byte-identical to `>>rus<<` 2/5, and modern Russian with no Old East
    Slavic morphology on the other three. Same shape as `>>kea<<` on
    `opus-mt-tc-big-itc-itc`, which is already excluded.
    """
    for model_id in ("opus-mt-tc-big-en-zle", "opus-mt-tc-big-en-zle-int8"):
        entry = _registry()[model_id]
        assert entry["tgt_languages"] == ["be", "ru", "rue", "uk"]
        assert "orv" not in entry["native_codes"]
        assert "orv-Cyrl" not in entry["native_codes"]


def test_the_generator_excludes_the_same_tokens():
    """The registry and the script that regenerates it have to agree."""
    import importlib.util
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "sync_registry_exclusions", root / "scripts" / "sync_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._MARIAN_GROUP_EXCLUSIONS["opus-mt-tc-big-en-zle"] == \
        ("orv", "orv_Cyrl")
