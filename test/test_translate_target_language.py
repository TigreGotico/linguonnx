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
    return [(model_id, entry) for model_id, entry in sorted(registry.items())
            if not entry.get("pair")]


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
