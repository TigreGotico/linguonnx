"""Per-architecture text handling, behind one interface.

Every architecture in the registry answers the same three questions, and gets
them wrong in the same silent way if the answers are mismatched:

1. what ids does this text become,
2. which token, if any, is forced as the decoder's first output,
3. what does the id sequence coming back mean.

Before this module those three lived as a chain of ``if arch == ...`` branches
inside ``TranslationModel.translate``. Two of the architectures - IndicTrans2
and OpenNMT-BPE - need real work on both ends of the model, not just a
different prefix token, and bolting that onto the chain would have made the
mismatch easy to introduce: preprocessing that transliterates with
postprocessing that does not gives you fluent output in the wrong script and
nothing raises.

So each architecture is a :class:`Pipeline`, registered by name, and
:meth:`Pipeline.encode` and :meth:`Pipeline.decode` are written next to each
other. A new architecture is a new subclass and a ``@register`` line; it never
touches ``translate()``.

Optional dependencies
---------------------
``IndicTrans2Pipeline`` needs ``linguonnx[indic]`` and ``OpenNmtBpePipeline``
needs ``linguonnx[opennmt]``. Both import lazily, on first use, and both raise
``ImportError`` naming the extra. Neither ever falls back to a simpler
tokenisation: a wrong tokenisation does not fail, it translates into the wrong
words.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Type

from linguonnx.limits import InputTooLongError

__all__ = ["Pipeline", "register", "pipeline_for"]


class Pipeline:
    """How one architecture turns text into ids and ids back into text."""

    #: Hard ceiling on encoder input length, in tokens, when the model's
    #: position table is frozen at export time. ``None`` means the
    #: architecture has no ceiling of its own and only the library-wide
    #: ``LINGUONNX_MAX_ENCODER_TOKENS`` bound applies.
    max_source_tokens: Optional[int] = None

    def encode(self, model, text: str, src: str, tgt: str,
               target_token: Optional[str] = None) -> List[int]:
        raise NotImplementedError

    def forced_bos(self, model, tgt: str) -> Optional[int]:
        """The decoder's first generated token, when the architecture forces one."""
        return None

    def decode(self, model, ids: Sequence[int], src: str, tgt: str) -> str:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------

    def check_length(self, ids: Sequence[int], model) -> List[int]:
        """Refuse an input the architecture cannot embed at all.

        This is stricter than the library-wide bound in
        :mod:`linguonnx.limits`, and it is checked here rather than in the
        decoder because the number that matters is per architecture. Refusing
        beats truncating: a truncated translation is a fluent translation of
        half the sentence, and nothing in the output says so.
        """
        limit = self.max_source_tokens
        if limit is not None and len(ids) > limit:
            raise InputTooLongError(
                f"input is {len(ids)} tokens and {model.model_id} accepts at "
                f"most {limit}; its position table is frozen at export time, "
                f"so a longer input cannot be embedded. Split the text into "
                f"sentences and translate them one at a time.")
        return list(ids)


_PIPELINES: Dict[str, Pipeline] = {}


def register(*archs: str) -> Callable[[Type[Pipeline]], Type[Pipeline]]:
    """Class decorator: bind a :class:`Pipeline` to one or more ``arch`` names."""
    def decorate(cls: Type[Pipeline]) -> Type[Pipeline]:
        instance = cls()
        for arch in archs:
            _PIPELINES[arch] = instance
        return cls
    return decorate


def pipeline_for(arch: str) -> Pipeline:
    try:
        return _PIPELINES[arch]
    except KeyError:
        raise ValueError(
            f"unknown architecture {arch!r}; linguonnx knows "
            f"{', '.join(sorted(_PIPELINES))}") from None


# --------------------------------------------------------------------------
# The SentencePiece architectures
# --------------------------------------------------------------------------

@register("m2m100", "nllb")
class SpmLangTokenPipeline(Pipeline):
    """M2M100 and NLLB. Source tag on the input, target tag forced on output."""

    def encode(self, model, text, src, tgt, target_token=None):
        return self.check_length(
            model.tokenizer.encode(text, model.native_code(src)), model)

    def forced_bos(self, model, tgt):
        return model.tokenizer.lang_id(model.native_code(tgt))

    def decode(self, model, ids, src, tgt):
        return model.tokenizer.decode(ids)


@register("marian")
class MarianPipeline(Pipeline):
    """opus-mt. The model is the pair; only group models take a target token."""

    def encode(self, model, text, src, tgt, target_token=None):
        return self.check_length(
            model.tokenizer.encode(text, target_token=target_token), model)

    def decode(self, model, ids, src, tgt):
        return model.tokenizer.decode(ids)


@register("madlad")
class MadladPipeline(Pipeline):
    """MADLAD (T5). The target language is a ``<2xx>`` piece on the *input*."""

    def encode(self, model, text, src, tgt, target_token=None):
        return self.check_length(
            model.tokenizer.encode(text, prefix=target_token), model)

    def decode(self, model, ids, src, tgt):
        return model.tokenizer.decode(ids)


# --------------------------------------------------------------------------
# IndicTrans2
# --------------------------------------------------------------------------

@register("indictrans2")
class IndicTrans2Pipeline(Pipeline):
    """IndicTrans2, with AI4Bharat's processing on both ends.

    In: punctuation normalisation, native digits to ASCII, URLs and numerals
    behind ``<IDn>`` placeholders, Moses or Indic tokenisation, and - for every
    Indic script except Perso-Arabic, Ol Chiki, Meetei Mayek and Latin -
    transliteration into Devanagari, which is the script the shared vocabulary
    is written in. Then the ``<src_tag> <tgt_tag> `` prefix that selects the
    pair.

    Out: exactly the inverse, in reverse order. Transliteration back to the
    target script is the step that matters most, because skipping it is silent:
    a ``tam_Taml`` request returns fluent Tamil spelled in Devanagari, and
    nothing anywhere raises.

    The placeholder map is threaded from :meth:`encode` to :meth:`decode`
    through the model object rather than a module-level queue, so a failed
    translation cannot leave a stale map behind for the next call.
    """

    #: ``tokenizer_config.json`` says ``model_max_length: 256`` and the export
    #: bakes a 256-row sinusoidal position table into the graph.
    max_source_tokens = 256

    def encode(self, model, text, src, tgt, target_token=None):
        processor = _indic_processor()
        src_tag, tgt_tag = model.native_code(src), model.native_code(tgt)
        tagged, entity_map = processor.preprocess(text, src_tag, tgt_tag)
        model._indic_entity_map = entity_map
        return self.check_length(model.tokenizer.encode(tagged), model)

    def decode(self, model, ids, src, tgt):
        processor = _indic_processor()
        raw = model.tokenizer.decode(ids)
        entity_map = getattr(model, "_indic_entity_map", None) or {}
        return processor.postprocess(raw, model.native_code(tgt), entity_map)


_PROCESSOR = None


def _indic_processor():
    """One shared :class:`IndicProcessor`; building Moses tools is not cheap."""
    global _PROCESSOR
    if _PROCESSOR is None:
        from linguonnx.translate._indic_processor import IndicProcessor
        _PROCESSOR = IndicProcessor()
    return _PROCESSOR


# --------------------------------------------------------------------------
# OpenNMT-py + subword-nmt BPE
# --------------------------------------------------------------------------

@register("opennmt-bpe")
class OpenNmtBpePipeline(Pipeline):
    """Proxecto Nós ``nos-coda_iacobus``: Moses tokenisation and BPE, both ways.

    In: Moses-tokenise with the source language's rules, apply the shipped
    ``*_35k.code`` merges, map through the source vocabulary, and add the
    ``source_offset`` that the export's concatenated ``[target | source]``
    embedding table needs.

    Out: map through the target vocabulary, remove the ``@@`` merge markers,
    Moses-detokenise. ``<unk>`` survives on purpose - see
    :class:`~linguonnx.translate.tokenizers.OpenNmtBpeTokenizer`.
    """

    def encode(self, model, text, src, tgt, target_token=None):
        return self.check_length(model.tokenizer.encode(text), model)

    def decode(self, model, ids, src, tgt):
        return model.tokenizer.decode(ids)
