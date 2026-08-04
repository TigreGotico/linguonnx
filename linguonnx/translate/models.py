"""Registry -> loaded ONNX translation model.

One class, :class:`TranslationModel`, covers all three architectures. What
differs between them is only *how the target language is chosen*, and that is
the part which fails silently if you get it wrong - the model happily produces
fluent text in the wrong language and nothing raises. So it is concentrated in
:meth:`TranslationModel.translate` and stated per architecture:

``m2m100``
    Source language goes on the *input* as the first token; target language is
    forced as the decoder's first generated token
    (``forced_bos_token_id = lang_id(tgt)``).
``nllb``
    Same mechanism, but the codes are FLORES-200 (``por_Latn``), so the
    language *and* the script are selected together.
``marian``
    Nothing to select. The model is the pair. A target token is only used by
    the multi-target ``tc-big``/``ROMANCE`` models, and only on request.
``madlad``
    A ``<2xx>`` piece prepended to the *input* text, not a forced decoder id.
``indictrans2``
    A ``<src_tag> <tgt_tag> `` prefix on preprocessed text, where "preprocessed"
    includes transliterating Indic scripts into Devanagari - which then has to
    be undone on the output.
``opennmt-bpe``
    Nothing to select; the model is the pair. The work is Moses tokenisation
    and BPE, on both ends.

The per-architecture parts of that live in
:mod:`linguonnx.translate.preprocess`, one :class:`~preprocess.Pipeline` each,
so that encoding and decoding for an architecture are written next to each
other and cannot drift apart.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from linguonnx.limits import MAX_ENCODER_TOKENS, DecodeError, has_visible_content
from linguonnx.model_manager import ensure_model_files, registry_entry
from linguonnx.translate.decode import GenerationConfig, Seq2SeqDecoder
from linguonnx.translate.graph import Capability, normalize_tag
from linguonnx.translate.preprocess import Pipeline, pipeline_for
from linguonnx.translate.tokenizers import load_tokenizer

LOG = logging.getLogger(__name__)

__all__ = ["TranslationModel", "capability_from_entry"]


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options,
                                providers=["CPUExecutionProvider"])


def capability_from_entry(entry: Dict) -> Capability:
    """Turn one ``translate.json`` entry into a graph :class:`Capability`.

    Bilingual entries carry ``pair``; multilingual entries carry ``languages``.
    Both are stored in the model's *own* codes and normalised to BCP-47 here,
    so the graph never sees ``por_Latn`` and the model never sees ``pt``.
    """
    langs = frozenset(normalize_tag(code) for code in entry.get("languages", ()))
    pair = entry.get("pair")
    src_langs = entry.get("src_languages")
    tgt_langs = entry.get("tgt_languages")
    return Capability(
        model_id=entry["model_id"],
        arch=entry["arch"],
        license=entry["license"],
        license_tier=entry["license_tier"],
        size_mb=int(entry["size_mb"]),
        languages=langs,
        pair=(normalize_tag(pair[0]), normalize_tag(pair[1])) if pair else None,
        src_languages=frozenset(normalize_tag(c) for c in src_langs)
            if src_langs is not None else None,
        tgt_languages=frozenset(normalize_tag(c) for c in tgt_langs)
            if tgt_langs is not None else None,
        provenance_org=entry.get("provenance_org"),
        release_date=entry.get("release_date"),
        precision=entry.get("precision"),
        quality=entry.get("quality"),
    )


class TranslationModel:
    """A single loaded translation model. Graphs load lazily, on first use."""

    def __init__(self, model_id: str, entry: Optional[Dict] = None,
                 enforce_download_budget: bool = True):
        """``enforce_download_budget=False`` is for a model the caller named
        by hand under no operator budget: :func:`load_translator` waives the
        *router's* size cap there, and the download check has to be waived
        with it or the route it plans cannot be executed. It never overrides
        ``LINGUONNX_MAX_DOWNLOAD_MB`` - see
        :func:`linguonnx.limits.operator_budget_is_set`.
        """
        self.model_id = model_id
        self._enforce_download_budget = enforce_download_budget
        self.entry = entry or registry_entry(model_id, kind="translate")
        self.arch = self.entry["arch"]
        self.capability = capability_from_entry(self.entry)
        self._files: Optional[Dict[str, Path]] = None
        self._decoder: Optional[Seq2SeqDecoder] = None
        self._tokenizer = None
        self._config: Optional[Dict] = None
        self._banned_token_ids: Optional[FrozenSet[int]] = None
        # model code <-> BCP-47, both ways, built from the registry's own list.
        self._to_native: Dict[str, str] = {}
        for code in self.entry.get("languages", ()):
            self._to_native.setdefault(normalize_tag(code), code)
        for code in (self.entry.get("src_languages") or ()):
            self._to_native.setdefault(normalize_tag(code), code)
        for code in (self.entry.get("tgt_languages") or ()):
            self._to_native.setdefault(normalize_tag(code), code)
        if self.capability.pair:
            for native, tag in zip(self.entry["pair"], self.capability.pair):
                self._to_native.setdefault(tag, native)
        # A hand-verified fix-up for a model whose own code collides with a
        # different language's ISO tag (liv4ever-mt's `<2li>` means Livonian,
        # not Limburgish - see MARIAN_MULTILINGUAL_OVERRIDES in
        # scripts/sync_registry.py). Applied last, so it always wins.
        for tag, native in (self.entry.get("native_codes") or {}).items():
            self._to_native[tag] = native

    # -- lazy loading -----------------------------------------------------

    @property
    def files(self) -> Dict[str, Path]:
        if self._files is None:
            self._files = ensure_model_files(
                self.model_id, kind="translate",
                enforce_budget=self._enforce_download_budget)
        return self._files

    @property
    def config(self) -> Dict:
        if self._config is None:
            with open(self.files["config"], encoding="utf-8") as handle:
                self._config = json.load(handle)
        return self._config

    @property
    def banned_token_ids(self) -> FrozenSet[int]:
        """Single-token ids this model's own ``generation_config.json`` bans.

        HiTZ's Marian exports (and others) carry ``bad_words_ids`` for exactly
        the failure this guards against: the decoder's own
        ``decoder_start_token_id``/``pad_token_id`` occasionally outscores
        every real word at generation step 1, and upstream already named the
        id to ban - a plain generation loop just has to read it. Multi-token
        entries are a `transformers` phrase-ban feature this decoder does not
        implement; they are skipped rather than silently truncated to their
        first id, which could ban a token this model needs.
        """
        if self._banned_token_ids is None:
            banned: set = set()
            path = self.files.get("generation_config")
            if path is not None and path.exists():
                with open(path, encoding="utf-8") as handle:
                    gen_config = json.load(handle)
                for entry in gen_config.get("bad_words_ids") or ():
                    if isinstance(entry, (list, tuple)) and len(entry) == 1:
                        banned.add(int(entry[0]))
                    else:
                        LOG.warning(
                            "%s: ignoring multi-token bad_words_ids entry %r; "
                            "linguonnx's decoder only bans single tokens",
                            self.model_id, entry)
            self._banned_token_ids = frozenset(banned)
        return self._banned_token_ids

    @property
    def declared_forced_bos_token_id(self) -> Optional[int]:
        """The target token this export bakes into its own generation config.

        A bilingual fine-tune of a multilingual model has exactly one target,
        and upstream writes it down: every Masakhane
        ``m2m100_418M_*_rel_news_ft`` carries ``forced_bos_token_id`` in its
        ``generation_config.json``. That is a fact about the weights, and it
        outranks anything derived from a language tag - especially here, where
        the low-resource target has **no token of its own** and Masakhane
        reused Swahili's slot (``__sw__``, id 128088) for it. No amount of
        correct BCP-47 reasoning about Bambara produces that id.

        Only read for a bilingual (``pair``) entry. A multi-target export
        picks its target per call and correctly declares none - M2M100-418M
        and ``-smugri`` both do - so honouring one there would pin every
        request to a single language.
        """
        if not self.capability.pair:
            return None
        path = self.files.get("generation_config")
        if path is None or not path.exists():
            return None
        with open(path, encoding="utf-8") as handle:
            declared = json.load(handle).get("forced_bos_token_id")
        return None if declared is None else int(declared)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            # An entry that declares languages states them in the *routing*
            # spelling, so they are translated back into the model's own
            # before the tokenizer sees them. An entry that declares none
            # (a bilingual `pair`) passes an empty sequence on purpose:
            # `load_tokenizer` then reads the codes out of the export itself,
            # which is the only source NLLB's positional block has.
            #
            # A *directional* entry states its languages in
            # `src_languages`/`tgt_languages` and carries no flat `languages`
            # list. Reading only `languages` would hand such an entry the
            # empty sequence and fall back to the export's own order - which
            # for M2M100/NLLB is the positional block that misrouted 64 of
            # M2M100's 100 languages. Harmless for the Marian group models
            # this shape currently describes (their tokenizer ignores the
            # argument), and that is exactly why it must not be left to be
            # discovered by the first non-Marian entry reshaped this way.
            declared = (self.entry.get("languages")
                        or self.entry.get("src_languages")
                        or self.entry.get("tgt_languages"))
            codes = self.native_codes if declared else ()
            self._tokenizer = load_tokenizer(
                self.arch, self.files, codes, pair=self.entry.get("pair"))
        return self._tokenizer

    @property
    def pipeline(self) -> Pipeline:
        """The pre/post-processing pair for this model's architecture."""
        return pipeline_for(self.arch)

    @property
    def decoder(self) -> Seq2SeqDecoder:
        if self._decoder is None:
            config = self.config
            self._decoder = Seq2SeqDecoder(
                _session(self.files["encoder"]),
                _session(self.files["decoder"]),
                _session(self.files["decoder_with_past"]),
                eos_id=int(config["eos_token_id"]),
                pad_id=int(config["pad_token_id"]),
                decoder_start_id=int(config["decoder_start_token_id"]),
                max_input_tokens=self._max_input_tokens(),
            )
        return self._decoder

    def _max_input_tokens(self) -> int:
        """The tightest encoder bound that applies to this model.

        Three numbers can cap the input and the smallest wins: the
        library-wide `LINGUONNX_MAX_ENCODER_TOKENS`, the architecture's own
        frozen position table (IndicTrans2's is 256), and whatever the
        exported config records. Reading the config means a re-export with a
        different table is respected without a code change.
        """
        limits = [MAX_ENCODER_TOKENS]
        if self.pipeline.max_source_tokens is not None:
            limits.append(self.pipeline.max_source_tokens)
        for key in ("max_source_positions", "max_position_embeddings"):
            value = self.config.get(key)
            if isinstance(value, int) and value > 0:
                limits.append(value)
                break
        return min(limits)

    # -- languages --------------------------------------------------------

    @property
    def languages(self) -> frozenset:
        return self.capability.endpoints()

    @property
    def native_codes(self) -> Tuple[str, ...]:
        """Every language the entry declares, in *the model's own* spelling.

        The registry's ``languages`` list is normalised BCP-47 - that is what
        the routing graph needs - and it is not the spelling the export uses.
        ``normalize_tag('tl')`` is ``'fil'``; M2M100's only Tagalog token is
        ``__tl__``. Handing the normalised list to a tokenizer that reads it
        as the model's own codes is what silently misrouted 64 of
        M2M100-418M's 100 languages; see
        :class:`~linguonnx.translate.tokenizers.SpmSeq2SeqTokenizer`.

        ``_to_native`` already holds that translation, ``native_codes``
        overrides included. The declared order is preserved because NLLB's
        language block is still positional.
        """
        declared: List[str] = list(self.entry.get("languages") or ())
        if not declared:
            declared = list(self.entry.get("src_languages") or ())
            declared += [tag for tag in (self.entry.get("tgt_languages") or ())
                         if tag not in declared]
        if not declared:
            declared = list(self.capability.pair or ())
        out: List[str] = []
        for tag in declared:
            native = self._to_native.get(normalize_tag(tag), tag)
            if native not in out:
                out.append(native)
        return tuple(out)

    def native_code(self, tag: str) -> str:
        """BCP-47 ``pt`` -> this model's own code (``pt``, ``por_Latn``, ...)."""
        tag = normalize_tag(tag)
        if tag not in self._to_native:
            raise KeyError(
                f"{self.model_id} does not support {tag!r}; "
                f"it supports {len(self._to_native)} languages")
        return self._to_native[tag]

    # -- translation ------------------------------------------------------

    def translate(self, text: str, src: str, tgt: str,
                  config: Optional[GenerationConfig] = None,
                  target_token: Optional[str] = None) -> str:
        if not text.strip():
            return ""
        config = config or GenerationConfig()
        # A multi-target Marian group model (opus-mt-en-sla and friends) picks
        # its target language from a prefix token, and picks it *wrong* when
        # the token is absent - fluently, with nothing raised. The registry
        # records the token the export was verified against; an explicit
        # caller argument still wins.
        if target_token is None:
            target_token = self.entry.get("target_token")
        if target_token is None and self.arch in ("marian", "madlad") \
                and not self.capability.pair:
            # A multilingual model that picks its target with a `<2xx>`
            # prefix token (MADLAD, liv4ever-mt) rather than a forced
            # decoder-start id: the token is built per call from `tgt`.
            # The registry records the template for the Marian group models,
            # where it is a property of the export. For MADLAD it is a
            # property of the *architecture* - every MADLAD checkpoint reads
            # `<2xx>` - so the pipeline supplies it and a registry entry that
            # forgets it cannot silently disable target selection.
            template = (self.entry.get("target_token_template")
                        or self.pipeline.default_target_token_template)
            if template:
                target_token = template.format(code=self.native_code(tgt))
        banned = self.banned_token_ids
        if banned - config.banned_token_ids:
            # Merge rather than replace: a caller-supplied config may already
            # carry its own bans, and neither side should silently win.
            config = GenerationConfig(
                max_new_tokens=config.max_new_tokens, num_beams=config.num_beams,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping,
                banned_token_ids=config.banned_token_ids | banned)
        pipeline = self.pipeline
        input_ids = pipeline.encode(self, text, src, tgt,
                                    target_token=target_token)
        output_ids = self.decoder.generate(
            input_ids, forced_bos_token_id=pipeline.forced_bos(self, tgt),
            config=config)
        result = pipeline.decode(self, output_ids, src, tgt)
        if not has_visible_content(result):
            # The decoder produced *something* (empty `output_ids` already
            # raises DecodeError inside Seq2SeqDecoder) but every token it
            # emitted was a special one the tokenizer strips on the way out -
            # exactly what a real decode failure looks like from here, and
            # exactly indistinguishable from "" if it were let through. See
            # linguonnx#42: mt-hitz-gl-eu did this for every input, silently,
            # behind an HTTP 200.
            raise DecodeError(
                f"{self.model_id} produced no visible output translating "
                f"{src!r} -> {tgt!r}; decoding emitted only special tokens")
        return result


@lru_cache(maxsize=None)
def load_model(model_id: str) -> TranslationModel:
    """Load (and memoise) one registry model, so a pivot chain reuses sessions."""
    return TranslationModel(model_id)
