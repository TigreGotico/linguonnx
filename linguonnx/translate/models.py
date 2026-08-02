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
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from linguonnx.model_manager import ensure_model_files, registry_entry
from linguonnx.translate.decode import GenerationConfig, Seq2SeqDecoder
from linguonnx.translate.graph import Capability, normalize_tag
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
    return Capability(
        model_id=entry["model_id"],
        arch=entry["arch"],
        license=entry["license"],
        license_tier=entry["license_tier"],
        size_mb=int(entry["size_mb"]),
        languages=langs,
        pair=(normalize_tag(pair[0]), normalize_tag(pair[1])) if pair else None,
    )


class TranslationModel:
    """A single loaded translation model. Graphs load lazily, on first use."""

    def __init__(self, model_id: str, entry: Optional[Dict] = None):
        self.model_id = model_id
        self.entry = entry or registry_entry(model_id, kind="translate")
        self.arch = self.entry["arch"]
        self.capability = capability_from_entry(self.entry)
        self._files: Optional[Dict[str, Path]] = None
        self._decoder: Optional[Seq2SeqDecoder] = None
        self._tokenizer = None
        self._config: Optional[Dict] = None
        # model code <-> BCP-47, both ways, built from the registry's own list.
        self._to_native: Dict[str, str] = {}
        for code in self.entry.get("languages", ()):
            self._to_native.setdefault(normalize_tag(code), code)
        if self.capability.pair:
            for native, tag in zip(self.entry["pair"], self.capability.pair):
                self._to_native.setdefault(tag, native)

    # -- lazy loading -----------------------------------------------------

    @property
    def files(self) -> Dict[str, Path]:
        if self._files is None:
            self._files = ensure_model_files(self.model_id, kind="translate")
        return self._files

    @property
    def config(self) -> Dict:
        if self._config is None:
            with open(self.files["config"], encoding="utf-8") as handle:
                self._config = json.load(handle)
        return self._config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(
                self.arch, self.files, self.entry.get("languages", ()))
        return self._tokenizer

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
            )
        return self._decoder

    # -- languages --------------------------------------------------------

    @property
    def languages(self) -> frozenset:
        return self.capability.endpoints()

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
        if self.arch == "marian":
            input_ids = self.tokenizer.encode(text, target_token=target_token)
            forced_bos = None
        else:
            input_ids = self.tokenizer.encode(text, self.native_code(src))
            forced_bos = self.tokenizer.lang_id(self.native_code(tgt))
        output_ids = self.decoder.generate(
            input_ids, forced_bos_token_id=forced_bos, config=config)
        return self.tokenizer.decode(output_ids)


@lru_cache(maxsize=None)
def load_model(model_id: str) -> TranslationModel:
    """Load (and memoise) one registry model, so a pivot chain reuses sessions."""
    return TranslationModel(model_id)
