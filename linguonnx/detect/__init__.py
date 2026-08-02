"""Language identification engine: fastText feature hashing (Python) feeding
an ONNX graph that does the embedding-average + matmul + softmax.

Models trained with hierarchical softmax (``loss=hs``, e.g. fastText's
classic ``lid.176``) end in ``Sigmoid`` over Huffman-tree nodes instead, and
need one more Python step to turn node scores into label probabilities; see
:mod:`linguonnx.detect.hs`. Which path a model takes is decided by its
declared loss, never by its name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort

from linguonnx import model_manager
from linguonnx.detect.hashing import GlotLIDFeaturizer
from linguonnx.detect.hs import HSCombiner
from linguonnx.detect.labels import LABEL_PREFIX, LabelMapper, collapse_variety

# GlotLID is the default on purpose: it is the only Apache-2.0 model in the
# registry. OpenLID v1/v2 are GPL-3.0 and lid.176 is CC-BY-SA-3.0, so a user
# must ask for those by name rather than inherit their terms by accident.
DEFAULT_MODEL_ID = "glotlid-int8"


class LanguageDetector:
    """Loads one fastText-ONNX LID model + its side files and runs detection."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 session_options: Optional[ort.SessionOptions] = None):
        self.model_id = model_id
        entry = model_manager.registry_entry(model_id)
        self.num_labels = entry["num_labels"]

        paths = model_manager.ensure_model_files(model_id)

        with open(paths["labels"], encoding="utf-8") as fh:
            raw_labels = json.load(fh)
        self._raw_labels = [
            l[len(LABEL_PREFIX):] if l.startswith(LABEL_PREFIX) else l
            for l in raw_labels
        ]
        self._label_mapper = LabelMapper(raw_labels)

        with open(paths["config"], encoding="utf-8") as fh:
            self._config = json.load(fh)

        self._featurizer = GlotLIDFeaturizer(
            words=Path(paths["vocab"]).read_text(encoding="utf-8").split("\n"),
            nwords=self._config["nwords"],
            minn=self._config["minn"],
            maxn=self._config["maxn"],
            bucket=self._config["bucket"],
        )
        # from_files() trims a trailing blank line from vocab.txt; do the same
        # here since we read the file ourselves to avoid a second disk read.
        if self._featurizer.words and self._featurizer.words[-1] == "":
            self._featurizer.words.pop()
            self._featurizer.word2id = {w: i for i, w in enumerate(self._featurizer.words)}

        self._session = ort.InferenceSession(
            str(paths["onnx_file"]), sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Which output path to take is decided by the model's declared loss,
        # not by its name: the registry entry wins (it is what we ship and
        # can fix without a re-upload), the model's own config.json is the
        # fallback, and a model that declares nothing is plain softmax -
        # every fastText LID export before lid.176 was.
        self.loss = entry.get("loss") or self._config.get("loss") or "softmax"
        self._hs_combiner: Optional[HSCombiner] = None
        if self.loss == "hs":
            if "hs_tree" not in paths:
                raise ValueError(
                    f"model {model_id!r} declares loss=hs but its registry "
                    "entry has no 'hs_tree' side file"
                )
            self._hs_combiner = HSCombiner.from_file(paths["hs_tree"])

    @property
    def available_languages(self) -> set:
        return self._label_mapper.available_languages

    def _probs(self, text: str) -> np.ndarray:
        feature_ids = self._featurizer(text)
        (raw,) = self._session.run(
            [self._output_name], {self._input_name: feature_ids}
        )
        # softmax models emit per-label probabilities directly; hs models
        # emit one sigmoid per Huffman node, which only becomes a per-label
        # distribution after the path walk.
        if self._hs_combiner is not None:
            return self._hs_combiner(raw)
        return raw

    def detect_raw(self, text: str) -> tuple:
        """Return (native GlotLID label, confidence), e.g. ("glg_Latn", 0.82)."""
        probs = self._probs(text)
        best = int(np.argmax(probs))
        return self._raw_labels[best], float(probs[best])

    def detect(self, text: str, collapse_varieties: bool = False) -> str:
        """
        Return the best-guess BCP-47 tag, e.g. "gl".

        GlotLID labels individual varieties, so casual Arabic is reported as
        e.g. "ajp-Arab" (South Levantine) rather than "ar". Pass
        ``collapse_varieties=True`` to fold varieties onto the macrolanguage
        a caller can act on ("ar"); use :meth:`detect_raw` when the variety
        itself is the answer you want.
        """
        raw_label, _ = self.detect_raw(text)
        tag = self._label_mapper.to_bcp47(raw_label)
        return collapse_variety(tag) if collapse_varieties else tag

    def detect_probs(self, text: str, top_k: int = 5) -> Dict[str, float]:
        """Return the top_k BCP-47 tags with their probabilities."""
        probs = self._probs(text)
        top_k = min(top_k, len(probs))
        top_idx = np.argpartition(probs, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
        out: Dict[str, float] = {}
        for i in top_idx:
            tag = self._label_mapper.to_bcp47(self._raw_labels[int(i)])
            # multiple raw labels can collapse onto the same BCP-47 tag; keep
            # whichever has the higher probability (top_idx is already sorted
            # descending, so the first occurrence wins).
            out.setdefault(tag, float(probs[int(i)]))
        return out
