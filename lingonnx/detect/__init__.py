"""Language identification engine: GlotLID feature hashing (Python) feeding
an ONNX graph that does the embedding-average + matmul + softmax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort

from lingonnx import model_manager
from lingonnx.detect.hashing import GlotLIDFeaturizer
from lingonnx.detect.labels import LABEL_PREFIX, LabelMapper

DEFAULT_MODEL_ID = "glotlid-int8"


class LanguageDetector:
    """Loads one GlotLID ONNX model + its side files and runs detection."""

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

    @property
    def available_languages(self) -> set:
        return self._label_mapper.available_languages

    def _probs(self, text: str) -> np.ndarray:
        feature_ids = self._featurizer(text)
        (probs,) = self._session.run(
            [self._output_name], {self._input_name: feature_ids}
        )
        return probs

    def detect_raw(self, text: str) -> tuple:
        """Return (native GlotLID label, confidence), e.g. ("glg_Latn", 0.82)."""
        probs = self._probs(text)
        best = int(np.argmax(probs))
        return self._raw_labels[best], float(probs[best])

    def detect(self, text: str) -> str:
        """Return the best-guess BCP-47 tag, e.g. "gl"."""
        raw_label, _ = self.detect_raw(text)
        return self._label_mapper.to_bcp47(raw_label)

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
