"""Language identification engine: a transformer encoder with a
sequence-classification head (XLM-R, BERT), exported to ONNX.

The tokenizer is the export's own ``tokenizer.json``, read by the
``tokenizers`` package (the ``fast-tokenizers`` extra). Its post-processor
adds the special tokens the model was trained with (``<s> ... </s>`` for
XLM-R, ``[CLS] ... [SEP]`` for BERT), so no model-specific wrapping happens
here. The graph returns logits; a softmax over them gives one probability per
``id2label`` entry of the export's ``config.json``, which is what the
``transformers`` text-classification pipeline does for a
``single_label_classification`` head.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from linguonnx.limits import MAX_DETECT_CHARS, InputTooLongError, check_length

# RoBERTa-family models number their positions from ``pad_token_id + 1``, so
# that many rows of the position table can never hold a token.
_OFFSET_POSITIONS = {"roberta", "xlm-roberta", "camembert"}


def max_input_tokens(config: Dict) -> int:
    """Longest token sequence the encoder's position table can hold."""
    positions = int(config["max_position_embeddings"])
    if config.get("model_type") in _OFFSET_POSITIONS:
        positions -= int(config.get("pad_token_id", 1)) + 1
    return positions


def id2label(config: Dict) -> List[str]:
    """``config.json``'s ``id2label`` as a list in logit order."""
    mapping = config["id2label"]
    return [mapping[str(i)] for i in range(len(mapping))]


class SequenceClassifier:
    """Tokenizes text and turns the graph's logits into label probabilities."""

    def __init__(self, tokenizer_json, config: Dict, session,
                 max_chars: int = MAX_DETECT_CHARS):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "this model's tokenizer is a `tokenizers`-library "
                "tokenizer.json, which needs the 'tokenizers' package. "
                "Install it: pip install 'linguonnx[fast-tokenizers]'") from exc
        problem_type = config.get("problem_type") or "single_label_classification"
        if problem_type != "single_label_classification":
            raise ValueError(
                f"problem_type {problem_type!r}: only a softmax "
                "(single_label_classification) head is a language decision")
        self._tok = Tokenizer.from_file(str(tokenizer_json))
        self._tok.no_truncation()
        self._tok.no_padding()
        self.labels = id2label(config)
        self.max_chars = max_chars
        self.max_tokens = max_input_tokens(config)
        self._session = session
        self._input_names = {i.name for i in session.get_inputs()}
        self._output_name = session.get_outputs()[0].name

    def encode(self, text: str) -> np.ndarray:
        """Token ids with the model's special tokens, shape ``(1, n)``."""
        check_length(len(text), self.max_chars, "characters",
                     "LINGUONNX_MAX_DETECT_CHARS")
        # AfroLID's SentencePiece tokenizer drops leading and trailing
        # whitespace; its tokenizer.json port turns a trailing space into a
        # lone "▁" piece. A WordPiece (BERT) tokenizer ignores it either way.
        ids = self._tok.encode(text.strip()).ids
        # The position table is part of the model, so this bound has no
        # environment variable: no setting makes a longer input valid.
        if len(ids) > self.max_tokens:
            raise InputTooLongError(
                f"input is {len(ids)} tokens, over the {self.max_tokens} "
                "positions this model has; split the input")
        return np.asarray([ids], dtype=np.int64)

    def __call__(self, text: str) -> np.ndarray:
        input_ids = self.encode(text)
        feed = {"input_ids": input_ids}
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = np.ones_like(input_ids)
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        (logits,) = self._session.run([self._output_name], feed)
        logits = np.asarray(logits[0], dtype=np.float64)
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()
