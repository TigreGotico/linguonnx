"""The ``transformer-onnx`` LID engine, offline.

A real ``tokenizers`` tokenizer.json (a small word-level one with XLM-R's
``Metaspace`` pre-tokenizer and ``<s> ... </s>`` post-processor) and a
stand-in ONNX session: the tests check
the ids the graph receives and the probabilities the detector reports, with
the softmax computed here by hand.
"""

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing

from linguonnx.detect import LanguageDetector
from linguonnx.limits import InputTooLongError

VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
         "\u2581bawo": 4, "\u2581ni": 5, "\u2581sannu": 6, "\u2581": 7}
ID2LABEL = {"0": "yor", "1": "hau", "2": "nan_lang"}
LOGITS = [2.0, 0.0, -1.0]


def _softmax_by_hand(logits):
    exps = [math.exp(x) for x in logits]
    return [e / sum(exps) for e in exps]


def _write_files(tmp_path, **config_overrides):
    tok = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = Metaspace()
    tok.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tok.save(str(tmp_path / "tokenizer.json"))
    config = {"model_type": "xlm-roberta", "max_position_embeddings": 8,
              "pad_token_id": 1, "id2label": ID2LABEL,
              "architectures": ["XLMRobertaForSequenceClassification"]}
    config.update(config_overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx file")
    return {"onnx_file": tmp_path / "model.onnx",
            "tokenizer_json": tmp_path / "tokenizer.json",
            "config": tmp_path / "config.json"}


def _detector(tmp_path, input_names=("input_ids", "attention_mask"),
              **config_overrides):
    paths = _write_files(tmp_path, **config_overrides)
    session = MagicMock()
    inputs = []
    for name in input_names:
        node = MagicMock()
        node.name = name
        inputs.append(node)
    session.get_inputs.return_value = inputs
    session.get_outputs.return_value = [MagicMock()]
    session.get_outputs.return_value[0].name = "logits"
    session.run.return_value = [np.asarray([LOGITS], dtype=np.float32)]
    entry = {"num_labels": 3, "engine": "transformer-onnx",
             "side_files": {"tokenizer_json": "tokenizer.json",
                            "config": "config.json"}}
    with patch("linguonnx.detect.model_manager.registry_entry", return_value=entry), \
         patch("linguonnx.detect.model_manager.ensure_model_files", return_value=paths), \
         patch("linguonnx.detect.ort.InferenceSession", return_value=session):
        return LanguageDetector(model_id="fake"), session


def test_graph_receives_the_tokenizer_ids_with_special_tokens(tmp_path):
    detector, session = _detector(tmp_path)
    detector.detect("bawo ni")
    feed = session.run.call_args[0][1]
    assert feed["input_ids"].tolist() == [[0, 4, 5, 2]]
    assert feed["input_ids"].dtype == np.int64
    assert feed["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert "token_type_ids" not in feed


def test_bert_graph_gets_token_type_ids(tmp_path):
    detector, session = _detector(
        tmp_path, input_names=("input_ids", "attention_mask", "token_type_ids"))
    detector.detect("sannu")
    feed = session.run.call_args[0][1]
    assert feed["token_type_ids"].tolist() == [[0, 0, 0]]


def test_probabilities_are_the_softmax_of_the_logits(tmp_path):
    detector, _ = _detector(tmp_path)
    expected = _softmax_by_hand(LOGITS)
    label, confidence = detector.detect_raw("bawo ni")
    assert label == "yor"
    assert confidence == pytest.approx(expected[0], abs=1e-6)
    probs = detector.detect_probs("bawo ni", top_k=3)
    assert list(probs) == ["yo", "ha", "und"]
    assert list(probs.values()) == pytest.approx(expected, abs=1e-6)


def test_labels_come_out_as_bcp47(tmp_path):
    detector, _ = _detector(tmp_path)
    assert detector.detect("bawo ni") == "yo"
    assert detector.available_languages == {"yo", "ha", "und"}


def test_input_over_the_position_table_is_refused(tmp_path):
    # 8 positions minus the pad offset (pad_token_id + 1) leaves 6 tokens:
    # 5 words plus <s> and </s> is 7.
    detector, session = _detector(tmp_path)
    with pytest.raises(InputTooLongError, match="7 tokens, over the 6"):
        detector.detect("bawo ni bawo ni sannu")
    session.run.assert_not_called()
    assert detector.detect("bawo ni bawo ni") == "yo"


def test_a_multi_label_head_is_refused(tmp_path):
    with pytest.raises(ValueError, match="multi_label_classification"):
        _detector(tmp_path, problem_type="multi_label_classification")


def test_surrounding_whitespace_is_not_tokenized(tmp_path):
    """The reference SentencePiece tokenizer drops it; this tokenizer.json,
    like AfroLID's, encodes "bawo ni " as <s> ▁bawo ▁ni ▁ </s>."""
    detector, session = _detector(tmp_path)
    detector.detect("bawo ni ")
    assert session.run.call_args[0][1]["input_ids"].tolist() == [[0, 4, 5, 2]]
    detector.detect("  bawo ni \n")
    assert session.run.call_args[0][1]["input_ids"].tolist() == [[0, 4, 5, 2]]
