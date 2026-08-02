import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lingonnx.detect import LanguageDetector

LABELS = ["__label__eng_Latn", "__label__por_Latn", "__label__glg_Latn"]


def _write_fake_model_files(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "config.json").write_text(json.dumps(
        {"dim": 4, "minn": 2, "maxn": 5, "bucket": 1000, "wordNgrams": 1,
         "nwords": 3, "nlabels": len(LABELS)}
    ))
    (tmp_path / "vocab.txt").write_text("the\nand\nhello\n")
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx file")
    return {
        "onnx_file": tmp_path / "model.onnx",
        "vocab": tmp_path / "vocab.txt",
        "labels": tmp_path / "labels.json",
        "config": tmp_path / "config.json",
    }


@pytest.fixture
def mocked_detector(tmp_path):
    paths = _write_fake_model_files(tmp_path)

    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [MagicMock(name="input_ids")]
    fake_session.get_inputs.return_value[0].name = "input_ids"
    fake_session.get_outputs.return_value = [MagicMock(name="probs")]
    fake_session.get_outputs.return_value[0].name = "probs"
    # eng_Latn wins with high confidence
    fake_session.run.return_value = [np.array([0.9, 0.05, 0.05], dtype=np.float32)]

    with patch("lingonnx.detect.model_manager.registry_entry",
               return_value={"num_labels": 3}), \
         patch("lingonnx.detect.model_manager.ensure_model_files",
               return_value=paths), \
         patch("lingonnx.detect.ort.InferenceSession", return_value=fake_session):
        detector = LanguageDetector(model_id="glotlid-int8")
    return detector, fake_session


def test_detect_returns_bcp47(mocked_detector):
    detector, _ = mocked_detector
    assert detector.detect("hello there") == "en"


def test_detect_raw_returns_native_label_and_confidence(mocked_detector):
    detector, _ = mocked_detector
    label, conf = detector.detect_raw("hello there")
    assert label == "eng_Latn"
    assert conf == pytest.approx(0.9)


def test_detect_probs_returns_top_k(mocked_detector):
    detector, _ = mocked_detector
    probs = detector.detect_probs("hello there", top_k=2)
    assert set(probs.keys()) <= {"en", "pt", "gl"}
    assert len(probs) == 2
    assert probs["en"] == pytest.approx(0.9)


def test_available_languages(mocked_detector):
    detector, _ = mocked_detector
    assert detector.available_languages == {"en", "pt", "gl"}


def test_onnx_session_called_with_feature_ids(mocked_detector):
    detector, fake_session = mocked_detector
    detector.detect("hello")
    args, kwargs = fake_session.run.call_args
    feed = args[1]
    assert "input_ids" in feed
    assert feed["input_ids"].dtype == np.int64
