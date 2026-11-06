import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from linguonnx.detect import LanguageDetector

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

    with patch("linguonnx.detect.model_manager.registry_entry",
               return_value={"num_labels": 3}), \
         patch("linguonnx.detect.model_manager.ensure_model_files",
               return_value=paths), \
         patch("linguonnx.detect.ort.InferenceSession", return_value=fake_session):
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


# ---------------------------------------------------------------------------
# Hierarchical softmax (lid.176)
# ---------------------------------------------------------------------------

HS_LABELS = ["__label__en", "__label__pt", "__label__gl"]


def _write_fake_hs_model_files(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps(HS_LABELS))
    (tmp_path / "config.json").write_text(json.dumps(
        {"dim": 4, "minn": 2, "maxn": 4, "bucket": 1000,
         "nwords": 3, "nlabels": len(HS_LABELS), "loss": "hs"}
    ))
    (tmp_path / "vocab.txt").write_text("the\nand\nhello\n")
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx file")
    # 3 labels -> 2 internal Huffman nodes. Label paths are leaf->root.
    #   en: node 1 with bit False
    #   pt: node 0 True,  node 1 True
    #   gl: node 0 False, node 1 True
    (tmp_path / "hs_tree.json").write_text(json.dumps({
        "paths": [[1], [0, 1], [0, 1]],
        "codes": [[False], [True, True], [False, True]],
    }))
    return {
        "onnx_file": tmp_path / "model.onnx",
        "vocab": tmp_path / "vocab.txt",
        "labels": tmp_path / "labels.json",
        "config": tmp_path / "config.json",
        "hs_tree": tmp_path / "hs_tree.json",
    }


def _make_detector(paths, entry, output):
    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [MagicMock()]
    fake_session.get_inputs.return_value[0].name = "input_ids"
    fake_session.get_outputs.return_value = [MagicMock()]
    fake_session.get_outputs.return_value[0].name = "node_probs"
    fake_session.run.return_value = [np.asarray(output, dtype=np.float32)]

    with patch("linguonnx.detect.model_manager.registry_entry", return_value=entry), \
         patch("linguonnx.detect.model_manager.ensure_model_files", return_value=paths), \
         patch("linguonnx.detect.ort.InferenceSession", return_value=fake_session):
        return LanguageDetector(model_id="fake"), fake_session


class TestHSCombiner:
    def test_path_walk_matches_hand_computed_probabilities(self):
        from linguonnx.detect.hs import HSCombiner

        combine = HSCombiner(
            paths=[[1], [0, 1], [0, 1]],
            codes=[[False], [True, True], [False, True]],
        )
        node_probs = np.array([0.25, 0.8], dtype=np.float32)
        got = combine(node_probs)
        # en: (1 - p1) = 0.2
        # pt: p0 * p1     = 0.25 * 0.8 = 0.2
        # gl: (1-p0) * p1 = 0.75 * 0.8 = 0.6
        assert got == pytest.approx([0.2, 0.2, 0.6], abs=1e-6)

    def test_probabilities_sum_to_one_over_a_full_tree(self):
        from linguonnx.detect.hs import HSCombiner

        combine = HSCombiner.from_counts([50, 30, 15, 5])
        rng = np.random.default_rng(0)
        node_probs = rng.uniform(0.05, 0.95, size=3)
        assert combine(node_probs).sum() == pytest.approx(1.0, abs=1e-9)

    def test_build_tree_shapes_and_determinism(self):
        from linguonnx.detect.hs import build_tree

        paths, codes = build_tree([100, 40, 40, 20, 1])
        assert len(paths) == len(codes) == 5
        assert all(len(p) == len(c) for p, c in zip(paths, codes))
        # every node index addresses one of the osz-1 internal nodes
        assert all(0 <= n < 4 for p in paths for n in p)
        # the most frequent label sits closest to the root
        assert len(paths[0]) <= min(len(p) for p in paths[1:])
        assert build_tree([100, 40, 40, 20, 1]) == (paths, codes)

    def test_deep_path_does_not_underflow(self):
        from linguonnx.detect.hs import HSCombiner

        combine = HSCombiner.from_counts([2 ** i for i in range(40)])
        probs = combine(np.full(39, 0.5))
        assert np.all(np.isfinite(probs))
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_from_file(self, tmp_path):
        from linguonnx.detect.hs import HSCombiner

        paths = _write_fake_hs_model_files(tmp_path)
        combine = HSCombiner.from_file(paths["hs_tree"])
        assert combine.paths == [[1], [0, 1], [0, 1]]


class TestLossDispatch:
    def test_hs_model_combines_node_probs(self, tmp_path):
        paths = _write_fake_hs_model_files(tmp_path)
        entry = {"num_labels": 3, "loss": "hs",
                 "side_files": {"hs_tree": "hs_tree.json"}}
        detector, _ = _make_detector(paths, entry, [0.25, 0.8])

        assert detector.loss == "hs"
        # gl wins at 0.6 - a plain argmax over the raw sigmoids would say
        # node 1 (index 1 -> "pt"), which is the wrong-by-construction answer
        assert detector.detect("hello") == "gl"
        label, conf = detector.detect_raw("hello")
        assert label == "gl"
        assert conf == pytest.approx(0.6, abs=1e-6)

    def test_softmax_model_output_is_untouched(self, tmp_path):
        paths = _write_fake_model_files(tmp_path)
        entry = {"num_labels": 3, "loss": "softmax", "side_files": {}}
        detector, _ = _make_detector(paths, entry, [0.1, 0.7, 0.2])

        assert detector.loss == "softmax"
        assert detector._hs_combiner is None
        assert detector.detect_raw("hello") == ("por_Latn", pytest.approx(0.7))

    def test_loss_falls_back_to_model_config(self, tmp_path):
        """A registry entry with no 'loss' takes the model's own config.json."""
        paths = _write_fake_hs_model_files(tmp_path)
        entry = {"num_labels": 3, "side_files": {"hs_tree": "hs_tree.json"}}
        detector, _ = _make_detector(paths, entry, [0.25, 0.8])
        assert detector.loss == "hs"

    def test_loss_defaults_to_softmax_when_nothing_declares_it(self, tmp_path):
        paths = _write_fake_model_files(tmp_path)   # config.json has no "loss"
        entry = {"num_labels": 3, "side_files": {}}
        detector, _ = _make_detector(paths, entry, [0.1, 0.7, 0.2])
        assert detector.loss == "softmax"

    def test_hs_model_without_a_tree_file_fails_loudly(self, tmp_path):
        paths = _write_fake_hs_model_files(tmp_path)
        del paths["hs_tree"]
        entry = {"num_labels": 3, "loss": "hs", "side_files": {}}
        with pytest.raises(ValueError, match="hs_tree"):
            _make_detector(paths, entry, [0.25, 0.8])

    def test_bare_labels_map_to_bcp47(self, tmp_path):
        paths = _write_fake_hs_model_files(tmp_path)
        entry = {"num_labels": 3, "loss": "hs",
                 "side_files": {"hs_tree": "hs_tree.json"}}
        detector, _ = _make_detector(paths, entry, [0.25, 0.8])
        assert detector.available_languages == {"en", "pt", "gl"}


class TestDetectorInputBounds:
    """The detect helpers sit behind unauthenticated HTTP in every deployment
    this library is meant for, so their input has to be bounded and their
    empty-input case has to be explicit."""

    def _detector(self, tmp_path, **kwargs):
        from unittest.mock import MagicMock, patch

        from linguonnx.detect import LanguageDetector

        paths = _write_fake_model_files(tmp_path)
        entry = {"num_labels": 3, "loss": "softmax", "side_files": {}}
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock()]
        session.get_inputs.return_value[0].name = "input_ids"
        session.get_outputs.return_value = [MagicMock()]
        session.get_outputs.return_value[0].name = "probs"
        session.run.return_value = [np.asarray([0.1, 0.7, 0.2], dtype=np.float32)]
        with patch("linguonnx.detect.model_manager.registry_entry", return_value=entry), \
             patch("linguonnx.detect.model_manager.ensure_model_files", return_value=paths), \
             patch("linguonnx.detect.ort.InferenceSession", return_value=session):
            return LanguageDetector(model_id="fake", **kwargs)

    def test_over_long_text_is_rejected_by_every_entry_point(self, tmp_path):
        from linguonnx.limits import InputTooLongError

        detector = self._detector(tmp_path, max_chars=20)
        text = "ola " * 20
        for call in (detector.detect, detector.detect_raw, detector.detect_probs):
            with pytest.raises(InputTooLongError, match="LINGUONNX_MAX_DETECT_CHARS"):
                call(text)

    def test_too_many_tokens_is_rejected(self, tmp_path):
        from linguonnx.limits import InputTooLongError

        detector = self._detector(tmp_path, max_chars=10_000, max_tokens=4)
        with pytest.raises(InputTooLongError, match="LINGUONNX_MAX_LINE_TOKENS"):
            detector.detect("ola " * 5)

    @pytest.mark.parametrize("text", ["", "   ", "​", "‍‍", "﻿"])
    def test_text_with_no_visible_content_is_rejected(self, tmp_path, text):
        from linguonnx.limits import EmptyInputError

        detector = self._detector(tmp_path)
        with pytest.raises(EmptyInputError):
            detector.detect(text)

    def test_normal_text_still_works(self, tmp_path):
        detector = self._detector(tmp_path)
        assert detector.detect("hello") == "pt"
