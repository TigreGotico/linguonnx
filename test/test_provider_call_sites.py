"""Regression coverage for the two real ``InferenceSession`` call sites:
``linguonnx.detect.LanguageDetector`` and
``linguonnx.translate.models.TranslationModel``. Both must default to
CPU-only, since existing deployments pin this package and must not change
behaviour by upgrading it, and both must honour an explicit ``providers=``.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from linguonnx.detect import LanguageDetector
from linguonnx.providers import CPU_PROVIDER
from linguonnx.translate import models as translate_models

LABELS = ["__label__eng_Latn", "__label__por_Latn"]


def _write_fake_lid_files(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "config.json").write_text(json.dumps(
        {"dim": 4, "minn": 2, "maxn": 5, "bucket": 1000, "wordNgrams": 1,
         "nwords": 2, "nlabels": len(LABELS)}
    ))
    (tmp_path / "vocab.txt").write_text("the\nand\n")
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx file")
    return {
        "onnx_file": tmp_path / "model.onnx",
        "vocab": tmp_path / "vocab.txt",
        "labels": tmp_path / "labels.json",
        "config": tmp_path / "config.json",
    }


def _names(resolved):
    return [p[0] if isinstance(p, tuple) else p for p in resolved]


class TestDetectorSessionDefaultsToCpu(object):
    def _build(self, tmp_path, fake_session, providers=None):
        paths = _write_fake_lid_files(tmp_path)
        with patch("linguonnx.detect.model_manager.registry_entry",
                   return_value={"num_labels": 2}), \
             patch("linguonnx.detect.model_manager.ensure_model_files",
                   return_value=paths), \
             patch("linguonnx.detect.ort.InferenceSession",
                   return_value=fake_session) as session_ctor:
            LanguageDetector(model_id="glotlid-int8", providers=providers)
        return session_ctor

    def test_default_resolves_to_cpu_only(self, tmp_path):
        session_ctor = self._build(tmp_path, MagicMock())
        _, kwargs = session_ctor.call_args
        assert _names(kwargs["providers"]) == [CPU_PROVIDER]

    def test_explicit_providers_are_honoured(self, tmp_path):
        with patch("linguonnx.detect.model_manager.registry_entry",
                   return_value={"num_labels": 2}):
            with patch("linguonnx.providers.available_providers",
                       return_value=["CUDAExecutionProvider", CPU_PROVIDER]):
                session_ctor = self._build(
                    tmp_path, MagicMock(), providers=["CUDAExecutionProvider"])
        _, kwargs = session_ctor.call_args
        assert _names(kwargs["providers"]) == ["CUDAExecutionProvider", CPU_PROVIDER]


class TestTranslateSessionDefaultsToCpu:
    def test_session_helper_defaults_to_cpu_only(self, tmp_path):
        model_path = tmp_path / "encoder.onnx"
        model_path.write_bytes(b"not a real onnx file")
        with patch("linguonnx.providers.onnxruntime.InferenceSession",
                  return_value=MagicMock()) as session_ctor:
            translate_models._session(model_path)
        _, kwargs = session_ctor.call_args
        assert _names(kwargs["providers"]) == [CPU_PROVIDER]

    def test_decoder_threads_providers_through_all_three_sessions(self, tmp_path):
        """`TranslationModel.decoder` builds encoder/decoder/decoder_with_past
        through the same `_session` helper, and its `providers=` argument
        must reach every one of them."""
        calls = []

        def _named(names):
            io = []
            for name in names:
                m = MagicMock()
                m.name = name
                io.append(m)
            return io

        def _fake_ort_session(inputs, outputs):
            session = MagicMock()
            session.get_inputs.return_value = _named(inputs)
            session.get_outputs.return_value = _named(outputs)
            return session

        encoder = _fake_ort_session(["input_ids"], ["encoder_hidden_states"])
        decoder = _fake_ort_session(
            ["input_ids", "encoder_hidden_states", "encoder_attention_mask"],
            ["logits"])
        decoder_past = _fake_ort_session(
            ["input_ids", "encoder_attention_mask"], ["logits"])
        by_role = iter([encoder, decoder, decoder_past])

        def fake_session(path, providers=None):
            calls.append(providers)
            return next(by_role)

        entry = {
            "model_id": "toy", "arch": "marian", "license": "MIT",
            "license_tier": "permissive", "size_mb": 1, "pair": ["en", "pt"],
        }
        model = translate_models.TranslationModel(
            "toy", entry, providers=["CUDAExecutionProvider"])
        model._files = {
            "encoder": tmp_path / "encoder.onnx",
            "decoder": tmp_path / "decoder.onnx",
            "decoder_with_past": tmp_path / "decoder_with_past.onnx",
        }
        model._config = {
            "eos_token_id": 2, "pad_token_id": 1, "decoder_start_token_id": 0,
        }
        with patch.object(translate_models, "_session", side_effect=fake_session):
            model.decoder
        assert calls == [["CUDAExecutionProvider"]] * 3
