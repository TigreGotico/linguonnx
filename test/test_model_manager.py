from pathlib import Path
from unittest.mock import patch

import pytest

from lingonnx import model_manager


def test_registry_has_expected_models():
    registry = model_manager.list_models()
    assert "glotlid" in registry
    assert "glotlid-int8" in registry
    for model_id, entry in registry.items():
        assert entry["model_id"] == model_id
        assert entry["hf_repo"] == "TigreGotico/glotlid-onnx"
        assert entry["license"] == "Apache-2.0"
        assert entry["engine"] == "fasttext-onnx"
        assert entry["num_labels"] == 2102
        assert entry["onnx_file"].endswith(".onnx")


def test_registry_entry_unknown_model_raises():
    with pytest.raises(ValueError):
        model_manager.registry_entry("does-not-exist")


def test_registry_entry_known_model():
    entry = model_manager.registry_entry("glotlid-int8")
    assert entry["onnx_file"] == "glotlid.int8.onnx"
    assert entry["precision"] == "int8"


def test_is_missing_or_empty_treats_zero_byte_file_as_missing(tmp_path):
    zero_byte = tmp_path / "zero.onnx"
    zero_byte.write_bytes(b"")
    assert model_manager._is_missing_or_empty(zero_byte) is True

    nonempty = tmp_path / "nonempty.onnx"
    nonempty.write_bytes(b"data")
    assert model_manager._is_missing_or_empty(nonempty) is False

    missing = tmp_path / "missing.onnx"
    assert model_manager._is_missing_or_empty(missing) is True


def test_fetch_one_skips_download_when_already_cached(tmp_path):
    dest_dir = tmp_path / "model"
    dest_dir.mkdir()
    (dest_dir / "vocab.txt").write_text("hello\n")

    with patch("lingonnx.model_manager.hf_hub_download") as mock_dl:
        result = model_manager._fetch_one("some/repo", "vocab.txt", dest_dir)

    mock_dl.assert_not_called()
    assert result == dest_dir / "vocab.txt"


def test_fetch_one_redownloads_zero_byte_file(tmp_path):
    dest_dir = tmp_path / "model"
    dest_dir.mkdir()
    (dest_dir / "vocab.txt").write_bytes(b"")

    fake_downloaded = tmp_path / "blob_source"
    fake_downloaded.write_text("real content\n")

    with patch("lingonnx.model_manager.hf_hub_download", return_value=str(fake_downloaded)) as mock_dl:
        result = model_manager._fetch_one("some/repo", "vocab.txt", dest_dir)

    mock_dl.assert_called_once()
    assert result.read_text() == "real content\n"


def test_ensure_model_files_downloads_onnx_and_side_files(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)

    fake_blob = tmp_path / "source_blob"
    fake_blob.write_text("x")

    with patch("lingonnx.model_manager.hf_hub_download", return_value=str(fake_blob)) as mock_dl:
        paths = model_manager.ensure_model_files("glotlid-int8")

    assert paths["onnx_file"].name == "glotlid.int8.onnx"
    assert paths["vocab"].name == "vocab.txt"
    assert paths["labels"].name == "labels.json"
    assert paths["config"].name == "config.json"
    assert mock_dl.call_count == 4
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
