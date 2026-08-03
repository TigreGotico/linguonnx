from pathlib import Path
from unittest.mock import patch

import pytest

from linguonnx import model_manager


EXPECTED = {
    # model_id: (hf_repo, license, loss, num_labels, precision)
    "glotlid": ("TigreGotico/glotlid-onnx", "Apache-2.0", "softmax", 2102, "fp32"),
    "glotlid-int8": ("TigreGotico/glotlid-onnx", "Apache-2.0", "softmax", 2102, "int8"),
    "lid176": ("TigreGotico/lid176-onnx", "CC-BY-SA-3.0", "hs", 176, "fp32"),
    "lid176-int8": ("TigreGotico/lid176-onnx", "CC-BY-SA-3.0", "hs", 176, "int8"),
    "openlid": ("TigreGotico/openlid-onnx", "GPL-3.0", "softmax", 201, "fp32"),
    "openlid-int8": ("TigreGotico/openlid-onnx", "GPL-3.0", "softmax", 201, "int8"),
    "openlid-v2": ("TigreGotico/openlid-v2-onnx", "GPL-3.0", "softmax", 200, "fp32"),
    "openlid-v2-int8": ("TigreGotico/openlid-v2-onnx", "GPL-3.0", "softmax", 200, "int8"),
}


def test_registry_has_exactly_the_expected_models():
    assert set(model_manager.list_models()) == set(EXPECTED)


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_registry_entry_fields(model_id):
    entry = model_manager.list_models()[model_id]
    repo, lic, loss, num_labels, precision = EXPECTED[model_id]
    assert entry["model_id"] == model_id
    assert entry["hf_repo"] == repo
    assert entry["license"] == lic
    assert entry["loss"] == loss
    assert entry["num_labels"] == num_labels
    assert entry["precision"] == precision
    assert entry["engine"] == "fasttext-onnx"
    assert entry["onnx_file"].endswith(".onnx")
    assert entry["size_mb"] > 0


def test_default_model_is_the_permissively_licensed_one():
    """GPL/CC-BY-SA models must never be what a caller gets without asking."""
    from linguonnx.detect import DEFAULT_MODEL_ID

    assert DEFAULT_MODEL_ID == "glotlid-int8"
    assert model_manager.registry_entry(DEFAULT_MODEL_ID)["license"] == "Apache-2.0"


def test_only_hs_models_declare_an_hs_tree_side_file():
    for model_id, entry in model_manager.list_models().items():
        has_tree = "hs_tree" in entry["side_files"]
        assert has_tree == (entry["loss"] == "hs"), model_id


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

    with patch("linguonnx.model_manager.hf_hub_download") as mock_dl:
        result = model_manager._fetch_one("some/repo", "vocab.txt", dest_dir)

    mock_dl.assert_not_called()
    assert result == dest_dir / "vocab.txt"


def test_fetch_one_redownloads_zero_byte_file(tmp_path):
    dest_dir = tmp_path / "model"
    dest_dir.mkdir()
    (dest_dir / "vocab.txt").write_bytes(b"")

    fake_downloaded = tmp_path / "blob_source"
    fake_downloaded.write_text("real content\n")

    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(fake_downloaded)) as mock_dl:
        result = model_manager._fetch_one("some/repo", "vocab.txt", dest_dir)

    mock_dl.assert_called_once()
    assert result.read_text() == "real content\n"


def test_ensure_model_files_downloads_onnx_and_side_files(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)

    fake_blob = tmp_path / "source_blob"
    fake_blob.write_text("x")

    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(fake_blob)) as mock_dl:
        paths = model_manager.ensure_model_files("glotlid-int8")

    assert paths["onnx_file"].name == "glotlid.int8.onnx"
    assert paths["vocab"].name == "vocab.txt"
    assert paths["labels"].name == "labels.json"
    assert paths["config"].name == "config.json"
    assert mock_dl.call_count == 4
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


class TestIsCached:
    """`is_cached` answers "would this cost a download?" without doing one."""

    ENTRY = {
        "model_id": "m", "hf_repo": "x/m", "onnx_file": "model.onnx",
        "side_files": {"config": "config.json"},
    }

    def _registry(self, monkeypatch, tmp_path):
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(model_manager, "registry_entry",
                            lambda model_id, kind="lid": self.ENTRY)

    def test_a_model_with_every_file_present_is_cached(self, tmp_path, monkeypatch):
        self._registry(monkeypatch, tmp_path)
        (tmp_path / "m").mkdir()
        for name in ("model.onnx", "config.json"):
            (tmp_path / "m" / name).write_bytes(b"x")
        assert model_manager.is_cached("m", kind="translate") is True

    def test_a_model_missing_one_file_is_not_cached(self, tmp_path, monkeypatch):
        self._registry(monkeypatch, tmp_path)
        (tmp_path / "m").mkdir()
        (tmp_path / "m" / "model.onnx").write_bytes(b"x")
        assert model_manager.is_cached("m", kind="translate") is False

    def test_a_zero_byte_file_is_not_cached(self, tmp_path, monkeypatch):
        """The same rule the fetch path uses, so the two cannot disagree."""
        self._registry(monkeypatch, tmp_path)
        (tmp_path / "m").mkdir()
        (tmp_path / "m" / "model.onnx").write_bytes(b"")
        (tmp_path / "m" / "config.json").write_bytes(b"x")
        assert model_manager.is_cached("m", kind="translate") is False

    def test_an_unknown_model_is_not_cached(self):
        assert model_manager.is_cached("no-such-model", kind="translate") is False
