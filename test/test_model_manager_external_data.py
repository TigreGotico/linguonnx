"""``extra_files`` is authored; a graph's own ``external_data`` is a fact.

Today's failure mode, with direct evidence: a registry entry can under-list
the ``.onnx_data`` blob(s) its own graph needs. The model then fetches
"successfully" - every file the entry *knows* to ask for lands - and fails
only once something builds an ``InferenceSession`` from it, deep inside
onnxruntime::

    [ONNXRuntimeError] : 1 : FAIL : External data path validation failed
    for initializer: embed_tokens.weight   (tensorprotoutils.cc:453)

That message names neither the model nor the missing file. Four DSFSI NLLB
models cost hours of investigation to it before the real cause - one entry
omitting ``int8/encoder_model.onnx_data`` (~1.2 GB) - was found.

``ensure_model_files`` no longer trusts ``extra_files`` for this: it reads
every graph's own initializers (:func:`model_manager._external_data_locations`,
a hand-rolled protobuf reader - ``onnx`` stays a test-only dependency per
pyproject.toml) and fetches exactly what they name, so an under-listed
``extra_files`` is no longer wrong, only redundant. This module proves the
part that still has to fail loudly: when a location genuinely cannot be
made present on disk, :class:`model_manager.MissingExternalDataError` fires
by name, before any session is built - never onnxruntime's opaque one.
"""
from unittest.mock import patch

import pytest

from linguonnx import model_manager


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _len_delim(field_no: int, payload: bytes) -> bytes:
    tag = _varint((field_no << 3) | 2)
    return tag + _varint(len(payload)) + payload


def _graph_bytes(location: str) -> bytes:
    """A minimal ``ModelProto`` with one initializer external_data location.

    Shape: ``ModelProto.graph(7) { GraphProto.initializer(5) {
    TensorProto.external_data(13) { key(1)="location", value(2)=<location> }
    } }`` - the exact path :func:`model_manager._external_data_locations`
    walks, field numbers confirmed against the ``onnx`` package's own
    descriptors (see the comment above that function).
    """
    entry = _len_delim(1, b"location") + _len_delim(2, location.encode())
    tensor = _len_delim(13, entry)
    graph = _len_delim(5, tensor)
    return _len_delim(7, graph)


ENTRY = {
    "model_id": "fake-nllb",
    "hf_repo": "x/fake-nllb-onnx",
    "arch": "nllb",
    "graphs": {"encoder": "int8/encoder_model.onnx"},
    "side_files": {},
    # Deliberately under-listed, exactly like the DSFSI entries: the blob
    # the graph actually needs is missing from `extra_files`.
    "extra_files": [],
}


def _stub_registry(monkeypatch):
    monkeypatch.setattr(model_manager, "registry_entry",
                        lambda model_id, kind="lid": ENTRY)


class TestMissingExternalData:
    """The location the graph names is never reachable: onnxruntime never sees it."""

    def test_fires_the_named_error_before_any_session_could_be_built(
            self, tmp_path, monkeypatch):
        _stub_registry(monkeypatch)
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)

        graph_src = tmp_path / "graph_source.onnx"
        graph_src.write_bytes(_graph_bytes("encoder_model.onnx_data"))
        empty_blob_src = tmp_path / "empty_blob_source"
        empty_blob_src.write_bytes(b"")  # "fetched", but nothing useful landed

        def fake_download(repo_id, filename, revision=None):
            if filename == "int8/encoder_model.onnx":
                return str(graph_src)
            if filename == "int8/encoder_model.onnx_data":
                return str(empty_blob_src)
            raise AssertionError(f"unexpected fetch of {filename!r}")

        with patch("linguonnx.model_manager.hf_hub_download",
                  side_effect=fake_download):
            with pytest.raises(model_manager.MissingExternalDataError) as excinfo:
                model_manager.ensure_model_files("fake-nllb", kind="translate")

        message = str(excinfo.value)
        assert "fake-nllb" in message
        assert "int8/encoder_model.onnx_data" in message

    def test_extra_files_omitting_the_blob_is_not_fatal_on_its_own(
            self, tmp_path, monkeypatch):
        """The derive step fetches the blob anyway when the hub actually has it.

        `extra_files` being wrong stops mattering once the graph is asked
        directly - this is the point of deriving rather than trusting it.
        """
        _stub_registry(monkeypatch)
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)

        graph_src = tmp_path / "graph_source.onnx"
        graph_src.write_bytes(_graph_bytes("encoder_model.onnx_data"))
        real_blob_src = tmp_path / "real_blob_source"
        real_blob_src.write_bytes(b"weights-go-here")

        def fake_download(repo_id, filename, revision=None):
            if filename == "int8/encoder_model.onnx":
                return str(graph_src)
            if filename == "int8/encoder_model.onnx_data":
                return str(real_blob_src)
            raise AssertionError(f"unexpected fetch of {filename!r}")

        with patch("linguonnx.model_manager.hf_hub_download",
                  side_effect=fake_download):
            paths = model_manager.ensure_model_files("fake-nllb", kind="translate")

        assert paths["encoder"].exists()
        blob = paths["encoder"].parent / "encoder_model.onnx_data"
        assert blob.exists()
        assert blob.read_bytes() == b"weights-go-here"
