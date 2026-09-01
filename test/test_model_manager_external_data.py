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

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

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


class TestNonEmptyMalformedGraphsAreNotSilentlySkipped:
    """A truncated/corrupt graph must be heard from, not silently treated as
    "no external data" - that silence is exactly how an under-listed blob
    went unnoticed for the DSFSI models in the first place."""

    def test_a_truncated_non_empty_graph_raises_instead_of_being_skipped(
            self, tmp_path, monkeypatch):
        _stub_registry(monkeypatch)
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)

        graph_src = tmp_path / "graph_source.onnx"
        # Non-empty, but truncated mid-varint: unparseable, not absent.
        graph_src.write_bytes(b"\xff\xff\xff")

        def fake_download(repo_id, filename, revision=None):
            if filename == "int8/encoder_model.onnx":
                return str(graph_src)
            raise AssertionError(f"unexpected fetch of {filename!r}")

        with patch("linguonnx.model_manager.hf_hub_download",
                  side_effect=fake_download):
            with pytest.raises((IndexError, ValueError)):
                model_manager.ensure_model_files("fake-nllb", kind="translate")

    def test_a_genuinely_empty_fetch_does_not_raise_the_parse_error(
            self, tmp_path, monkeypatch):
        """An empty graph is already a broken fetch on its own terms; this
        module's parser has nothing to add, so it stays quiet rather than
        piling a second, confusing error onto the same underlying problem."""
        _stub_registry(monkeypatch)
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)

        graph_src = tmp_path / "graph_source.onnx"
        graph_src.write_bytes(b"")

        def fake_download(repo_id, filename, revision=None):
            if filename == "int8/encoder_model.onnx":
                return str(graph_src)
            raise AssertionError(f"unexpected fetch of {filename!r}")

        with patch("linguonnx.model_manager.hf_hub_download",
                  side_effect=fake_download):
            paths = model_manager.ensure_model_files("fake-nllb", kind="translate")
        assert paths["encoder"].stat().st_size == 0


class TestIsCachedAgreesWithEnsureModelFiles:
    """`is_cached` must not say True on a cache `ensure_model_files` would
    still have to fetch into - that disagreement is R1: an under-listed entry
    used to report cached, so routing counted its download as free."""

    def _write_stub_cache(self, tmp_path, with_blob: bool):
        model_dir = tmp_path / "fake-nllb" / "int8"
        model_dir.mkdir(parents=True)
        (model_dir / "encoder_model.onnx").write_bytes(
            _graph_bytes("encoder_model.onnx_data"))
        if with_blob:
            (model_dir / "encoder_model.onnx_data").write_bytes(b"weights")

    def test_not_cached_when_the_graph_s_own_blob_is_missing(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(model_manager, "registry_entry",
                            lambda model_id, kind="lid": ENTRY)
        self._write_stub_cache(tmp_path, with_blob=False)
        assert model_manager.is_cached("fake-nllb", kind="translate") is False

    def test_cached_once_the_derived_blob_is_present_despite_empty_extra_files(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(model_manager, "registry_entry",
                            lambda model_id, kind="lid": ENTRY)
        self._write_stub_cache(tmp_path, with_blob=True)
        assert model_manager.is_cached("fake-nllb", kind="translate") is True


class TestBudgetSeesDerivedBlobsToo:
    """The download budget must run even when every registry-listed file is
    already on disk but the graph's own blob is not - the other half of R1:
    the budget gate must not wave through what `is_cached` would have caught."""

    def test_budget_still_fires_when_only_the_derived_blob_is_missing(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
        big_entry = dict(ENTRY, size_mb=9000)
        monkeypatch.setattr(model_manager, "registry_entry",
                            lambda model_id, kind="lid": big_entry)
        monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "100")

        model_dir = tmp_path / "fake-nllb" / "int8"
        model_dir.mkdir(parents=True)
        (model_dir / "encoder_model.onnx").write_bytes(
            _graph_bytes("encoder_model.onnx_data"))
        # The registry-listed graph is present; only the derived blob is not.

        with pytest.raises(model_manager.DownloadTooLargeError):
            model_manager.ensure_model_files("fake-nllb", kind="translate")


def test_matches_real_onnx_reader_on_a_real_external_data_export(tmp_path):
    """Differential against `onnx`'s own loader on a *real* export.

    The other tests in this module hand-assemble bytes with the same field
    numbers (7/5/13/1/2) `_external_data_locations` parses - a consistent
    transposition on both sides would still ship green. This builds a real
    graph, saves it with `onnx.save_model(..., save_as_external_data=True)`,
    and checks the hand-rolled reader against `onnx.load(...,
    load_external_data=False)`'s own answer, independent of what field
    numbers this test file otherwise assumes.
    """
    weight = numpy_helper.from_array(
        np.arange(6, dtype=np.float32).reshape(2, 3), name="w")
    node = helper.make_node("Identity", ["w"], ["out"])
    graph = helper.make_graph(
        [node], "g", inputs=[],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
        initializer=[weight])
    model = helper.make_model(graph)

    onnx_path = tmp_path / "model.onnx"
    onnx.save_model(model, str(onnx_path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.onnx_data",
                    size_threshold=0)

    reference = sorted({
        data_entry.value
        for initializer in
        onnx.load(str(onnx_path), load_external_data=False).graph.initializer
        for data_entry in initializer.external_data
        if data_entry.key == "location"
    })
    actual = sorted(model_manager._external_data_locations(onnx_path))
    assert actual == reference == ["model.onnx_data"]
