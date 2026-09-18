"""Download safety: unique temp files, path containment, pinning, budgets.

Nothing here touches the network - `hf_hub_download` is always replaced by a
stub that hands back a local file.
"""

import hashlib
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from linguonnx import model_manager
from linguonnx.model_manager import (DownloadTooLargeError,
                                     UnsafeRegistryPathError)

#: A minimal, syntactically-valid (but empty) protobuf message - see the
#: identical constant and comment in test_model_manager.py. Stands in for a
#: real `.onnx` graph wherever a fixture's fake blob content is fetched
#: through `ensure_model_files`/`prefetch` (which now parses every graph for
#: its `external_data`, and - since R3 - raises rather than silently
#: swallowing non-empty content it cannot parse). A single byte like `b"x"`
#: is a truncated varint and would trip that raise.
FAKE_ONNX_BYTES = b"\x9a\x06\x00"


# -- R1: concurrent writers must not publish a corrupt file -------------------

def test_temp_file_name_is_unique_per_call(tmp_path):
    """Two calls for the same destination must never pick the same `.part`."""
    dest_dir = tmp_path / "model"
    seen = []

    real_copyfile = model_manager.shutil.copyfile

    def spy(src, dst):
        seen.append(Path(dst).name)
        return real_copyfile(src, dst)

    blob = tmp_path / "blob"
    blob.write_text("payload")

    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)), \
            patch("linguonnx.model_manager.shutil.copyfile", side_effect=spy):
        model_manager._fetch_one("x/repo", "a.onnx", dest_dir)
        (dest_dir / "a.onnx").unlink()
        model_manager._fetch_one("x/repo", "a.onnx", dest_dir)

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert str(os.getpid()) in seen[0]


def test_two_concurrent_writers_leave_a_complete_file(tmp_path):
    """The interleaving that a shared `.part` name allowed must be impossible.

    Both threads copy a *different* payload of the same length. With one shared
    temp file the published bytes can be a mix of the two; with per-call temp
    files the winner's copy is whole.
    """
    dest_dir = tmp_path / "model"
    dest_dir.mkdir()
    payloads = {"A": b"A" * 4096, "B": b"B" * 4096}
    blobs = {}
    for tag, data in payloads.items():
        blob = tmp_path / f"blob_{tag}"
        blob.write_bytes(data)
        blobs[tag] = str(blob)

    barrier = threading.Barrier(2)

    def chunked_copy(src, dst):
        # Copy in two halves with a rendezvous between them, which is exactly
        # the window where two writers sharing one temp name would corrupt it.
        data = Path(src).read_bytes()
        half = len(data) // 2
        with open(dst, "wb") as handle:
            handle.write(data[:half])
            handle.flush()
            barrier.wait(timeout=10)
            handle.write(data[half:])

    def fake_download(repo_id, filename, revision=None):
        # Each thread downloads its own payload; the destination is shared.
        return blobs[threading.current_thread().name]

    def worker():
        model_manager._fetch_one("x/repo", "w.onnx", dest_dir)

    # Patched once, around both threads: unwinding a patch from one thread
    # while the other is inside it would be its own race.
    with patch("linguonnx.model_manager.hf_hub_download", side_effect=fake_download), \
            patch("linguonnx.model_manager.shutil.copyfile", side_effect=chunked_copy):
        threads = [threading.Thread(target=worker, name=tag) for tag in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

    published = (dest_dir / "w.onnx").read_bytes()
    assert published in payloads.values(), "published file is a mix of two writers"
    # No temp files survive either.
    assert not list(dest_dir.glob("*.part"))


# -- R2: registry filenames must stay inside the cache ------------------------

@pytest.mark.parametrize("filename", [
    "../../../../../../tmp/pwned.onnx",
    "sub/../../escape.onnx",
    "..",
    "a/../../b.onnx",
])
def test_traversal_filenames_are_rejected(tmp_path, filename):
    with pytest.raises(UnsafeRegistryPathError):
        model_manager._safe_dest(tmp_path / "model", filename)


@pytest.mark.parametrize("filename", ["/etc/cron.d/x", "/tmp/pwned.onnx"])
def test_absolute_filenames_are_rejected(tmp_path, filename):
    with pytest.raises(UnsafeRegistryPathError):
        model_manager._safe_dest(tmp_path / "model", filename)


def test_fetch_one_rejects_traversal_before_creating_anything(tmp_path):
    dest_dir = tmp_path / "model"
    outside = tmp_path / "outside"
    outside.mkdir()
    with patch("linguonnx.model_manager.hf_hub_download") as mock_dl:
        with pytest.raises(UnsafeRegistryPathError):
            model_manager._fetch_one("x/repo", "../outside/pwned.onnx", dest_dir)
    mock_dl.assert_not_called()
    assert not dest_dir.exists()
    assert list(outside.iterdir()) == []


def test_subfolder_filenames_are_still_allowed(tmp_path):
    """A repo subfolder is legitimate - int8 exports live in one."""
    dest = model_manager._safe_dest(tmp_path / "model", "int8/encoder_model.onnx")
    assert dest == tmp_path / "model" / "int8" / "encoder_model.onnx"


def test_ensure_model_files_rejects_a_poisoned_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path / "cache")
    poisoned = {
        "evil": {"model_id": "evil", "hf_repo": "x/evil", "size_mb": 1,
                 "onnx_file": "../../../../../../tmp/pwned.onnx",
                 "graphs": {}, "side_files": {}},
    }
    monkeypatch.setattr(model_manager, "_load_registry", lambda kind="lid": poisoned)
    with patch("linguonnx.model_manager.hf_hub_download") as mock_dl:
        with pytest.raises(UnsafeRegistryPathError):
            model_manager.ensure_model_files("evil")
    mock_dl.assert_not_called()


# -- R3: revision pinning and checksum verification ---------------------------

def test_revision_is_passed_through_when_the_entry_pins_one(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(FAKE_ONNX_BYTES)
    entry = {"model_id": "m", "hf_repo": "x/m", "size_mb": 1,
             "revision": "0123456789abcdef0123456789abcdef01234567",
             "onnx_file": "m.onnx", "graphs": {}, "side_files": {}}
    monkeypatch.setattr(model_manager, "_load_registry", lambda kind="lid": {"m": entry})

    with patch("linguonnx.model_manager.hf_hub_download",
               return_value=str(blob)) as mock_dl:
        model_manager.ensure_model_files("m")

    assert mock_dl.call_args.kwargs["revision"] == entry["revision"]


def test_no_revision_still_works_and_asks_for_none(tmp_path, monkeypatch):
    """Registry entries do not carry `revision` yet; absence must be fine."""
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(FAKE_ONNX_BYTES)
    with patch("linguonnx.model_manager.hf_hub_download",
               return_value=str(blob)) as mock_dl:
        model_manager.ensure_model_files("glotlid-int8")
    assert mock_dl.call_args.kwargs["revision"] is None


def test_matching_sha256_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(b"real weights")
    digest = hashlib.sha256(b"real weights").hexdigest()
    entry = {"model_id": "m", "hf_repo": "x/m", "size_mb": 1,
             "onnx_file": "m.onnx", "graphs": {}, "side_files": {},
             "sha256": {"m.onnx": digest}}
    monkeypatch.setattr(model_manager, "_load_registry", lambda kind="lid": {"m": entry})

    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)):
        paths = model_manager.ensure_model_files("m")
    assert paths["onnx_file"].read_bytes() == b"real weights"


def test_mismatched_sha256_raises_and_publishes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(b"tampered weights")
    entry = {"model_id": "m", "hf_repo": "x/m", "size_mb": 1,
             "onnx_file": "m.onnx", "graphs": {}, "side_files": {},
             "sha256": {"m.onnx": hashlib.sha256(b"real weights").hexdigest()}}
    monkeypatch.setattr(model_manager, "_load_registry", lambda kind="lid": {"m": entry})

    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)):
        with pytest.raises(ValueError, match="checksum mismatch"):
            model_manager.ensure_model_files("m")

    model_dir = tmp_path / "m"
    assert not (model_dir / "m.onnx").exists()
    assert not list(model_dir.glob("*.part"))


# -- R10: download budget and prefetch ----------------------------------------

def test_oversized_cold_fetch_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "10")
    with patch("linguonnx.model_manager.hf_hub_download") as mock_dl:
        with pytest.raises(DownloadTooLargeError, match="prefetch"):
            model_manager.ensure_model_files("glotlid")
    mock_dl.assert_not_called()


def test_budget_does_not_apply_to_a_warm_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(FAKE_ONNX_BYTES)
    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)):
        model_manager.ensure_model_files("glotlid-int8")
    # Now everything is cached; a tiny budget must not break the warm path.
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "1")
    with patch("linguonnx.model_manager.hf_hub_download") as mock_dl:
        model_manager.ensure_model_files("glotlid-int8")
    mock_dl.assert_not_called()


def test_budget_of_zero_disables_the_check(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "0")
    blob = tmp_path / "blob"
    blob.write_bytes(FAKE_ONNX_BYTES)
    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)):
        model_manager.ensure_model_files("glotlid")


def test_non_integer_budget_is_reported_clearly(monkeypatch):
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "lots")
    with pytest.raises(ValueError, match="must be an integer"):
        model_manager._max_download_mb()


def test_prefetch_ignores_the_request_path_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "1")
    blob = tmp_path / "blob"
    blob.write_bytes(FAKE_ONNX_BYTES)
    with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)):
        fetched = model_manager.prefetch("glotlid", "glotlid-int8")
    assert set(fetched) == {"glotlid", "glotlid-int8"}
    assert fetched["glotlid"]["onnx_file"].exists()
    # The environment is untouched afterwards.
    assert os.environ["LINGUONNX_MAX_DOWNLOAD_MB"] == "1"


def test_prefetch_can_keep_the_budget_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("LINGUONNX_MAX_DOWNLOAD_MB", "1")
    with pytest.raises(DownloadTooLargeError):
        model_manager.prefetch("glotlid", enforce_budget=True)


# -- R9: an operator can see a download happen --------------------------------

def test_download_is_logged_with_repo_and_size(tmp_path, caplog):
    blob = tmp_path / "blob"
    blob.write_bytes(b"z" * 2048)
    with caplog.at_level("INFO", logger="linguonnx.model_manager"):
        with patch("linguonnx.model_manager.hf_hub_download", return_value=str(blob)):
            model_manager._fetch_one("x/repo", "a.onnx", tmp_path / "model")
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "fetching a.onnx from x/repo" in messages
    assert "fetched a.onnx from x/repo" in messages
    assert "MB in" in messages
