"""Download + local cache for linguonnx models.

Models are cached under ``~/.cache/linguonnx/models/<model_id>/<filename>``.
Downloads go through ``huggingface_hub.hf_hub_download`` (which does its own
resumable/verified download into HF's blob cache) and are then copied into our
cache path atomically: written to a temp sibling whose name is unique per
process *and* per call, then ``os.replace``'d into place. The uniqueness is
what makes the guarantee hold with more than one writer: two workers cold-
starting the same model write to different temp files, and whichever
``os.replace`` lands second wins with a complete file. A shared temp name would
let them interleave writes and publish a corrupt-but-non-zero file, which the
"is it zero bytes?" freshness check cannot detect - so the corruption would
survive every restart until someone deleted the cache by hand.

A zero-byte file at the final path is always treated as "not cached" and
re-downloaded.

Two things are checked before anything is written:

* the registry filename must stay inside the model's cache directory, so an
  edited registry cannot turn a download into an arbitrary file write; and
* a cold fetch bigger than ``LINGUONNX_MAX_DOWNLOAD_MB`` fails fast instead of
  holding a request thread for an hour on a slow link. Use :func:`prefetch` at
  startup to keep downloads off the request path entirely.

If a registry entry carries ``revision`` (a commit SHA - HF tags and branches
are mutable, so they pin nothing) it is passed to the hub, and if it carries a
``sha256`` for a file, the bytes are verified after the copy. Both are optional
today; ``scripts/sync_registry.py`` does not emit them yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from huggingface_hub import hf_hub_download

LOG = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "linguonnx"
MODELS_DIR = CACHE_ROOT / "models"
INDEX_DIR = Path(__file__).parent / "model_index"
REGISTRY_PATH = INDEX_DIR / "lid.json"

#: One registry file per task. ``kind`` selects which one.
REGISTRY_PATHS = {
    "lid": REGISTRY_PATH,
    "translate": INDEX_DIR / "translate.json",
}

#: Cold fetches larger than this fail fast rather than occupying a request
#: thread for an unbounded time. The biggest registry model is ~7.4 GB, so the
#: default refuses nothing; it exists so a server can lower it and keep the
#: request path predictable. ``0`` or negative disables the check.
DEFAULT_MAX_DOWNLOAD_MB = 8192

#: A download this big is worth a line in the log even on a healthy host: it is
#: the difference between a 200 ms request and a multi-minute one.
LARGE_DOWNLOAD_MB = 512


class DownloadTooLargeError(RuntimeError):
    """A cold fetch would exceed the configured download budget."""


class UnsafeRegistryPathError(ValueError):
    """A registry filename would write outside the model's cache directory."""


def _max_download_mb() -> int:
    raw = os.environ.get("LINGUONNX_MAX_DOWNLOAD_MB")
    if raw is None:
        return DEFAULT_MAX_DOWNLOAD_MB
    try:
        return int(raw)
    except ValueError:
        raise ValueError("LINGUONNX_MAX_DOWNLOAD_MB must be an integer, "
                         f"got {raw!r}") from None


def _load_registry(kind: str = "lid") -> Dict[str, Any]:
    try:
        path = REGISTRY_PATHS[kind]
    except KeyError:
        raise ValueError(f"unknown registry {kind!r}; "
                         f"available: {', '.join(sorted(REGISTRY_PATHS))}") from None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def registry_entry(model_id: str, kind: str = "lid") -> Dict[str, Any]:
    registry = _load_registry(kind)
    if model_id not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"unknown model_id {model_id!r}; available: {available}")
    return registry[model_id]


def _is_missing_or_empty(path: Path) -> bool:
    return not path.exists() or path.stat().st_size == 0


def _safe_dest(dest_dir: Path, filename: str) -> Path:
    """``dest_dir / filename``, proven to stay inside ``dest_dir``.

    A registry filename may legitimately carry a repo subfolder
    ("int8/encoder_model.onnx"), so it cannot simply be rejected for containing
    a separator. It is not caller input either - the public entry points only
    accept whitelisted registry keys - but it *is* read out of a JSON file, and
    without this check editing that one file turns into an arbitrary file write
    as the service user: ``dest_dir / "/etc/cron.d/x"`` discards ``dest_dir``
    entirely, and ``os.replace`` then overwrites the target atomically.
    """
    candidate = Path(filename)
    if candidate.is_absolute() or candidate.drive or filename.startswith("/"):
        raise UnsafeRegistryPathError(
            f"registry filename must be relative, got {filename!r}")
    if ".." in candidate.parts:
        raise UnsafeRegistryPathError(
            f"registry filename must not traverse upwards, got {filename!r}")
    dest = dest_dir / candidate
    # Resolved comparison, so a symlink already in the cache cannot redirect
    # the write either. `dest_dir` is `MODELS_DIR / model_id`, so containment
    # here implies containment in MODELS_DIR.
    root = dest_dir.resolve()
    resolved = dest.resolve()
    if resolved != root and root not in resolved.parents:
        raise UnsafeRegistryPathError(
            f"registry filename {filename!r} escapes the model cache directory")
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_one(repo_id: str, filename: str, dest_dir: Path,
               revision: Optional[str] = None,
               sha256: Optional[str] = None) -> Path:
    dest = _safe_dest(dest_dir, filename)
    if not _is_missing_or_empty(dest):
        return dest
    # The layout is kept as-is in the cache, because ONNX external-data files
    # (`*.onnx_data`) are found by relative path next to their `.onnx`.
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("fetching %s from %s%s", filename, repo_id,
             f" @ {revision}" if revision else " @ main (unpinned)")
    started = time.monotonic()
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename,
                                 revision=revision)
    # Unique per process and per call: two workers cold-starting the same model
    # must not write into one temp file, or they interleave and publish a
    # corrupt file that no later run will notice.
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        shutil.copyfile(downloaded, tmp)
        if sha256 is not None:
            actual = _sha256(tmp)
            if actual != sha256:
                raise ValueError(
                    f"checksum mismatch for {repo_id}/{filename}: registry "
                    f"says {sha256}, downloaded file is {actual}")
        os.replace(tmp, dest)  # atomic on POSIX
    finally:
        # A failed copy or verification must not leave the temp file behind.
        # The final path is untouched either way, so the next run retries.
        if tmp.exists():
            tmp.unlink()
    size_mb = dest.stat().st_size / (1024 * 1024)
    log = LOG.warning if size_mb >= LARGE_DOWNLOAD_MB else LOG.info
    log("fetched %s from %s (%.1f MB in %.1fs)", filename, repo_id, size_mb,
        time.monotonic() - started)
    return dest


def _entry_files(entry: Dict[str, Any]) -> Iterator[Tuple[Optional[str], str]]:
    """(key or None, filename) for every file the entry needs, in fetch order."""
    if "onnx_file" in entry:
        yield "onnx_file", entry["onnx_file"]
    for key, filename in entry.get("graphs", {}).items():
        yield key, filename
    for key, filename in entry.get("side_files", {}).items():
        yield key, filename
    # External weight blobs are not opened by name; they only have to sit next
    # to their graph, so they are fetched but not returned under a key.
    for filename in entry.get("extra_files", ()):
        yield None, filename


def _check_download_budget(model_id: str, entry: Dict[str, Any]) -> None:
    """Refuse an obviously oversized cold fetch before it starts.

    `size_mb` covers the whole model, so this only runs when something is
    actually missing; a warm cache never trips it. The check is deliberately
    coarse - it exists so a 5 GB fetch fails in milliseconds instead of holding
    a worker for an hour, not to account for partially cached models.
    """
    budget = _max_download_mb()
    if budget <= 0:
        return
    size_mb = int(entry.get("size_mb") or 0)
    if size_mb > budget:
        raise DownloadTooLargeError(
            f"{model_id} needs a {size_mb} MB download, over the {budget} MB "
            f"budget (LINGUONNX_MAX_DOWNLOAD_MB). Raise the budget, or call "
            f"linguonnx.model_manager.prefetch({model_id!r}) at startup so the "
            f"request path never has to fetch it.")


def ensure_model_files(model_id: str, kind: str = "lid",
                       enforce_budget: bool = True) -> Dict[str, Path]:
    """Download (if needed) every file a model needs; return name -> local path."""
    entry = registry_entry(model_id, kind)
    repo_id = entry["hf_repo"]
    revision = entry.get("revision")
    checksums = entry.get("sha256") or {}
    dest_dir = MODELS_DIR / model_id

    wanted = list(_entry_files(entry))
    if enforce_budget and any(_is_missing_or_empty(_safe_dest(dest_dir, name))
                              for _, name in wanted):
        _check_download_budget(model_id, entry)

    paths: Dict[str, Path] = {}
    for key, filename in wanted:
        path = _fetch_one(repo_id, filename, dest_dir, revision=revision,
                          sha256=checksums.get(filename))
        if key is not None:
            paths[key] = path
    return paths


def prefetch(*model_ids: str, kind: str = "lid",
             enforce_budget: bool = False) -> Dict[str, Dict[str, Path]]:
    """Warm the cache for ``model_ids`` before any request needs them.

    Call this at startup. Downloads here are off the request path, so the size
    budget that protects that path does not apply by default.
    """
    fetched = {}
    for model_id in model_ids:
        LOG.info("prefetching %s (%s)", model_id, kind)
        fetched[model_id] = ensure_model_files(model_id, kind=kind,
                                               enforce_budget=enforce_budget)
    return fetched


def list_models(kind: str = "lid") -> Dict[str, Any]:
    return _load_registry(kind)
