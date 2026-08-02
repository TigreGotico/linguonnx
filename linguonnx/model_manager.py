"""Download + local cache for linguonnx models.

Models are cached under ``~/.cache/linguonnx/models/<model_id>/<filename>``.
Downloads go through ``huggingface_hub.hf_hub_download`` (which does its own
resumable/verified download into HF's blob cache) and are then copied into
our cache path atomically: written to a ``.part`` sibling file and
``os.replace``'d into place, so a killed process never leaves a corrupt,
non-zero-length file sitting at the final path. A zero-byte file at the
final path is always treated as "not cached" and re-downloaded.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import hf_hub_download

CACHE_ROOT = Path.home() / ".cache" / "linguonnx"
MODELS_DIR = CACHE_ROOT / "models"
REGISTRY_PATH = Path(__file__).parent / "model_index" / "lid.json"


def _load_registry() -> Dict[str, Any]:
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def registry_entry(model_id: str) -> Dict[str, Any]:
    registry = _load_registry()
    if model_id not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"unknown model_id {model_id!r}; available: {available}")
    return registry[model_id]


def _is_missing_or_empty(path: Path) -> bool:
    return not path.exists() or path.stat().st_size == 0


def _fetch_one(repo_id: str, filename: str, dest_dir: Path) -> Path:
    dest = dest_dir / filename
    if not _is_missing_or_empty(dest):
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
    tmp = dest.with_suffix(dest.suffix + ".part")
    shutil.copyfile(downloaded, tmp)
    os.replace(tmp, dest)  # atomic on POSIX
    return dest


def ensure_model_files(model_id: str) -> Dict[str, Path]:
    """Download (if needed) every file a model needs; return name -> local path."""
    entry = registry_entry(model_id)
    repo_id = entry["hf_repo"]
    dest_dir = MODELS_DIR / model_id

    paths: Dict[str, Path] = {}
    paths["onnx_file"] = _fetch_one(repo_id, entry["onnx_file"], dest_dir)
    for key, filename in entry.get("side_files", {}).items():
        paths[key] = _fetch_one(repo_id, filename, dest_dir)
    return paths


def list_models() -> Dict[str, Any]:
    return _load_registry()
