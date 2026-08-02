#!/usr/bin/env python3
"""Regenerate ``linguonnx/model_index/*.json`` from the HuggingFace API.

The registries are **generated**, not hand-maintained. Exports land in the
``TigreGotico`` org faster than a human can copy file lists into JSON, and a
stale registry is not a cosmetic problem: a missing entry means a language pair
silently routes the long way round, and a *wrong* entry means the router picks
a model that cannot do the pair and returns fluent text in the wrong language.

Run it::

    python scripts/sync_registry.py            # rewrite the JSON in place
    python scripts/sync_registry.py --check    # exit 1 if the committed JSON drifted

What is derived from where
--------------------------

Everything that can be read off the Hub is read off the Hub:

``arch``
    ``config.model_type`` from the repo's own config. ``marian`` is Marian.
    Otherwise the tokenizer decides: a repo with ``vocab.json`` reads ids
    straight out of it (M2M100), a repo without one uses fairseq's
    ``id = sp_id + 1`` convention (NLLB). This is a file-level fact, not a
    guess from the repo name.
``license``
    ``cardData.license`` when the card has YAML front matter. Most opus-mt
    exports do not, so the fallback is the ``**License:** ...`` line the export
    script writes into the README body. There is no third fallback: a repo
    whose licence cannot be read is skipped, because "probably Apache" is not
    a licence claim this library is willing to publish.
``languages`` (multilingual models)
    ``additional_special_tokens`` in ``special_tokens_map.json`` - the model's
    own tokenizer metadata. Never a list typed out here.
``pair`` (bilingual models)
    The repo name, **cross-checked against the base model named in the card**.
    See :func:`_marian_pair` - this is where the sharp edges are.
``size_mb``
    Summed blob sizes of the files the entry actually references. Not the
    repo total: the repos carry a ``decoder_model_merged.onnx`` that linguonnx
    never downloads.

Sharp edges, and why the base-model cross-check exists
------------------------------------------------------

A repo called ``opus-mt-en-pl-onnx`` is not necessarily an export of
``Helsinki-NLP/opus-mt-en-pl``. Several are exports of *group* models:

- ``opus-mt-pt-en-onnx`` is ``opus-mt-ROMANCE-en`` - multi-*source*. Harmless:
  the source language is inferred from the text, so ``pt -> en`` still holds.
- ``opus-mt-en-pl-onnx`` is ``opus-mt-en-sla`` - multi-*target*. **Not**
  harmless: without a ``>>pol<<`` prefix token on the input the model picks a
  Slavic language on its own and you get Czech. The token is recorded as
  ``target_token`` and applied automatically by ``TranslationModel.translate``.
- ``opus-mt-en-pt``/``en-tr``/``en-ko`` are ``tc-big`` variants - a bigger
  model for the same pair, and a different licence (cc-by-4.0, not apache-2.0).

So: when the base model's target segment differs from the repo's target, a
target token is **required**. If the README does not show one, the entry is
skipped rather than published with a coverage claim that cannot be honoured.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import http.client
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HF_API = "/api/models"
HF_RAW = "/{repo}/raw/main/{path}"
AUTHOR = "TigreGotico"

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "linguonnx" / "model_index"

#: Repo name -> registry model id, where stripping ``-onnx`` gives the wrong
#: answer. Kept short on purpose; anything not listed uses the derived id.
MODEL_ID_OVERRIDES: Dict[str, str] = {
    "nllb-200-distilled-600M-onnx": "nllb-600M",
    "aina-translator-es-oc-onnx": "aina-es-oc",
}

#: Bilingual fine-tunes of a multilingual base, where the language codes the
#: model expects are *not* recoverable from the TigreGotico card and had to be
#: read from the upstream source card. Each entry cites where it came from.
#:
#: ``aina-es-oc``: a full fine-tune of NLLB-200-600M for Spanish -> Aranese.
#: Aranese is not in NLLB, so Projecte Aina *added* a token for it - and chose
#: ``arn_Latn``, which collides with the ISO code for Mapudungun. The pair is
#: stated in the model's own codes; ``normalize_tag`` maps ``arn_Latn`` to
#: ``oc`` so the routing graph never sees the collision.
#: Source: https://huggingface.co/projecte-aina/aina-translator-es-oc
#:   "Since the original NLLB-200-600M doesn't support Aranese, we added a new
#:    token ("arn_Latn") to enable translation into Aranese."
BILINGUAL_FINETUNES: Dict[str, Dict] = {
    "aina-es-oc": {
        "pair": ["spa_Latn", "arn_Latn"],
        "notes": "Spanish -> Aranese (Gascon Occitan). NLLB-200-600M fine-tune "
                 "by Projecte Aina; Aranese uses the added token 'arn_Latn'. "
                 "One-way: there is no Aranese -> Spanish direction.",
    },
}

#: Canonical spelling for the licence ids the Hub reports lowercased.
LICENSE_DISPLAY: Dict[str, str] = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-3.0": "CC-BY-SA-3.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "gpl-3.0": "GPL-3.0",
    "afl-3.0": "AFL-3.0",
}

#: Licence -> routing tier. The graph will not put a more restrictive tier into
#: a caller's output than they asked for, so this classification is load-bearing.
LICENSE_TIERS: Dict[str, str] = {
    "Apache-2.0": "permissive",
    "MIT": "permissive",
    "CC-BY-4.0": "permissive",
    "CC-BY-SA-3.0": "share-alike",
    "CC-BY-SA-4.0": "share-alike",
    "GPL-3.0": "share-alike",
    "AFL-3.0": "permissive",
    "CC-BY-NC-4.0": "non-commercial",
}

#: Side files per architecture: registry key -> filename in the repo.
#: A file that is absent from the repo is dropped from the entry, except for
#: the ones listed in REQUIRED_SIDE_FILES.
SIDE_FILES: Dict[str, Dict[str, str]] = {
    "marian": {
        "source_spm": "source.spm",
        "target_spm": "target.spm",
        "vocab": "vocab.json",
        "config": "config.json",
        "generation_config": "generation_config.json",
    },
    "m2m100": {
        "spm": "sentencepiece.bpe.model",
        "config": "config.json",
        "generation_config": "generation_config.json",
        "special_tokens_map": "special_tokens_map.json",
        "vocab": "vocab.json",
        "added_tokens": "added_tokens.json",
    },
    "nllb": {
        "spm": "sentencepiece.bpe.model",
        "config": "config.json",
        "generation_config": "generation_config.json",
        "special_tokens_map": "special_tokens_map.json",
    },
}

REQUIRED_SIDE_FILES: Dict[str, Tuple[str, ...]] = {
    "marian": ("source_spm", "target_spm", "vocab", "config"),
    "m2m100": ("spm", "config", "special_tokens_map", "vocab"),
    "nllb": ("spm", "config", "special_tokens_map"),
}

GRAPHS: Dict[str, str] = {
    "encoder": "encoder_model.onnx",
    "decoder": "decoder_model.onnx",
    "decoder_with_past": "decoder_with_past_model.onnx",
}


class SkipRepo(Exception):
    """This repo will not get a registry entry, and why."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

#: Sent on every request. The Hub throttles hard on an unidentified agent.
USER_AGENT = "linguonnx-sync-registry/1.0 (+https://github.com/TigreGotico/linguonnx)"

HOST = "huggingface.co"

#: One kept-alive connection for the whole sync. A fresh TLS handshake per
#: request turns a 141-repo crawl into a twenty-minute one; reusing the socket
#: brings it under three. Reset to None on any error so the next call redials.
_CONNECTION = None


def _get(path: str, retries: int = 4) -> bytes:
    """GET an absolute path on the Hub, over a reused HTTPS connection."""
    global _CONNECTION
    last = None
    for attempt in range(retries):
        if attempt:
            time.sleep(2 ** attempt)  # back off; the Hub rate-limits bursts
        try:
            if _CONNECTION is None:
                _CONNECTION = http.client.HTTPSConnection(HOST, timeout=30)
            _CONNECTION.request("GET", path, headers={
                "User-Agent": USER_AGENT, "Accept": "*/*"})
            response = _CONNECTION.getresponse()
            body = response.read()
            if response.status == 404:
                raise FileNotFoundError(path)
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {path}")
            return body
        except FileNotFoundError:
            raise
        except Exception as err:
            # A half-consumed or reset socket cannot be reused.
            try:
                if _CONNECTION is not None:
                    _CONNECTION.close()
            finally:
                _CONNECTION = None
            last = err
    raise RuntimeError(f"GET {path} failed after {retries} tries: {last}")


def _get_json(path: str):
    return json.loads(_get(path))


def list_repos() -> List[dict]:
    return _get_json(f"{HF_API}?author={AUTHOR}&limit=1000")


def repo_detail(repo_id: str) -> dict:
    return _get_json(f"{HF_API}/{repo_id}?blobs=true")


def raw_file(repo_id: str, path: str) -> str:
    return _get(HF_RAW.format(repo=repo_id, path=path)).decode("utf-8")


# ---------------------------------------------------------------------------
# Shared parsing
# ---------------------------------------------------------------------------

_LICENSE_BODY_RE = re.compile(r"[Ll]icen[cs]e:?\**\s*[`\"]?([A-Za-z0-9.\-]+)")
_BASE_MODEL_RE = re.compile(r"Helsinki-NLP/([A-Za-z0-9._\-]+)")
_TARGET_TOKEN_RE = re.compile(r">>(\w+)<<")


def _license_of(detail: dict, readme: str) -> str:
    """Licence id, from card metadata first and the README body second."""
    raw = (detail.get("cardData") or {}).get("license")
    if not raw:
        match = _LICENSE_BODY_RE.search(readme)
        raw = match.group(1) if match else None
    if not raw:
        raise SkipRepo("no licence in cardData or README")
    key = str(raw).strip().lower()
    if key not in LICENSE_DISPLAY:
        raise SkipRepo(f"unrecognised licence {raw!r}")
    return LICENSE_DISPLAY[key]


def _files_of(detail: dict) -> Dict[str, int]:
    return {s["rfilename"]: int(s.get("size") or 0)
            for s in detail.get("siblings", [])}


def _resolve(files: Dict[str, int], prefix: str, name: str) -> Optional[str]:
    """``int8/vocab.json`` if the variant ships one, else the root copy.

    Several int8 subdirectories carry only the graphs, and share the root
    tokenizer files. Both layouts have to produce a working entry.
    """
    if prefix and (prefix + name) in files:
        return prefix + name
    return name if name in files else None


def _with_external_data(files: Dict[str, int], paths: Iterable[str]) -> List[str]:
    """``*.onnx_data`` blobs for the given graphs.

    They are never opened by name; ONNX Runtime finds them by relative path
    next to the graph, so they only have to be fetched, not keyed.
    """
    extra = []
    for path in paths:
        blob = path + "_data"
        if blob in files:
            extra.append(blob)
    return sorted(extra)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(files: Dict[str, int]) -> Optional[str]:
    """``"lid"``, ``"translate"`` or None for a repo we do not index."""
    if "labels.json" in files and any(f.endswith("_hash.py") for f in files):
        return "lid"
    if "encoder_model.onnx" in files:
        return "translate"
    return None


def _arch(detail: dict, files: Dict[str, int]) -> str:
    model_type = (detail.get("config") or {}).get("model_type", "")
    if model_type == "marian":
        return "marian"
    # M2M100 and NLLB share `model_type: m2m_100`; the tokenizer tells them
    # apart. M2M100 ships a vocab.json and reads ids from it; NLLB has none and
    # uses fairseq's sp_id + 1 offset instead.
    if "vocab.json" in files:
        return "m2m100"
    return "nllb"


# ---------------------------------------------------------------------------
# Translation entries
# ---------------------------------------------------------------------------

def _non_opus_marian_pair(repo_name: str, detail: dict
                          ) -> Tuple[List[str], Optional[str], Optional[str]]:
    """A bilingual Marian export that is not from the OPUS-MT project.

    ``mt-hitz-es-eu-onnx`` (HiTZ's Spanish -> Basque model) is a plain
    single-pair Marian with a properly filled card, so there is no group-model
    ambiguity and no prefix token. The pair still is not *taken* from the repo
    name: the name proposes it and ``cardData.language`` has to agree, as a
    set. A name and a card that disagree mean nobody actually knows which
    direction the weights run, and that is a skip.
    """
    card = detail.get("cardData") or {}
    declared = [str(code) for code in (card.get("language") or [])]
    if len(declared) != 2:
        raise SkipRepo(
            f"cannot read a language pair from the repo name {repo_name!r} and "
            f"the card declares {len(declared)} languages")
    stem = repo_name[:-len("-onnx")] if repo_name.endswith("-onnx") else repo_name
    parts = stem.split("-")
    if len(parts) < 2:
        raise SkipRepo(f"cannot read a language pair from {repo_name!r}")
    src, tgt = parts[-2], parts[-1]
    if {src, tgt} != set(declared):
        raise SkipRepo(
            f"repo name {repo_name!r} says {src}->{tgt} but the card declares "
            f"{declared}; refusing to guess the direction")
    base = card.get("base_model")
    if not base:
        raise SkipRepo("no base_model in the card to verify the pair against")
    return [src, tgt], str(base), None


def _marian_pair(repo_name: str, readme: str, detail: dict
                 ) -> Tuple[List[str], Optional[str], Optional[str]]:
    """``(pair, base_model, target_token)`` for one Marian export.

    The repo name gives the intended pair; the base model named in the card
    says whether the export is actually that pair or a slice of a group model.
    Disagreement on the *target* side means a prefix token is mandatory.
    """
    match = re.fullmatch(r"opus-mt-([a-z]{2,3})-([a-z]{2,3})-onnx", repo_name)
    if not match:
        return _non_opus_marian_pair(repo_name, detail)
    src, tgt = match.group(1), match.group(2)

    base_match = _BASE_MODEL_RE.search(readme)
    if not base_match:
        raise SkipRepo("no Helsinki-NLP base model named in the card")
    base = base_match.group(1)

    # Strip the `tc-big` marker: a bigger model for the very same pair.
    stem = base[len("opus-mt-"):] if base.startswith("opus-mt-") else base
    stem = stem[len("tc-big-"):] if stem.startswith("tc-big-") else stem
    parts = stem.split("-")
    if len(parts) != 2:
        raise SkipRepo(f"cannot parse base model {base!r}")
    base_src, base_tgt = parts

    token = None
    if base_tgt != tgt:
        # Multi-target group model: the target is chosen by a prefix token, and
        # is wrong by default. Only the card can tell us which token.
        tokens = sorted(set(_TARGET_TOKEN_RE.findall(readme)))
        if len(tokens) != 1:
            raise SkipRepo(
                f"base model {base!r} is multi-target for {src}->{tgt} but the "
                f"card shows {len(tokens)} target tokens; refusing to guess")
        token = f">>{tokens[0]}<<"
    # base_src != src is a multi-*source* group model (e.g. ROMANCE-en). The
    # source is inferred from the text, so the pair claim still holds as-is.
    return [src, tgt], base, token


def _assert_not_narrow_finetune(detail: dict, model_id: str) -> None:
    """Refuse to read a covering set off a *bilingual fine-tune*.

    A fine-tune keeps the whole base tokenizer, so ``special_tokens_map.json``
    still lists all 100 (or 202) languages long after the weights stopped being
    able to serve them. Masakhane's ``m2m100_418M_bbj_fr_rel_news_ft`` is a
    French <-> Ghomala' model wearing M2M100's full token inventory; trusting
    the tokenizer would publish a 100-language claim for a two-language model
    and let the router send anything through it.

    The tell is ``cardData.language``: general models declare
    ``["multilingual"]``, fine-tunes name their two or three languages. Those
    need a hand-verified :data:`BILINGUAL_FINETUNES` entry stating the pair in
    the model's own codes, because which token a fine-tune reused for a
    language the base never had is not recoverable from the export.
    """
    languages = (detail.get("cardData") or {}).get("language") or []
    if not isinstance(languages, list) or "multilingual" in languages:
        return
    if 0 < len(languages) <= 3:
        raise SkipRepo(
            f"bilingual fine-tune ({'/'.join(languages)}) of a multilingual "
            f"base: its tokenizer still lists every base language, so coverage "
            f"cannot be derived. Add a verified BILINGUAL_FINETUNES entry for "
            f"{model_id!r} to register it")


def _multilingual_languages(repo_id: str, files: Dict[str, int]) -> List[str]:
    """The model's covered set, from its own tokenizer metadata.

    ``additional_special_tokens`` in ``special_tokens_map.json`` is the list
    the model was actually built with - M2M100's ``__pt__``, NLLB's
    ``por_Latn``. Nothing is added to or removed from it here.
    """
    if "special_tokens_map.json" not in files:
        raise SkipRepo("no special_tokens_map.json to read languages from")
    data = json.loads(raw_file(repo_id, "special_tokens_map.json"))
    tokens = data.get("additional_special_tokens") or []
    codes = []
    for token in tokens:
        code = token[2:-2] if token.startswith("__") and token.endswith("__") \
            else token
        if code and code != "<mask>":
            codes.append(code)
    if not codes:
        raise SkipRepo("special_tokens_map.json lists no language tokens")
    return sorted(codes)


def translate_entries(repo_id: str, detail: dict, readme: str) -> Dict[str, dict]:
    """Every precision variant of one translation repo."""
    name = repo_id.split("/")[1]
    model_id = MODEL_ID_OVERRIDES.get(
        name, name[:-len("-onnx")] if name.endswith("-onnx") else name)
    files = _files_of(detail)
    arch = _arch(detail, files)
    license_id = _license_of(detail, readme)

    shared: Dict[str, object] = {"arch": arch}
    if arch == "marian":
        pair, base, token = _marian_pair(name, readme, detail)
        shared["pair"] = pair
        shared["base_model"] = base if "/" in base else f"Helsinki-NLP/{base}"
        if token:
            shared["target_token"] = token
    elif model_id in BILINGUAL_FINETUNES:
        shared.update(BILINGUAL_FINETUNES[model_id])
    else:
        _assert_not_narrow_finetune(detail, model_id)
        shared["languages"] = _multilingual_languages(repo_id, files)

    entries: Dict[str, dict] = {}
    for suffix, prefix in (("", ""), ("-int8", "int8/")):
        if prefix and (prefix + GRAPHS["encoder"]) not in files:
            continue
        graphs = {}
        for key, filename in GRAPHS.items():
            path = _resolve(files, prefix, filename)
            if path is None:
                raise SkipRepo(f"missing graph {filename}")
            graphs[key] = path

        side: Dict[str, str] = {}
        for key, filename in SIDE_FILES[arch].items():
            path = _resolve(files, prefix, filename)
            if path is not None:
                side[key] = path
        missing = [k for k in REQUIRED_SIDE_FILES[arch] if k not in side]
        if missing:
            raise SkipRepo(f"missing required side files: {', '.join(missing)}")

        extra = _with_external_data(files, graphs.values())
        size = sum(files.get(p, 0)
                   for p in list(graphs.values()) + list(side.values()) + extra)

        entry = {
            "model_id": model_id + suffix,
            "hf_repo": repo_id,
            "arch": arch,
            "graphs": graphs,
            "side_files": side,
            "extra_files": extra,
        }
        entry.update({k: v for k, v in shared.items() if k != "arch"})
        entry["license"] = license_id
        entry["license_tier"] = LICENSE_TIERS[license_id]
        entry["size_mb"] = max(1, round(size / 1e6))
        entry["precision"] = "int8" if prefix else "fp32"
        entries[entry["model_id"]] = entry
    return entries


# ---------------------------------------------------------------------------
# LID entries
# ---------------------------------------------------------------------------

def lid_entries(repo_id: str, detail: dict, readme: str) -> Dict[str, dict]:
    name = repo_id.split("/")[1]
    model_id = MODEL_ID_OVERRIDES.get(
        name, name[:-len("-onnx")] if name.endswith("-onnx") else name)
    files = _files_of(detail)
    license_id = _license_of(detail, readme)

    config = json.loads(raw_file(repo_id, "config.json"))
    labels = json.loads(raw_file(repo_id, "labels.json"))
    num_labels = len(labels) if isinstance(labels, (list, dict)) else None
    if not num_labels:
        raise SkipRepo("labels.json is empty")
    # lid.176 is hierarchical-softmax; everything else is a plain softmax head.
    loss = str(config.get("loss", "softmax")).lower()
    if loss not in ("softmax", "hs"):
        raise SkipRepo(f"unknown loss {loss!r}")

    side = {}
    for key, filename in (("vocab", "vocab.txt"), ("labels", "labels.json"),
                          ("config", "config.json")):
        if filename not in files:
            raise SkipRepo(f"missing {filename}")
        side[key] = filename
    # A hierarchical-softmax model (lid.176) cannot decode without its tree:
    # the ONNX graph emits scores over internal nodes, not labels. Required
    # exactly when loss == "hs", and refused when it is missing, because the
    # entry would download cleanly and then fail at the first prediction.
    if loss == "hs":
        if "hs_tree.json" not in files:
            raise SkipRepo("loss is 'hs' but the repo has no hs_tree.json")
        side["hs_tree"] = "hs_tree.json"
    elif "hs_tree.json" in files:
        raise SkipRepo("repo ships hs_tree.json but declares a softmax loss")

    entries = {}
    for suffix, candidates in (
            ("", [f for f in files if f.endswith(".onnx")
                  and ".int8." not in f]),
            ("-int8", [f for f in files if f.endswith(".onnx")
                       and ".int8." in f])):
        if not candidates:
            continue
        onnx_file = sorted(candidates)[0]
        entries[model_id + suffix] = {
            "model_id": model_id + suffix,
            "hf_repo": repo_id,
            "onnx_file": onnx_file,
            "side_files": dict(side),
            "license": license_id,
            "engine": "fasttext-onnx",
            "loss": loss,
            "num_labels": num_labels,
            "size_mb": max(1, round(
                (files.get(onnx_file, 0)
                 + sum(files.get(f, 0) for f in side.values())) / 1e6)),
            "precision": "int8" if suffix else "fp32",
        }
    return entries


# ---------------------------------------------------------------------------
# Merge + write
# ---------------------------------------------------------------------------

def merge_preserving(existing: Dict[str, dict],
                     generated: Dict[str, dict]) -> Dict[str, dict]:
    """Generated values win; hand-authored *extra* keys survive.

    A curated ``notes``, a pinned override, anything a human added to an entry
    and the API knows nothing about is carried across. Only keys this script
    produces are allowed to change, so a re-run never silently discards
    editorial work.
    """
    merged = {}
    for model_id, entry in generated.items():
        previous = existing.get(model_id, {})
        kept = {k: v for k, v in previous.items() if k not in entry}
        combined = dict(entry)
        combined.update(kept)
        merged[model_id] = combined
    return merged


def _sorted_json(registry: Dict[str, dict]) -> str:
    """Stable text: entries sorted by id, keys sorted inside each entry.

    Re-running on an unchanged Hub must produce a byte-identical file, or
    ``--check`` is worthless and every sync makes a noisy diff.
    """
    ordered = {model_id: dict(sorted(registry[model_id].items()))
               for model_id in sorted(registry)}
    return json.dumps(ordered, indent=2, ensure_ascii=False,
                      sort_keys=False) + "\n"


def build() -> Tuple[Dict[str, dict], Dict[str, dict], List[Tuple[str, str]]]:
    translate: Dict[str, dict] = {}
    lid: Dict[str, dict] = {}
    skipped: List[Tuple[str, str]] = []

    repos = list_repos()
    for index, summary in enumerate(sorted(repos, key=lambda r: r["id"]), 1):
        repo_id = summary["id"]
        print(f"[{index}/{len(repos)}] {repo_id}", file=sys.stderr)
        try:
            detail = repo_detail(repo_id)
        except Exception as err:
            skipped.append((repo_id, f"detail fetch failed: {err}"))
            continue
        files = _files_of(detail)
        kind = classify(files)
        if kind is None:
            continue
        try:
            readme = raw_file(repo_id, "README.md") if "README.md" in files else ""
            if kind == "translate":
                translate.update(translate_entries(repo_id, detail, readme))
            else:
                lid.update(lid_entries(repo_id, detail, readme))
        except SkipRepo as err:
            skipped.append((repo_id, str(err)))
        except Exception as err:
            skipped.append((repo_id, f"{type(err).__name__}: {err}"))
    return translate, lid, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the committed registry "
                             "differs from what the Hub now says")
    args = parser.parse_args(argv)

    translate, lid, skipped = build()
    if not translate or not lid:
        print("refusing to write an empty registry; is the Hub reachable?",
              file=sys.stderr)
        return 2

    drift = False
    for kind, generated in (("translate", translate), ("lid", lid)):
        path = INDEX_DIR / f"{kind}.json"
        existing = json.loads(path.read_text(encoding="utf-8")) \
            if path.exists() else {}
        text = _sorted_json(merge_preserving(existing, generated))
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if text == current:
            print(f"{kind}.json: up to date ({len(generated)} entries)")
            continue
        drift = True
        added = sorted(set(generated) - set(existing))
        removed = sorted(set(existing) - set(generated))
        if args.check:
            print(f"{kind}.json: DRIFT - {len(added)} new, {len(removed)} "
                  f"stale, content differs", file=sys.stderr)
            for model_id in added:
                print(f"  + {model_id}", file=sys.stderr)
            for model_id in removed:
                print(f"  - {model_id}", file=sys.stderr)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{kind}.json: wrote {len(generated)} entries "
                  f"(+{len(added)} new, -{len(removed)} stale)")

    for repo_id, reason in skipped:
        print(f"skipped {repo_id}: {reason}", file=sys.stderr)

    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
