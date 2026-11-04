"""A model whose processing needs an uninstalled extra is never routed to.

linguonnx#33: without ``linguonnx[opennmt]`` the Proxecto Nós models stayed in
the routing graph, were chosen, downloaded, and then raised ``ImportError`` in
the middle of a request. Every test here runs offline against the committed
registry. An uninstalled package is simulated by making
``importlib.metadata.version`` report it absent.
"""

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate import load_translator
from linguonnx.translate import models as models_module
from linguonnx.translate.graph import (NoRouteError, _registry_runnability,
                                       entry_runnability,
                                       missing_extra_packages)

ROOT = Path(__file__).resolve().parent.parent
NOS_EN_GL = "nos-coda_iacobus-en-gl-int8"


def _optional_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["optional-dependencies"]


@pytest.fixture
def without(monkeypatch):
    """Make the named distributions look uninstalled; clear the runnability memo."""
    real_version = importlib.metadata.version

    def hide(*names):
        hidden = {name.lower() for name in names}

        def version(name):
            if name.lower() in hidden:
                raise importlib.metadata.PackageNotFoundError(name)
            return real_version(name)

        monkeypatch.setattr(importlib.metadata, "version", version)
        _registry_runnability.cache_clear()

    yield hide
    _registry_runnability.cache_clear()


def test_every_extra_requirement_is_installed_here():
    for extra in ("opennmt", "indic", "fast-tokenizers"):
        assert missing_extra_packages(extra) == []


def test_a_hidden_package_is_reported_by_name(without):
    without("subword-nmt")
    assert missing_extra_packages("opennmt") == ["subword-nmt"]
    # `sacremoses` is in both `opennmt` and `indic`; hiding only subword-nmt
    # must not touch `indic`.
    assert missing_extra_packages("indic") == []


def test_extra_names_are_matched_after_normalisation(without):
    without("tokenizers")
    assert missing_extra_packages("fast_tokenizers") == ["tokenizers"]


def test_the_registry_names_the_extra_each_architecture_needs():
    expected = {"opennmt-bpe": "opennmt", "indictrans2": "indic",
                "pegasus-fast": "fast-tokenizers", "causal-lm": "fast-tokenizers",
                "rules": "rules"}
    for model_id, entry in list_models(kind="translate").items():
        assert entry.get("extra") == expected.get(entry["arch"]), model_id


def test_every_extra_the_registry_names_exists_in_pyproject():
    declared = set(_optional_dependencies())
    named = {entry["extra"] for entry in list_models(kind="translate").values()
             if entry.get("extra")}
    assert named <= declared, named - declared


def test_without_the_extra_the_entry_is_unrunnable_and_says_what_to_install(without):
    without("sacremoses")
    runnable, reason = entry_runnability(list_models(kind="translate")[NOS_EN_GL])
    assert not runnable
    assert reason == "sacremoses not installed; install linguonnx[opennmt]"


def test_without_the_extra_the_router_never_offers_the_model(without):
    offered = {model_id for route in load_translator(include_noncommercial=False)
               .routes("en", "gl", limit=50) for model_id in route.model_ids}
    assert NOS_EN_GL in offered
    without("sacremoses")
    offered = {model_id for route in load_translator(include_noncommercial=False)
               .routes("en", "gl", limit=50) for model_id in route.model_ids}
    assert not any(model_id.startswith("nos-") for model_id in offered)
    assert offered, "en->gl must still route through the other models"


def test_without_the_extra_a_pinned_model_refuses_before_downloading(
        without, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("files were fetched for a model that cannot run")

    monkeypatch.setattr(models_module, "ensure_model_files", no_download)
    without("subword-nmt")
    translator = load_translator(include_noncommercial=False)
    with pytest.raises(NoRouteError, match=r"install linguonnx\[opennmt\]"):
        translator.translate("hello world", src="en", tgt="gl", model=NOS_EN_GL)


def test_an_indic_model_follows_the_same_rule(without):
    entries = list_models(kind="translate")
    model_id = next(m for m, e in entries.items() if e["arch"] == "indictrans2")
    assert entry_runnability(entries[model_id]) == (True, None)
    without("indic-nlp-library-itt")
    assert entry_runnability(entries[model_id]) == (
        False, "indic-nlp-library-itt not installed; install linguonnx[indic]")
