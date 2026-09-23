"""linguonnx#42-adjacent regression: `sweep_empty_output.py` must never feed a
model text from a language it was not asked to translate.

The script used to fall back to a hardcoded English sentence
(`FALLBACK_SAMPLE`) for any source language missing from `SAMPLES`, then
judge the translation as if the input had been correct. That produced
verdicts nothing verified: a bogus DEFECT if the model choked on the
mismatched input, a bogus OK if it happened to cope. The same pattern in a
separate ad-hoc harness affected 32/183 int8 models and, once traced back,
invalidated every one of the 7 recorded rows it had touched.

The fix: a source language with no confirmed sample is a loud, explicit
"skipped-no-sample" verdict, and the model is never run for it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "linguonnx"
    / "scripts"
    / "sweep_empty_output.py"
)


def _load_module():
    """Import sweep_empty_output.py fresh, isolated from any prior import."""
    spec = importlib.util.spec_from_file_location(
        "sweep_empty_output_under_test", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def sweep():
    module = _load_module()
    yield module
    sys.modules.pop("sweep_empty_output_under_test", None)


def test_unknown_source_language_is_skipped_not_substituted(sweep, monkeypatch):
    """A source language absent from SAMPLES must yield 'skipped-no-sample'
    and must NOT run the model with a different language's text."""

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError(
            "TranslationModel must not be instantiated when there is no "
            "confirmed sample for the source language - that would silently "
            "feed the model text from the wrong language."
        )

    monkeypatch.setattr(sweep, "TranslationModel", _must_not_be_called)

    # A source language guaranteed not to be in SAMPLES (and not a real tag
    # linguonnx would ever register), so the test does not depend on which
    # confirmed samples happen to exist.
    entry = {"pair": ["zzz-not-a-real-language", "en"]}
    result = sweep.check_model("fake-model-id", entry)

    assert result["status"] == "skipped-no-sample"
    assert result["src"] == "zzz-not-a-real-language"


def test_known_source_language_is_not_skipped(sweep):
    """Sanity check: a language that IS in SAMPLES must not be reported as
    skipped by _sample_for (guards against the fixture itself being wrong)."""
    assert sweep._sample_for("en") is not None
    assert sweep._sample_for("zzz-not-a-real-language") is None
