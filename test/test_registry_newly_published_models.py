"""Coverage for the DSFSI Northern Sotho pair, and an honest record of the
published repos that cannot be registered.

``m2m100_418m-eng-nso`` / ``m2m100_418m-nso-eng``: DSFSI fine-tunes of
`facebook/m2m100_418M`, same shape as the Masakhane fine-tunes already in
``BILINGUAL_FINETUNES``: the card declares 2 languages but the export's
tokenizer still carries the base model's full 100-language
``special_tokens_map.json``/``added_tokens.json`` inventory. Northern
Sotho/Sepedi's own token is spelled ``__ns__`` (m2m100 uses the ISO 639-1
code ``ns``), not ``__nso__`` - ``to_bcp47("ns")`` cannot resolve it (it is
not a registered IANA subtag on its own), so routing uses the BCP-47/ISO
639-3 macrolanguage code ``nso`` and ``native_codes`` carries the model's own
``ns`` spelling through to the tokenizer, the same fix shape PR #56 used for
IndicTrans2's FLORES tags and the AINA bilingual entries - not a
``normalize_tag`` alias, which would apply the substitution to every caller
of ``ns``/``nso`` project-wide rather than just this one model.

``arat5-arabic-dialects-translation`` (T5, no MADLAD-shaped ``spiece.model``)
and ``vinai-translate-vi2en-v2`` (mBART) are architectures this library
implements no loader for - `scripts/sync_registry.py` now records both with
that reason rather than the mBART one silently mislabelling itself "nllb"
(both share NLLB's lone ``sentencepiece.bpe.model`` +
``special_tokens_map.json`` shape, which is what let the mislabel happen
unnoticed before this fix).

``arabic-MARBERT-dialect-identification-city`` and ``afrolid_1.5`` are
transformer (MARBERT/XLM-R) sequence classifiers - fastText-hash is the only
LID engine ``linguonnx.detect`` implements, and the MARBERT model's own
output (Arabic city names) would not fit ``lid.json``'s BCP-47 contract even
with such an engine. Both are recorded in ``skipped.json``, not registered.
"""

import json
from pathlib import Path

import pytest

from linguonnx.model_manager import list_models
from linguonnx.translate.models import TranslationModel

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATE = list_models("translate")
SKIPPED = json.loads(
    (REPO_ROOT / "linguonnx" / "model_index" / "skipped.json")
    .read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Registered: the DSFSI Northern Sotho pair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", [
    "m2m100_418m-eng-nso", "m2m100_418m-eng-nso-int8",
    "m2m100_418m-nso-eng", "m2m100_418m-nso-eng-int8",
])
def test_the_dsfsi_nso_entries_are_registered(model_id):
    assert model_id in TRANSLATE
    entry = TRANSLATE[model_id]
    assert entry["arch"] == "m2m100"
    assert set(entry["pair"]) == {"en", "nso"}


@pytest.mark.parametrize("model_id", [
    "m2m100_418m-eng-nso", "m2m100_418m-nso-eng",
])
def test_nso_routes_to_the_models_own_ns_token(model_id):
    """`native_code('nso')` must answer `'ns'`, or `lang_id` raises on every
    call - the tokenizer's vocabulary has no `nso` token, only `ns`."""
    entry = TRANSLATE[model_id]
    model = TranslationModel(model_id, entry=entry)
    assert model.native_code("nso") == "ns"


def test_nso_is_not_bent_into_a_global_alias():
    """The fix is `native_codes` on these two entries, not a `normalize_tag`
    rewrite - `to_bcp47('ns')` must not resolve to `'nso'` project-wide. `ns`
    is not a registered IANA subtag on its own (`to_bcp47` emits it as-is,
    unresolved); the `ns` -> `nso` correspondence is scoped to these two
    models' `native_codes`, not a global alias every other caller inherits."""
    from linguonnx.detect.labels import to_bcp47

    assert to_bcp47("ns") != "nso"


def test_the_registry_still_has_bilingual_finetune_entries():
    """Guards the tests above from passing because they found nothing."""
    assert any(e.get("pair") for e in TRANSLATE.values())


# ---------------------------------------------------------------------------
# Skipped, honestly: architectures this library does not implement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hf_repo,expect_substring", [
    ("TigreGotico/arat5-arabic-dialects-translation-onnx", "spiece.model"),
    ("TigreGotico/vinai-translate-vi2en-v2-onnx", "mbart"),
    ("TigreGotico/arabic-MARBERT-dialect-identification-city-onnx", "MARBERT"),
    ("TigreGotico/afrolid_1.5-onnx", "XLM-R"),
])
def test_unregisterable_repos_are_recorded_with_their_real_reason(
        hf_repo, expect_substring):
    reason = SKIPPED.get(hf_repo)
    assert reason, f"{hf_repo} is missing from skipped.json entirely"
    assert expect_substring in reason


def test_arat5_and_vinai_are_not_claimed_by_the_registry():
    """T5-non-MADLAD and mBART are not architectures `linguonnx.translate`
    implements; neither model may appear under any model_id."""
    claimed = {entry["hf_repo"] for entry in TRANSLATE.values()}
    assert "TigreGotico/arat5-arabic-dialects-translation-onnx" not in claimed
    assert "TigreGotico/vinai-translate-vi2en-v2-onnx" not in claimed


def test_afrolid_upstream_defect_is_documented_not_papered_over():
    """The Yoruba misclassification is confirmed upstream (see the module
    docstring and skipped.json), so the caveat must survive in the skip
    reason rather than being silently dropped - it is what tells a future
    maintainer who *does* add a transformer-classifier engine that this
    model still needs a `quality` entry, not a plain registration."""
    reason = SKIPPED["TigreGotico/afrolid_1.5-onnx"]
    assert "yor" in reason.lower() or "yoruba" in reason.lower()
    assert "upstream" in reason.lower()
