"""A Marian pair whose one target language has several varieties.

`opus-mt-en-ar` is named as a pair, so the registry built it as a pair and
recorded no `>>xxx<<` target token. Its vocabulary carries four: `>>ara<<`,
`>>ara_Latn<<`, `>>arq<<` and `>>arz<<`. Without one the model is not told
which variety to write, and it does not fail - it answers.

Measured on `TigreGotico/opus-mt-en-ar-onnx`, int8, 2026-09-28:

    no token    "cancel the alarm" -> "هذا"          ("this")
    no token    "tell me a joke"   -> "تُخبرُني a يَمْزحُ"  (a gloss, English "a" left in)
    >>ara<<     "cancel the alarm" -> "ألغِ المنبه"
    >>ara<<     "tell me a joke"   -> "قل لي نكتة"

So the token is recorded in the registry entry, where
`TranslationModel.translate` already reads it before the pair check.
"""
from linguonnx import model_manager


def _entries():
    return model_manager.list_models(kind="translate")


def test_the_en_ar_entries_record_their_target_token():
    entries = _entries()
    for model_id in ("opus-mt-en-ar", "opus-mt-en-ar-int8"):
        assert entries[model_id].get("target_token") == ">>ara<<", (
            f"{model_id} is a pair entry over a multi-variety Arabic model; "
            f"without >>ara<< it answers in a gloss")


def test_a_recorded_target_token_is_one_the_model_could_carry():
    """Every recorded token keeps the `>>xxx<<` shape the vocabulary uses."""
    for model_id, entry in _entries().items():
        token = entry.get("target_token")
        if token is None:
            continue
        assert token.startswith(">>") and token.endswith("<<"), (
            f"{model_id} records {token!r}, which is not a Marian target "
            f"token; a plain word is a word in the source text")


def test_the_pair_entries_that_need_a_token_are_the_ones_that_have_one():
    """A control on the shape of the claim: a recorded token names the
    entry's own target side, never the source side."""
    for model_id, entry in _entries().items():
        token = entry.get("target_token")
        pair = entry.get("pair")
        if token is None or not pair:
            continue
        from linguonnx.detect.labels import to_bcp47
        raw = token[2:-2]
        # `to_bcp47` returns the tag unchanged when it cannot map it, so it
        # needs no fallback here.
        tag = to_bcp47(raw)
        assert tag == pair[1], (
            f"{model_id} records {token!r} for the pair {pair}; the token "
            f"must select the target, not the source")
