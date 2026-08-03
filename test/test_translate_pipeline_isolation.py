"""IndicTrans2's per-call state must not leak across models or architectures.

``linguonnx#44``: a real server lazy-loads whatever models a client asks for,
into one process, in whatever order requests happen to arrive. Every
architecture's :class:`~linguonnx.translate.preprocess.Pipeline` is a
process-wide singleton (``_PIPELINES`` in
:mod:`linguonnx.translate.preprocess`) - one instance shared by every model of
that architecture, and by every ``Translator`` in the process. IndicTrans2 is
the one architecture with real per-call state to leak: :meth:`encode` builds
an ``entity_map`` (URLs, emails and numerals swapped for ``<IDn>`` placeholders)
that :meth:`decode` needs back to restore them. Park that map anywhere shared -
a module-level variable, an attribute on the singleton pipeline or on the
shared :class:`IndicProcessor` - and it is is a race the moment two calls
interleave: model B's ``encode`` overwrites the map before model A's
``decode`` reads it, and model A's placeholders come back untouched in the
output instead of the original text. Nothing raises; the translation is just
wrong.

The regression test for this (``linguonnx#44``, ``test_translate_integration.py``)
downloads three real ONNX graphs and needs ``optimum``+``transformers`` to
build a reference - reasonably, since it is checking real translations
end-to-end. But that guard sits behind
``pytest.importorskip("optimum.onnxruntime")``, so in every environment where
that was never installed (which was every CI run - see the `test` extra
history in `pyproject.toml`) it was skipped outright, not run and passed. A
broken build could ship behind it and nothing would say so.

This module checks the actual mechanism - where the entity map is stored, and
whether it survives another model's call in between - without a network, a
real model, or ``optimum``/``transformers``. It needs ``linguonnx[indic]``
only, to exercise the real ``IndicProcessor`` rather than a stub of it: a stub
that always agrees with the code under test would not catch a regression that
changes both together.
"""

import pytest

pytest.importorskip("regex")
pytest.importorskip("sacremoses")
pytest.importorskip("indicnlp")

from linguonnx.translate.preprocess import pipeline_for


class _FakeTokenizer:
    """Just enough tokenizer for the pipelines to call, no vocabulary at all.

    IndicTrans2 and the SentencePiece architectures each want *ids* back from
    ``encode`` and *text* back from ``decode``; nothing here checks that a
    real model would produce the same ids, only that whichever pipeline calls
    it gets the string it put in, so entity-map corruption is not masked by
    a lossy fake tokenizer disagreeing with itself.
    """

    def encode(self, text, *_args):
        return [text]

    def decode(self, ids):
        return ids[0]

    def lang_id(self, code):
        return 1


class _FakeIndicTrans2Model:
    """Enough of ``TranslationModel`` for ``IndicTrans2Pipeline`` to run."""

    def __init__(self, name):
        self.model_id = name
        self.tokenizer = _FakeTokenizer()

    def native_code(self, tag):
        return tag


class _FakeSpmModel:
    """Enough of ``TranslationModel`` for ``SpmLangTokenPipeline`` to run."""

    def __init__(self, name):
        self.model_id = name
        self.tokenizer = _FakeTokenizer()

    def native_code(self, tag):
        return tag


def test_the_indic_pipeline_is_one_shared_instance():
    """Guards the assumption the rest of this module tests against: if this
    ever becomes "one instance per model", the leak this file guards against
    cannot happen, and every test below would be exercising a fake problem."""
    assert pipeline_for("indictrans2") is pipeline_for("indictrans2")


def test_two_indictrans2_models_do_not_share_an_entity_map():
    """Two IndicTrans2 ``TranslationModel``s, same shared pipeline, interleaved.

    model_a encodes a sentence with an email in it; model_b encodes a
    different sentence with a phone number in it, *before* model_a is
    decoded. If the map were parked anywhere but the model object itself,
    model_b's encode would have already overwritten it by the time model_a
    decodes, and model_a's output would come back with the email's
    placeholder token still in it instead of the address.
    """
    pipeline = pipeline_for("indictrans2")
    model_a = _FakeIndicTrans2Model("indictrans2-a")
    model_b = _FakeIndicTrans2Model("indictrans2-b")

    ids_a = pipeline.encode(model_a, "reach me at jane@example.com please",
                            "eng_Latn", "hin_Deva")
    ids_b = pipeline.encode(model_b, "call on 12/03/2024 please",
                            "eng_Latn", "hin_Deva")

    # model_a's own map must have survived model_b's encode call untouched.
    assert model_a._indic_entity_map != model_b._indic_entity_map
    assert any("example.com" in v for v in model_a._indic_entity_map.values())
    assert any("12/03/2024" in v for v in model_b._indic_entity_map.values())

    out_a = pipeline.decode(model_a, ids_a, "eng_Latn", "hin_Deva")
    out_b = pipeline.decode(model_b, ids_b, "eng_Latn", "hin_Deva")

    assert "jane@example.com" in out_a, out_a
    assert "12/03/2024" in out_b, out_b
    # Cross-contamination would show up as the other call's original text.
    assert "12/03/2024" not in out_a
    assert "jane@example.com" not in out_b


def test_a_spm_architecture_between_two_indictrans2_calls_does_not_corrupt_them():
    """The exact shape of linguonnx#44: load another architecture in between.

    A real process loads m2m100/nllb and IndicTrans2 side by side. Encode
    IndicTrans2 model_a, run an unrelated m2m100 call, then decode model_a -
    the entity map must not have moved just because a different architecture's
    singleton pipeline ran in between.
    """
    indic_pipeline = pipeline_for("indictrans2")
    spm_pipeline = pipeline_for("m2m100")

    model_a = _FakeIndicTrans2Model("indictrans2-a")
    ids_a = indic_pipeline.encode(model_a, "email me at a@b.com now",
                                  "eng_Latn", "hin_Deva")

    other = _FakeSpmModel("m2m100-418M-int8")
    spm_pipeline.encode(other, "hello world", "en", "fr")

    out_a = indic_pipeline.decode(model_a, ids_a, "eng_Latn", "hin_Deva")
    assert "a@b.com" in out_a, out_a
