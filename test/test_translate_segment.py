"""Sentence cutting, and how ``translate()`` puts the sentences back."""

import pytest

from linguonnx.limits import OutputTruncatedError
from linguonnx.model_manager import list_models
from linguonnx.translate.graph import _default_script
from linguonnx.translate.models import TranslationModel
from linguonnx.translate.segment import split_sentences

CUTS = [
    ("One sentence", [("One sentence", "")]),
    ("One sentence.", [("One sentence.", "")]),
    ("One sentence. ", [("One sentence. ", "")]),
    ("Hi there. Bye now.", [("Hi there.", " "), ("Bye now.", "")]),
    ("Really? Yes! Fine.", [("Really?", " "), ("Yes!", " "), ("Fine.", "")]),
    ("Wide.   Gap.\tTab.", [("Wide.", "   "), ("Gap.", "\t"), ("Tab.", "")]),
    ("Wait... what?! Ok.", [("Wait...", " "), ("what?!", " "), ("Ok.", "")]),
    ('He said "Go." Then left.', [('He said "Go."', " "), ("Then left.", "")]),
    ("Use e.g. this one.", [("Use e.g. this one.", "")]),
    ("Dr. Silva is in.", [("Dr. Silva is in.", "")]),
    ("Pi is 3.14 today.", [("Pi is 3.14 today.", "")]),
    ("see example.com now", [("see example.com now", "")]),
    ("1. Buy milk. Then go.", [("1. Buy milk.", " "), ("Then go.", "")]),
    ("... and then.", [("... and then.", "")]),
    ("今天很好。明天下雨。", [("今天很好。", ""), ("明天下雨。", "")]),
    ("元気？はい！", [("元気？", ""), ("はい！", "")]),
    ("「行く。」そうだ。", [("「行く。」", ""), ("そうだ。", "")]),
    ("今天很好。 明天下雨。", [("今天很好。", " "), ("明天下雨。", "")]),
    ("كيف حالك؟ أنا بخير.", [("كيف حالك؟", " "), ("أنا بخير.", "")]),
    ("آپ کیسے ہیں۔ میں ٹھیک ہوں۔", [("آپ کیسے ہیں۔", " "), ("میں ٹھیک ہوں۔", "")]),
    ("यह अच्छा है। वह बुरा है।", [("यह अच्छा है।", " "), ("वह बुरा है।", "")]),
]


@pytest.mark.parametrize("line, expected", CUTS)
def test_split_sentences(line, expected):
    assert split_sentences(line) == expected


@pytest.mark.parametrize("line, _", CUTS)
def test_the_parts_join_back_into_the_line(line, _):
    assert "".join(s + gap for s, gap in split_sentences(line)) == line


@pytest.fixture
def model(monkeypatch):
    """A real model object whose sentence step only records what it got."""
    seen = []

    def sentence(self, text, src, tgt, config, target_token):
        seen.append(text)
        return f"<{text.strip()}>"

    monkeypatch.setattr(TranslationModel, "_translate_sentence", sentence)
    built = TranslationModel("opus-mt-pt-en-int8")
    built.seen = seen
    return built


@pytest.mark.parametrize("text, tgt, expected", [
    ("Olá.", "en", "<Olá.>"),
    ("Um dois. Três.", "en", "<Um dois.> <Três.>"),
    ("Um dois.   Três.", "en", "<Um dois.>   <Três.>"),
    ("Um dois.\tTrês.", "en", "<Um dois.>\t<Três.>"),
    ("Um dois. Três.", "ja", "<Um dois.><Três.>"),
    ("Um dois. Três.", "zh-Hant", "<Um dois.><Três.>"),
    ("Um dois. Três.", "yue", "<Um dois.><Três.>"),
    ("Um dois. Três.", "th", "<Um dois.><Três.>"),
    ("Um dois. Três.", "ko", "<Um dois.> <Três.>"),
    ("今天很好。明天下雨。", "en", "<今天很好。> <明天下雨。>"),
    ("Um.\nDois.", "en", "<Um.>\n<Dois.>"),
    ("Um.\n\nDois.\n", "en", "<Um.>\n\n<Dois.>\n"),
    ("\nUm.", "en", "\n<Um.>"),
    ("Um.\n   \nDois.", "en", "<Um.>\n   \n<Dois.>"),
    ("Um dois. \u200b", "en", "<Um dois.>"),
    ("Um dois. \u200b Três.", "en", "<Um dois.> <\u200b Três.>"),
])
def test_translate_cuts_and_rejoins(model, text, tgt, expected):
    assert model.translate(text, "pt", tgt) == expected


def test_one_sentence_reaches_the_model_unchanged(model):
    model.translate("  Olá, mundo.  ", "pt", "en")
    assert model.seen == ["  Olá, mundo.  "]


@pytest.mark.parametrize("text", ["", " ", "\n", "\u200b", "\u200b\u200e \n"])
def test_invisible_input_gives_nothing_and_reaches_no_model(model, text):
    assert model.translate(text, "pt", "en") == ""
    assert model.seen == []


def test_a_sentence_over_the_budget_still_raises(monkeypatch):
    def sentence(self, text, src, tgt, config, target_token):
        if text.startswith("Long"):
            raise OutputTruncatedError("cut", [1, 2])
        return text

    monkeypatch.setattr(TranslationModel, "_translate_sentence", sentence)
    with pytest.raises(OutputTruncatedError):
        TranslationModel("opus-mt-pt-en-int8").translate(
            "Short one. Long one. Short again.", "pt", "en")


def test_every_registry_target_has_a_script_answer():
    targets = {lang for entry in list_models()
               for lang in entry.get("languages", [])}
    assert len(targets) > 300
    for tag in targets:
        _default_script(tag)
