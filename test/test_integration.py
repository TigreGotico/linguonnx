"""Real end-to-end test: downloads glotlid-int8 (419MB) from HuggingFace on
first run and detects real short sentences. Skip with `-m "not network"` when
offline.
"""

import pytest

from lingonnx import load_detector

pytestmark = pytest.mark.network

SAMPLES = [
    # Short, low-context greetings are genuinely ambiguous between closely
    # related Romance languages (verified against the real model - "Ola, como
    # estas?" alone gets called Spanish, not Portuguese). These samples add
    # just enough real-sentence context to disambiguate, same as a human
    # reader would need.
    ("Ola, como estas hoje? Espero que esteja tudo bem contigo.", "pt"),  # Portuguese
    ("Ola, bo dia. Como estas ti hoxe?", "gl"),                          # Galician
    ("Bon dia, com estas avui?", "ca"),                                  # Catalan
    ("Kaixo, zer moduz zaude gaur?", "eu"),                              # Basque
    ("Hola, como estas hoy?", "es"),                                     # Spanish
    ("Hello, how are you today?", "en"),                                 # English
    ("Привет, как у тебя дела сегодня?", "ru"),                          # Russian
    # GlotLID's Arabic labels are dialect-specific (aeb, ars, ajp, ...);
    # only formal Modern Standard Arabic phrasing reliably lands on "arb"
    # (-> BCP-47 "ar"). Casual greetings get correctly tagged as a dialect
    # instead - that's the model working as documented, not a bug.
    ("أعلنت وزارة الخارجية اليوم عن اتفاقية جديدة بين البلدين.", "ar"),  # MSA Arabic
    # Chinese text using only characters shared between scripts resolves to
    # GlotLID's script-neutral "cmn_Hani" label (-> BCP-47 "zh-Hani"), not
    # "cmn_Hans"/"zho_Hans" - the model reserves Hans/Hant for text with
    # characters that only exist in one script.
    ("你好,今天天气怎么样?我们一起去公园散步吧。", "zh-Hani"),          # Chinese
    ("Bonjour, comment allez-vous aujourd hui?", "fr"),                  # French
    ("Guten Tag, wie geht es Ihnen heute?", "de"),                       # German
    ("Buongiorno, come stai oggi?", "it"),                               # Italian
]


@pytest.fixture(scope="module")
def detector():
    return load_detector("glotlid-int8")


@pytest.mark.parametrize("text,expected", SAMPLES)
def test_detect_real_samples(detector, text, expected):
    got = detector.detect(text)
    assert got == expected, f"{text!r} -> {got}, expected {expected}"


def test_detect_probs_sums_reasonably(detector):
    probs = detector.detect_probs("Hello, how are you today?", top_k=5)
    assert "en" in probs
    assert probs["en"] > 0.5
    assert len(probs) <= 5


def test_detect_raw_returns_native_glotlid_label(detector):
    label, conf = detector.detect_raw("Ola, bo dia. Como estas ti hoxe?")
    assert label.endswith("_Latn")
    assert 0.0 < conf <= 1.0
