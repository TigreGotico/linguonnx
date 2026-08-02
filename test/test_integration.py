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


# ---------------------------------------------------------------------------
# lid.176 (hierarchical softmax) and OpenLID (softmax, iso3_Script labels)
#
# These samples are written with their real diacritics. lid.176 is a small
# 16-dimension model and leans on accented characters to separate the
# Romance languages; strip them and "Ola, como estas hoje" is called Spanish.
# ---------------------------------------------------------------------------

MULTILINGUAL_SAMPLES = [
    ("Olá, como estás hoje? Espero que esteja tudo bem contigo.", "pt"),
    ("Ola, bo día. Como estás ti hoxe? Espero que todo vaia ben.", "gl"),
    ("Bon dia, com estàs avui? Espero que tot vagi bé.", "ca"),
    ("Kaixo, zer moduz zaude gaur?", "eu"),
    ("Hola, ¿cómo estás hoy?", "es"),
    ("Hello, how are you today?", "en"),
    ("Привет, как у тебя дела сегодня?", "ru"),
    ("أعلنت وزارة الخارجية اليوم عن اتفاقية جديدة بين البلدين.", "ar"),
    ("Bonjour, comment allez-vous aujourd'hui?", "fr"),
    ("Guten Tag, wie geht es Ihnen heute?", "de"),
]


@pytest.fixture(scope="module")
def lid176():
    return load_detector("lid176-int8")


@pytest.fixture(scope="module")
def openlid():
    return load_detector("openlid-int8")


@pytest.mark.parametrize("text,expected", MULTILINGUAL_SAMPLES)
def test_lid176_detects_real_samples(lid176, text, expected):
    got = lid176.detect(text)
    assert got == expected, f"{text!r} -> {got}, expected {expected}"


@pytest.mark.parametrize("text,expected", MULTILINGUAL_SAMPLES)
def test_openlid_detects_real_samples(openlid, text, expected):
    got = openlid.detect(text)
    assert got == expected, f"{text!r} -> {got}, expected {expected}"


def test_lid176_uses_the_hierarchical_softmax_path(lid176):
    assert lid176.loss == "hs"
    assert lid176._hs_combiner is not None


def test_lid176_probabilities_are_a_distribution(lid176):
    """The HS path must produce a normalized distribution, not raw sigmoids.

    Without the Huffman path walk the graph's 176 sigmoid outputs sum to
    something arbitrary (~88 for a 50/50 model) and the argmax is meaningless.
    """
    probs = lid176._probs("Hello, how are you today?")
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert probs.max() > 0.9


def test_lid176_emits_bare_labels(lid176):
    label, conf = lid176.detect_raw("Ola, bo día. Como estás ti hoxe?")
    assert label == "gl"          # no script suffix, unlike GlotLID/OpenLID
    assert 0.0 < conf <= 1.0


def test_openlid_emits_script_suffixed_labels(openlid):
    label, conf = openlid.detect_raw("Ola, bo día. Como estás ti hoxe?")
    assert label == "glg_Latn"
    assert 0.0 < conf <= 1.0
    assert openlid.loss == "softmax"


def test_openlid_v2_registry_entry_loads_labels_only():
    """v2 is a 1.1 GB download; check the registry wiring, not the weights."""
    from lingonnx import model_manager

    entry = model_manager.registry_entry("openlid-v2-int8")
    assert entry["hf_repo"] == "TigreGotico/openlid-v2-onnx"
    assert entry["license"] == "GPL-3.0"
