"""Translator model cache: bounded, thread-safe, and licence-enforcing.

Models are stubbed, so nothing downloads and no ONNX session is built.
"""

import threading
import time

import pytest

from linguonnx.translate import DEFAULT_MODEL_CACHE_SIZE, Translator
from linguonnx.translate.graph import Hop, Route

REGISTRY = {
    "multi": {
        "model_id": "multi", "hf_repo": "x/multi", "arch": "m2m100",
        "languages": ["en", "pt", "es", "ru"], "license": "MIT",
        "license_tier": "permissive", "size_mb": 1200, "precision": "int8",
    },
    "pt-en": {
        "model_id": "pt-en", "hf_repo": "x/pt-en", "arch": "marian",
        "pair": ["pt", "en"], "license": "Apache-2.0",
        "license_tier": "permissive", "size_mb": 350, "precision": "int8",
    },
    "en-eu": {
        "model_id": "en-eu", "hf_repo": "x/en-eu", "arch": "marian",
        "pair": ["en", "eu"], "license": "Apache-2.0",
        "license_tier": "permissive", "size_mb": 350, "precision": "int8",
    },
    "en-ru": {
        "model_id": "en-ru", "hf_repo": "x/en-ru", "arch": "marian",
        "pair": ["en", "ru"], "license": "Apache-2.0",
        "license_tier": "permissive", "size_mb": 350, "precision": "int8",
    },
}

NONCOMMERCIAL = {
    "model_id": "nllb-600M", "hf_repo": "x/nllb", "arch": "nllb",
    "languages": ["eng_Latn", "por_Latn", "eus_Latn"],
    "license": "CC-BY-NC-4.0", "license_tier": "non-commercial",
    "size_mb": 1800, "precision": "int8",
}


class FakeModel:
    def __init__(self, model_id, entry, delay=0.0):
        from linguonnx.translate.models import capability_from_entry
        self.model_id = model_id
        self.capability = capability_from_entry(entry)
        # The attributes eviction clears; a real TranslationModel holds the
        # three InferenceSessions behind `_decoder`.
        self._decoder = object()
        self._tokenizer = object()
        if delay:
            time.sleep(delay)

    def translate(self, text, src, tgt, config=None, target_token=None):
        return f"{text}|{self.model_id}"


@pytest.fixture
def counting_factory(monkeypatch):
    """Patches TranslationModel with a builder that counts constructions."""
    built = []
    lock = threading.Lock()

    def factory(model_id, entry=None, delay=0.0,
                enforce_download_budget=True):
        # `enforce_download_budget` is accepted because `Translator` passes
        # it to every `TranslationModel` it builds; this double stands in
        # for that constructor and has to have its shape.
        with lock:
            built.append(model_id)
        return FakeModel(model_id, entry or REGISTRY[model_id], delay=delay)

    monkeypatch.setattr("linguonnx.translate.TranslationModel", factory)
    return built


# -- R5: the cache is bounded and eviction releases the sessions --------------

def test_cache_never_grows_past_its_limit(counting_factory):
    tx = Translator(REGISTRY, model_cache_size=2)
    for model_id in ("multi", "pt-en", "en-eu", "en-ru"):
        tx.model(model_id)
    assert len(tx.loaded_models) == 2


def test_eviction_is_least_recently_used(counting_factory):
    tx = Translator(REGISTRY, model_cache_size=2)
    tx.model("multi")
    tx.model("pt-en")
    tx.model("multi")     # multi is now the most recent, pt-en the oldest
    tx.model("en-eu")     # evicts pt-en, not multi
    assert set(tx.loaded_models) == {"multi", "en-eu"}


def test_evicted_model_releases_its_onnx_sessions(counting_factory):
    tx = Translator(REGISTRY, model_cache_size=1)
    first = tx.model("multi")
    tx.model("pt-en")
    # A caller who kept the object must not keep three InferenceSessions alive
    # with it; the lazy attributes are cleared on eviction.
    assert first._decoder is None
    assert first._tokenizer is None


def test_evicted_model_is_rebuilt_on_next_use(counting_factory):
    tx = Translator(REGISTRY, model_cache_size=1)
    tx.model("multi")
    tx.model("pt-en")
    tx.model("multi")
    assert counting_factory == ["multi", "pt-en", "multi"]


def test_cache_size_must_be_at_least_one(counting_factory):
    with pytest.raises(ValueError, match="at least 1"):
        Translator(REGISTRY, model_cache_size=0)


def test_default_cache_size_is_bounded():
    assert 1 <= DEFAULT_MODEL_CACHE_SIZE < len(REGISTRY) + 100


def test_eviction_is_logged(counting_factory, caplog):
    tx = Translator(REGISTRY, model_cache_size=1)
    with caplog.at_level("DEBUG", logger="linguonnx.translate"):
        tx.model("multi")
        tx.model("pt-en")
    assert any("evicting multi" in record.getMessage() for record in caplog.records)


# -- R6: a cold load happens once, even from two threads ----------------------

def test_two_threads_racing_one_cold_model_build_it_once(monkeypatch):
    built = []
    lock = threading.Lock()
    start = threading.Barrier(2)

    def slow_factory(model_id, entry=None,
                     enforce_download_budget=True):
        # `Translator` passes `enforce_download_budget` to every
        # `TranslationModel` it builds; this double stands in for that
        # constructor and has to have its shape.
        with lock:
            built.append(model_id)
        time.sleep(0.2)  # the window a plain check-then-set leaves open
        return FakeModel(model_id, entry or REGISTRY[model_id])

    monkeypatch.setattr("linguonnx.translate.TranslationModel", slow_factory)
    tx = Translator(REGISTRY, model_cache_size=4)
    got = []

    def worker():
        start.wait(timeout=10)
        got.append(tx.model("multi"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert built == ["multi"], "the model was constructed more than once"
    assert got[0] is got[1], "one thread got an orphaned model"


def test_loading_one_model_does_not_block_another(monkeypatch):
    """Per-model locks, not one global one: a 2 GB load must not stall a hop
    through an unrelated model."""
    releases = {"multi": threading.Event()}
    inside = threading.Event()

    def blocking_factory(model_id, entry=None,
                         enforce_download_budget=True):
        # `Translator` passes `enforce_download_budget` to every
        # `TranslationModel` it builds; this double stands in for that
        # constructor and has to have its shape.
        if model_id in releases:
            inside.set()
            releases[model_id].wait(timeout=10)
        return FakeModel(model_id, entry or REGISTRY[model_id])

    monkeypatch.setattr("linguonnx.translate.TranslationModel", blocking_factory)
    tx = Translator(REGISTRY, model_cache_size=4)

    slow = threading.Thread(target=lambda: tx.model("multi"))
    slow.start()
    assert inside.wait(timeout=10)

    done = threading.Event()
    threading.Thread(target=lambda: (tx.model("pt-en"), done.set())).start()
    assert done.wait(timeout=5), "a second model waited on an unrelated load"

    releases["multi"].set()
    slow.join(timeout=10)


# -- R7: the licence filter holds at load time --------------------------------

def test_model_outside_the_selection_is_refused(counting_factory):
    tx = Translator(REGISTRY)
    with pytest.raises(ValueError, match="include_noncommercial"):
        tx.model("nllb-600M")


def test_caller_supplied_route_cannot_smuggle_in_a_filtered_model(counting_factory):
    """`include_noncommercial=False` is documented as a guarantee, so a route=
    naming an excluded model must fail rather than translate through it."""
    tx = Translator(REGISTRY)
    hop = Hop(model_id="nllb-600M", src="pt", tgt="eu", arch="nllb",
              license="CC-BY-NC-4.0", license_tier="non-commercial",
              size_mb=1800, dedicated=False)
    route = Route("pt", "eu", (hop,), prefer="pinned", max_hops=1)
    with pytest.raises(ValueError, match="include_noncommercial"):
        tx.translate("bom dia", route=route)


def test_pinned_model_outside_the_selection_is_refused(counting_factory):
    tx = Translator(REGISTRY)
    with pytest.raises(ValueError, match="include_noncommercial"):
        tx.translate("bom dia", src="pt", tgt="eu", model="nllb-600M")


def test_a_selected_noncommercial_model_still_loads(monkeypatch):
    """The guard is about *this translator's* selection, not the licence tier:
    with the model included, it must load normally."""
    entries = dict(REGISTRY, **{"nllb-600M": NONCOMMERCIAL})

    monkeypatch.setattr(
        "linguonnx.translate.TranslationModel",
        lambda model_id, entry=None, enforce_download_budget=True:
        FakeModel(model_id, entry or entries[model_id]))
    tx = Translator(entries)
    assert tx.model("nllb-600M").model_id == "nllb-600M"


# -- R9: the chosen route is visible to an operator ---------------------------

def test_route_selection_is_logged(counting_factory, caplog):
    tx = Translator(REGISTRY)
    with caplog.at_level("INFO", logger="linguonnx.translate"):
        tx.translate("bom dia", src="pt", tgt="eu")
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "translating pt -> eu via" in messages
    assert "pt-en" in messages and "en-eu" in messages


def test_model_load_is_logged(counting_factory, caplog):
    tx = Translator(REGISTRY)
    with caplog.at_level("INFO", logger="linguonnx.translate"):
        tx.model("multi")
    assert any("loading translation model multi" in record.getMessage()
               for record in caplog.records)
