"""What bounds memory, and what does not.

The doubles here are **lazily built on purpose**. A `TranslationModel` sets
`_decoder` and `_tokenizer` to `None` in `__init__` (models.py) and builds the
ONNX sessions on first use, so a double that assigns them in its constructor
models "always built" - the opposite of the contract - and makes every
lazy-path bug in the cache unreachable. `LazyFake` below is the faithful
shape: `None` at construction, set during `translate`, and never set at all
when `translate` returns early on blank input the way the real one does.
"""

import threading

import pytest

from linguonnx.translate import Translator
from linguonnx.translate.models import capability_from_entry


def _entry(model_id, size_mb):
    return {"model_id": model_id, "hf_repo": f"x/{model_id}", "arch": "marian",
            "pair": ["pt", "en"], "license": "Apache-2.0",
            "license_tier": "permissive", "size_mb": size_mb,
            "precision": "int8"}


class LazyFake:
    """A double with the real model's lazy-build shape."""

    def __init__(self, model_id, entry=None, enforce_download_budget=True):
        self.model_id = model_id
        self.entry = entry
        self.capability = capability_from_entry(entry)
        # Exactly as TranslationModel.__init__ leaves them.
        self._decoder = None
        self._tokenizer = None
        self.hold = None

    def translate(self, text, src, tgt, config=None, target_token=None):
        # TranslationModel.translate returns before touching either lazy
        # attribute when the input is blank.
        if not text.strip():
            return ""
        self._decoder = object()
        self._tokenizer = object()
        if self.hold is not None:
            self.hold()
        return f"[{self.model_id}]{text}"


@pytest.fixture
def make(monkeypatch):
    def _make(sizes, hold=None, **kwargs):
        registry = {model_id: _entry(model_id, size_mb)
                    for model_id, size_mb in sizes.items()}

        def factory(model_id, entry=None, enforce_download_budget=True):
            # `hold` is attached here, not after `model()` returns, so it
            # survives an eviction and rebuild - otherwise a concurrency
            # test would silently stop blocking and measure nothing.
            model = LazyFake(model_id, entry or registry[model_id])
            model.hold = hold
            return model

        monkeypatch.setattr("linguonnx.translate.TranslationModel", factory)
        return Translator(registry, **kwargs)
    return _make


class TestBlankInputDoesNotUnbookAModel:
    """A whitespace-only request must not disturb the cache.

    `utterance` is a path parameter on ovos-translate-server, so `%20` is a
    request an unauthenticated caller can send on repeat. If a blank request
    can drop a healthy model from the cache, that is a free cache-busting DoS:
    every following request pays a cold rebuild, up to MADLAD's 4945 MB.
    """

    def test_blank_input_keeps_the_model_cached(self, make):
        tx = make({"m": 350}, model_cache_size=4, max_loaded_mb=10_000)
        assert tx.translate("   ", model="m") == ""
        assert tx.loaded_models == ["m"]
        assert tx.loaded_mb == 350

    def test_blank_request_does_not_evict_a_warm_model(self, make):
        tx = make({"a": 350, "b": 350}, model_cache_size=4)
        tx.translate("ola", model="a")
        tx.translate("   ", model="b")
        assert sorted(tx.loaded_models) == ["a", "b"]


class TestEvictionDoesNotCorruptAnInFlightModel:
    """Eviction removes the cache entry and touches nothing else.

    Clearing the evicted model's lazy attributes is a no-op when the cache
    held the last reference and a bug when it did not: the thread still
    decoding rebuilds its sessions on an object no longer in `_loaded`, so
    the bytes are resident, uncounted and unevictable.
    """

    def test_evicted_model_keeps_the_sessions_its_user_built(self, make):
        tx = make({"a": 350, "b": 350}, model_cache_size=1)
        held = tx.model("a")
        held.translate("ola", "pt", "en")
        assert held._decoder is not None
        tx.translate("hi", model="b")          # evicts "a"
        assert tx.loaded_models == ["b"]
        # The object the caller still holds is untouched, so it cannot
        # silently rebuild a second set of sessions behind the cache's back.
        assert held._decoder is not None
        assert held._tokenizer is not None


class TestCacheBudgetIsExact:
    """`max_loaded_mb` bounds what the cache retains, always."""

    def test_byte_budget_evicts_before_the_count_budget(self, make):
        tx = make({"a": 400, "b": 400, "c": 400},
                  model_cache_size=10, max_loaded_mb=900)
        for model_id in ("a", "b", "c"):
            tx.translate("ola", model=model_id)
        assert tx.loaded_mb <= 900
        assert tx.loaded_models == ["b", "c"]

    def test_a_model_larger_than_the_whole_budget_still_loads(self, make):
        tx = make({"big": 4945}, model_cache_size=4, max_loaded_mb=2000)
        assert tx.translate("ola", model="big") == "[big]ola"
        assert tx.loaded_models == ["big"]

    def test_budget_holds_while_32_models_are_in_flight(self, make):
        """The cache never retains more than its budget, whatever runs.

        This is the honest claim, and the narrow one: it is about what the
        cache RETAINS, not about peak RSS - see TestConcurrencyIsWhatBoundsPeak
        for the term that a byte budget cannot reach.

        The measurement is taken while all 32 decodes are genuinely parked
        in flight, not after they drain. A design that keeps in-flight models
        booked in the cache reports 32 x 441 = 14112 MB here against a
        2000 MB budget.
        """
        sizes = {f"m{i}": 441 for i in range(32)}
        sampled = []
        probe = _ConcurrencyProbe(expected=32)
        tx = make(sizes, hold=probe, model_cache_size=4, max_loaded_mb=2000)
        # Sampled from inside the parked window, so the cache is observed at
        # its fullest rather than after the threads have drained.
        probe.on_full = lambda: sampled.append(tx.loaded_mb)
        assert probe.run(tx, threads=32) == 32, "the 32 decodes did not overlap"
        assert sampled, "never observed the cache with all 32 in flight"
        assert max(sampled) <= 2000, \
            f"cache retained {max(sampled)} MB over a 2000 MB budget"
        assert tx.loaded_mb <= 2000


class TestConcurrencyIsWhatBoundsPeak:
    """`max_concurrent_translations` is the only cap on models in flight."""

    def test_no_limit_lets_every_thread_hold_its_own_model(self, make):
        # Unbounded: all eight decode at once, so eight models are resident
        # at the same instant no matter what model_cache_size says. This is
        # the term a cache budget cannot touch.
        probe = _ConcurrencyProbe(expected=8)
        tx = make({f"m{i}": 441 for i in range(8)}, hold=probe,
                  model_cache_size=4)
        assert probe.run(tx, threads=8) == 8

    def test_limit_caps_models_in_flight(self, make):
        probe = _ConcurrencyProbe(expected=2)
        tx = make({f"m{i}": 441 for i in range(8)}, hold=probe,
                  model_cache_size=4, max_concurrent_translations=2)
        assert probe.run(tx, threads=8) == 2

    def test_rejects_a_nonsense_limit(self, make):
        with pytest.raises(ValueError, match="max_concurrent_translations"):
            make({"a": 350}, max_concurrent_translations=0)

    def test_multi_hop_route_does_not_deadlock_on_one_slot(self, monkeypatch):
        """A slot is taken per translation, not per hop.

        Per-hop acquisition would let a two-hop route block on itself the
        moment the limit is 1 - a deadlock that only shows up in production,
        on the pivot routes that are most of the graph.
        """
        registry = {"pt-en": _entry("pt-en", 350),
                    "en-eu": _entry("en-eu", 350)}
        registry["en-eu"]["pair"] = ["en", "eu"]

        def factory(model_id, entry=None, enforce_download_budget=True):
            return LazyFake(model_id, entry or registry[model_id])

        monkeypatch.setattr("linguonnx.translate.TranslationModel", factory)
        tx = Translator(registry, max_concurrent_translations=1)
        done = []

        def run():
            done.append(tx.translate("ola", src="pt", tgt="eu",
                                     return_route=True))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive(), "two-hop route deadlocked on its own slot"
        out, route = done[0]
        assert len(route.hops) == 2
        assert out == "[en-eu][pt-en]ola"


class _ConcurrencyProbe:
    """Parks each decode until `expected` of them are parked together.

    Measuring a peak by sampling from outside races the workers and reports
    whatever it happened to catch. This blocks instead: the barrier can only
    fill if `expected` decodes really are in flight at the same instant, and
    if the limit forbids that the test fails on the barrier timeout rather
    than passing on a lucky sample.
    """

    def __init__(self, expected, timeout=30):
        self.expected = expected
        self.gate = threading.Barrier(expected, timeout=timeout)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        #: Called once, while the window is full, for tests that want to
        #: observe state at the moment of maximum overlap.
        self.on_full = None

    def __call__(self):
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            full = self.in_flight == self.expected
        if full and self.on_full is not None:
            self.on_full()
        self.gate.wait()
        with self.lock:
            self.in_flight -= 1

    def run(self, tx, threads):
        workers = [threading.Thread(target=tx.translate, args=("ola",),
                                    kwargs={"model": f"m{i}"})
                   for i in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        return self.peak
