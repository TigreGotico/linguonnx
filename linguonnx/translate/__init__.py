"""Offline translation on ONNX Runtime.

```python
from linguonnx import load_translator

tx = load_translator()
tx.translate("bom dia", src="pt", tgt="eu")            # -> str
tx.translate("bom dia", src="pt", tgt="eu", return_route=True)   # -> (str, Route)
tx.route("pt", "eu")                                   # inspect, translate nothing
tx.routes("pt", "eu")                                  # every viable route, ranked
tx.available_languages                                 # union over the graph
```

Routing is explained in :mod:`linguonnx.translate.graph`, the decode loop in
:mod:`linguonnx.translate.decode`. Nothing here imports torch.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import (Dict, Iterable, Iterator, List, Optional, Sequence, Tuple,
                    Union)

from linguonnx.limits import operator_budget_is_set
from linguonnx import model_manager
from linguonnx.model_manager import list_models
from linguonnx.translate.decode import GenerationConfig
from linguonnx.translate.graph import (DEFAULT_PIVOT_PREFERENCE, UNSET,
                                       Capability, Hop, NoRouteError, Route,
                                       TranslationGraph, _Unset, normalize_tag)
from linguonnx.translate.models import TranslationModel, capability_from_entry
from linguonnx.translate.quality import (flagged_target_languages,
                                         is_language_flagged,
                                         is_quality_flagged,
                                         language_flag_reason_for,
                                         quality_flag_reasons_for)

LOG = logging.getLogger(__name__)

__all__ = ["Translator", "load_translator", "Route", "Hop", "NoRouteError",
           "GenerationConfig", "is_quality_flagged", "quality_flag_reasons_for",
           "is_language_flagged", "language_flag_reason_for"]

#: How many loaded models a Translator keeps alive at once. The full default
#: graph is 73 models / ~25 GB, so an unbounded cache converges on the whole
#: registry in a long-lived server and gets the process OOM-killed; it then
#: restarts cold and pays every download again. Four keeps a two-hop route and
#: its neighbours warm while staying inside a few GB.
DEFAULT_MODEL_CACHE_SIZE = 4

#: Byte budget for the *cache*, in MB. ``None`` means no byte budget, so
#: ``model_cache_size`` alone bounds it, exactly as before this field existed.
#:
#: This bounds what the cache RETAINS. It does not bound peak RSS, and no
#: cache setting can: a model being translated through is resident because a
#: thread is using it, not because the cache kept it. See
#: :attr:`Translator.loaded_mb` and ``docs/models.md`` for the real formula.
DEFAULT_MAX_LOADED_MB = None

#: How many translations may run at once, or ``None`` for no limit. This is
#: the only setting that bounds peak RSS, because the in-flight models are
#: the dominant term on a threaded server - see ``docs/models.md``.
DEFAULT_MAX_CONCURRENT_TRANSLATIONS = None


def _select_entries(precision: Optional[str], include_noncommercial: bool,
                    models: Optional[Sequence[str]],
                    exclude_flagged: bool = False,
                    min_chrf: Optional[float] = None) -> Dict[str, dict]:
    registry = list_models(kind="translate")
    if models is not None:
        # An explicit `models=` is a caller naming exactly what they want -
        # it overrides every other filter, quality included, the same way it
        # already overrode `precision=`/`include_noncommercial=` before this
        # field existed.
        missing = [m for m in models if m not in registry]
        if missing:
            raise ValueError(f"unknown translation model(s): {', '.join(missing)}")
        return {m: registry[m] for m in models}
    chosen = {}
    for model_id, entry in registry.items():
        if precision and entry["precision"] != precision:
            continue
        if not include_noncommercial and entry["license_tier"] == "non-commercial":
            continue
        if exclude_flagged and is_quality_flagged(model_id, registry):
            continue
        if min_chrf is not None:
            quality = entry.get("quality") or {}
            chrf = quality.get("chrf_vs_ref")
            # Absence means "not measured", never "scored badly" - it must
            # not be excluded by a quality floor it was never checked
            # against. Only a model actually measured below the floor is cut.
            if chrf is not None and chrf < min_chrf:
                continue
        chosen[model_id] = entry
    return chosen


class Translator:
    """A routing graph plus the models it can route through.

    Models are downloaded and their ONNX sessions built **lazily**, on the
    first hop that actually needs them, so building a Translator over the whole
    registry costs nothing but JSON parsing.
    """

    def __init__(self, entries: Dict[str, dict],
                 prefer: str = "fewest_hops", max_hops: int = 2,
                 pivot_preference: Sequence[str] = DEFAULT_PIVOT_PREFERENCE,
                 max_routes: int = 10,
                 pivot_ranking: str = "auto",
                 max_model_mb: Union[int, None, _Unset] = UNSET,
                 count_cached_as_free: Union[bool, _Unset] = UNSET,
                 oversize_fallback: bool = False,
                 num_beams: int = 4, max_new_tokens: int = 128,
                 length_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
                 model_cache_size: int = DEFAULT_MODEL_CACHE_SIZE,
                 max_loaded_mb: Optional[int] = DEFAULT_MAX_LOADED_MB,
                 max_concurrent_translations: Optional[int] =
                 DEFAULT_MAX_CONCURRENT_TRANSLATIONS,
                 enforce_download_budget: bool = True,
                 fetch_on_demand: bool = True):
        # Whether an absent model may be downloaded on the request path.
        # `False` is the embedded/metered stance: routing sees only what is
        # already on disk, so latency is bounded by decode instead of by a
        # download nobody asked for. The absent models are kept - not
        # forgotten - so a miss can name what to prefetch.
        self.fetch_on_demand = bool(fetch_on_demand)
        absent: Dict[str, dict] = {}
        if not self.fetch_on_demand:
            cached: Dict[str, dict] = {}
            for model_id, entry in entries.items():
                target = cached if model_manager.is_cached(
                    model_id, kind="translate") else absent
                target[model_id] = entry
            if not cached:
                raise ValueError(
                    "fetch_on_demand=False and none of the "
                    f"{len(absent)} selected translation model(s) are cached; "
                    "prefetch at least one (linguonnx.model_manager.prefetch) "
                    "or pass fetch_on_demand=True")
            entries = cached
        self._entries = entries
        # Routing and downloading have to agree. `max_model_mb` keeps the
        # router from proposing a model the download path would refuse; when
        # the router's budget is waived for models the caller named by hand,
        # the download budget has to be waived with it, or `route()` answers
        # with a hop that `translate()` then rejects with
        # `DownloadTooLargeError` - `can_translate` lying, which this library
        # already fixed once for unrunnable architectures.
        self._enforce_download_budget = enforce_download_budget
        self.graph = TranslationGraph(
            [capability_from_entry(e) for e in entries.values()],
            pivot_preference=pivot_preference, prefer=prefer,
            max_hops=max_hops, max_routes=max_routes,
            pivot_ranking=pivot_ranking, max_model_mb=max_model_mb,
            count_cached_as_free=count_cached_as_free,
            oversize_fallback=oversize_fallback)
        if pivot_ranking == "auto" and self.graph.pivot_ranking != "phonological":
            # `auto` degrades silently when orthography2ipa is absent, and the
            # two bases can pick different pivots - so two hosts in one fleet
            # would return different translations with nothing to explain it.
            LOG.warning("pivot_ranking='auto' fell back to %r: orthography2ipa "
                        "is not installed (pip install linguonnx[distance])",
                        self.graph.pivot_ranking)
        # Capabilities left out by ANY selection filter - licence tier,
        # precision, or an explicit `models=` - so a failed lookup can say
        # "a non-commercial model covers this" or "an fp32 model covers this"
        # instead of looking unsupported. Uniform across filters on purpose:
        # licence was not special-cased here before, and a caller who tuned
        # `precision=` deserves the same "here is what you filtered out" as
        # one who left non-commercial models out.
        chosen = set(entries)
        self.graph.excluded_capabilities = [
            capability_from_entry(e)
            for model_id, e in list_models(kind="translate").items()
            if model_id not in chosen and model_id not in absent
        ] if entries else []
        # Left out by `fetch_on_demand=False` specifically, and kept apart
        # from the filter exclusions above: "a model covers this pair, it is
        # just not on this disk yet" is a different sentence, with a different
        # fix, from "a model covers this pair and you filtered it out".
        self.graph.uncached_capabilities = [capability_from_entry(e)
                                            for e in absent.values()]
        self.generation = GenerationConfig(
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size)
        if model_cache_size < 1:
            raise ValueError("model_cache_size must be at least 1")
        self._cache_size = model_cache_size
        if max_loaded_mb is not None and max_loaded_mb < 1:
            raise ValueError("max_loaded_mb must be at least 1, or None")
        self._max_loaded_mb = max_loaded_mb
        if max_concurrent_translations is not None \
                and max_concurrent_translations < 1:
            raise ValueError(
                "max_concurrent_translations must be at least 1, or None")
        self._max_concurrent = max_concurrent_translations
        # The only bound on peak RSS. A cache limit cannot provide one: an
        # in-flight model is held by the thread decoding through it, not by
        # the cache, so evicting it frees nothing. Limiting how many decodes
        # run at once limits how many models can be resident at once.
        self._slots = (threading.Semaphore(max_concurrent_translations)
                       if max_concurrent_translations is not None else None)
        self._loaded: "OrderedDict[str, TranslationModel]" = OrderedDict()
        # `_cache_lock` guards the cache bookkeeping only and is never held
        # across a load. The per-model locks in `_load_locks` are what a cold
        # load holds, so building model A does not block a caller who needs
        # model B.
        self._cache_lock = threading.Lock()
        self._load_locks: Dict[str, threading.Lock] = {}

    # -- introspection ----------------------------------------------------

    @property
    def available_languages(self) -> frozenset:
        """Every BCP-47 tag reachable as a source or a target, over all models."""
        return self.graph.languages

    def quality_flag_reasons(self, model_id: str) -> Tuple[str, ...]:
        """Why ``model_id`` is quality-flagged, or ``()`` when it is not.

        Looked up against the full registry, not just this ``Translator``'s
        own selection, so an int8 entry's counterpart is found even when the
        caller filtered it out with ``precision="int8"`` - the gap check
        needs fp32's number regardless of whether fp32 itself is in this
        graph.
        """
        return quality_flag_reasons_for(model_id)

    def language_flag_reason(self, model_id: str, lang: str,
                             side: str = "target") -> Optional[str]:
        """Why ``model_id`` must not be used for ``lang``, or ``None``.

        Looked up against the full registry for the same reason
        :meth:`quality_flag_reasons` is: the flag is a fact about the export,
        so the precision counterpart's flags count even when this
        ``Translator`` filtered that precision out.
        """
        return language_flag_reason_for(model_id, lang, side)

    @property
    def flagged_languages(self) -> Dict[str, Tuple[str, ...]]:
        """``lang -> reasons`` for languages this translator cannot *write*.

        A language is listed only when every selected model that covers it as
        a target is flagged for it, so a flag on one of two providers costs
        nothing. See :func:`~linguonnx.translate.quality.flagged_target_languages`.
        """
        return flagged_target_languages(self._entries,
                                        list_models(kind="translate"))

    @property
    def unflagged_languages(self) -> frozenset:
        """:attr:`available_languages` minus :attr:`flagged_languages`.

        **Routable is not usable.** ``available_languages`` counts every tag
        some in-budget model *claims*, and a claim can be wrong: MADLAD
        advertises Chuvash and answers it in Russian. This is the number to
        quote as coverage.

        A language reachable only as a *source* is kept: a target-side flag
        says nothing about reading it.
        """
        return self.available_languages - frozenset(self.flagged_languages)

    @property
    def models(self) -> Dict[str, dict]:
        """The registry entries this translator routes over."""
        return dict(self._entries)

    @property
    def prefer(self) -> str:
        return self.graph.prefer

    @property
    def max_hops(self) -> int:
        return self.graph.max_hops

    @property
    def max_model_mb(self) -> Optional[int]:
        """The per-model size budget in force, or ``None`` for no budget."""
        return self.graph.max_model_mb

    @property
    def count_cached_as_free(self) -> bool:
        """Whether a model already on disk is exempt from :attr:`max_model_mb`."""
        return self.graph.count_cached_as_free

    @property
    def oversize_fallback(self) -> bool:
        """Whether a pair no model under :attr:`max_model_mb` covers may fall
        back to the smallest oversized model that does."""
        return self.graph.oversize_fallback

    @property
    def pivot_ranking(self) -> str:
        """The pivot ranking in force, ``"phonological"`` or ``"table"``."""
        return self.graph.pivot_ranking

    def route(self, src: str, tgt: str, max_hops: Optional[int] = None,
              prefer: Optional[str] = None,
              max_model_mb: Union[int, None, _Unset] = UNSET,
              count_cached_as_free: Optional[bool] = None,
              oversize_fallback: Optional[bool] = None) -> Route:
        """The best route, without translating. Raises :class:`NoRouteError`."""
        return self.graph.route(src, tgt, max_hops=max_hops, prefer=prefer,
                                max_model_mb=max_model_mb,
                                count_cached_as_free=count_cached_as_free,
                                oversize_fallback=oversize_fallback)

    def routes(self, src: str, tgt: str, max_hops: Optional[int] = None,
               prefer: Optional[str] = None,
               limit: Optional[int] = None,
               max_model_mb: Union[int, None, _Unset] = UNSET,
               count_cached_as_free: Optional[bool] = None,
               oversize_fallback: Optional[bool] = None) -> List[Route]:
        """Every viable route, ranked best-first. Bounded; see the graph docstring."""
        return self.graph.routes(src, tgt, max_hops=max_hops, prefer=prefer,
                                 limit=limit, max_model_mb=max_model_mb,
                                 count_cached_as_free=count_cached_as_free,
                                 oversize_fallback=oversize_fallback)

    def can_translate(self, src: str, tgt: str,
                      max_hops: Optional[int] = None,
                      max_model_mb: Union[int, None, _Unset] = UNSET,
                      count_cached_as_free: Optional[bool] = None,
                      oversize_fallback: Optional[bool] = None) -> bool:
        return self.graph.can_translate(
            src, tgt, max_hops=max_hops, max_model_mb=max_model_mb,
            count_cached_as_free=count_cached_as_free,
            oversize_fallback=oversize_fallback)

    # -- models -----------------------------------------------------------

    @property
    def loaded_models(self) -> List[str]:
        """Ids currently holding ONNX sessions, least recently used first."""
        with self._cache_lock:
            return list(self._loaded)

    @property
    def max_loaded_mb(self) -> Optional[int]:
        """The cache's byte budget in MB, or ``None`` for no budget."""
        return self._max_loaded_mb

    @property
    def max_concurrent_translations(self) -> Optional[int]:
        """How many translations may run at once, or ``None`` for no limit."""
        return self._max_concurrent

    @property
    def loaded_mb(self) -> int:
        """Declared MB the cache currently retains, over :attr:`loaded_models`.

        This is the registry's ``size_mb``, **not** measured RSS, and that is
        deliberate. The ONNX sessions are memory-mapped, so a session's RSS
        depends on which pages the last request happened to touch: it climbs
        as a model is used and does not fall when it stops being used. You
        cannot evict against a number that moves under you, and two hosts
        running the same route would read different numbers. ``size_mb``
        already drives routing (``max_model_mb``), it is stable, and it is
        known *before* the model is loaded - which is what a budget needs.

        It is a proxy, and a generous one. Measured on this library's own
        models, peak RSS runs **1.84x to 3.7x** the declared size, and the
        multiplier moves *inversely* with size: 78 MB -> 288 MB is 3.7x,
        1207 MB -> 2161 MB is 1.84x. A fixed per-process cost - ONNX Runtime
        and its arenas - dominates a small model and is noise for a large one.

        This counts what the **cache retains**, which is bounded by
        ``max_loaded_mb`` at all times. It is not process RSS and does not
        try to be: a model being translated through by a thread that already
        took it out of the cache is resident and is not counted here, because
        the cache neither owns it nor can free it. Bound that term with
        ``max_concurrent_translations``; see ``docs/models.md``.
        """
        with self._cache_lock:
            return self._loaded_mb_locked()

    def _touch(self, model_id: str) -> Optional[TranslationModel]:
        """The cached model, moved to the most-recent end. None if not cached."""
        with self._cache_lock:
            model = self._loaded.get(model_id)
            if model is not None:
                self._loaded.move_to_end(model_id)
            return model

    def _size_mb(self, model_id: str) -> int:
        """The registry's declared size for ``model_id``, in MB."""
        entry = self._entries.get(model_id)
        return int(entry["size_mb"]) if entry else 0

    def _loaded_mb_locked(self) -> int:
        return sum(self._size_mb(model_id) for model_id in self._loaded)

    def _evict_down_to_size(self) -> None:
        """Drop least-recently-used entries until both budgets are met.

        Eviction removes the dict entry and nothing else. It deliberately
        does **not** clear the evicted model's ``_decoder``/``_tokenizer``.
        Clearing them is a no-op when the cache held the last reference -
        dropping the entry already frees the sessions - and it is a bug when
        it did not: a thread decoding through that model still holds it, and
        the next attribute access rebuilds three InferenceSessions on an
        object no longer in ``_loaded``. Those bytes are then resident,
        uncounted and unevictable for the life of the object.

        Leaving the object alone makes eviction mean "the cache stops
        retaining this". The sessions die with the last reference, which is
        the last thread using them, which is exactly when they may die.
        """
        while len(self._loaded) > self._cache_size:
            model_id, _ = self._loaded.popitem(last=False)
            LOG.debug("evicting %s from the model cache (limit %d)",
                      model_id, self._cache_size)
        if self._max_loaded_mb is None:
            return
        # The byte budget, applied after the count budget: whichever binds
        # first evicts. Same LRU order, so the two limits cannot disagree
        # about *which* model goes - only about how many. One model is always
        # kept: a model bigger than the whole budget still loads, because
        # refusing it would delete a language rather than shrink a cache, and
        # the models covering the long tail are exactly the ones no small
        # budget admits.
        while len(self._loaded) > 1 and \
                self._loaded_mb_locked() > self._max_loaded_mb:
            model_id, _ = self._loaded.popitem(last=False)
            LOG.debug("evicting %s from the model cache (%d MB budget)",
                      model_id, self._max_loaded_mb)
        resident = self._loaded_mb_locked()
        if resident > self._max_loaded_mb:
            LOG.warning(
                "model cache holds %s at %d MB, over the %d MB max_loaded_mb "
                "budget; loading it anyway rather than refusing the language",
                next(iter(self._loaded)), resident, self._max_loaded_mb)

    @contextmanager
    def _translation_slot(self) -> Iterator[None]:
        """Hold one of ``max_concurrent_translations`` slots, or nothing.

        Callers over the limit block here rather than loading another model.
        Blocking is the point: this is what turns "peak RSS grows with
        whatever the web server's threadpool admits" into a number an
        operator chose.
        """
        if self._slots is None:
            yield
            return
        self._slots.acquire()
        try:
            yield
        finally:
            self._slots.release()

    def _reject_unselected(self, model_id: str) -> None:
        """Refuse a model this translator did not select, and say why.

        The routing graph can only report such a model as unknown, because it
        never saw it. This layer knows the difference between "no such model"
        and "filtered out", and naming the flag is the difference between a
        caller fixing their call and a caller assuming the model is missing.
        """
        raise ValueError(
            f"{model_id!r} is not in this translator's models. "
            f"It is filtered out (see include_noncommercial=, precision= "
            f"and models= on load_translator) or it does not exist.")

    def model(self, model_id: str) -> TranslationModel:
        """The loaded model for ``model_id``, building it on first use.

        Only ids inside this translator's own selection are loadable. Falling
        back to the full registry here would let a caller-supplied ``route=``
        translate through a model the licence filter excluded, which is exactly
        what ``include_noncommercial=False`` promises will not happen.
        """
        cached = self._touch(model_id)
        if cached is not None:
            return cached

        entry = self._entries.get(model_id)
        if entry is None:
            self._reject_unselected(model_id)

        # One lock per model id, so a cold 2 GB load blocks only the callers
        # who want that same model. Without it, two threads on the same cold
        # model each build three InferenceSessions and one set is orphaned.
        with self._cache_lock:
            lock = self._load_locks.setdefault(model_id, threading.Lock())
        with lock:
            # Re-check: another thread may have finished the load while this
            # one waited on the lock.
            cached = self._touch(model_id)
            if cached is not None:
                return cached
            LOG.info("loading translation model %s (%s, %s MB)", model_id,
                     entry["arch"], entry["size_mb"])
            model = TranslationModel(
                model_id, entry,
                enforce_download_budget=self._enforce_download_budget)
            with self._cache_lock:
                self._loaded[model_id] = model
                self._loaded.move_to_end(model_id)
                self._evict_down_to_size()
            return model

    # -- translation ------------------------------------------------------

    def translate(self, text: str, src: Optional[str] = None,
                  tgt: Optional[str] = None, *,
                  route: Optional[Route] = None,
                  model: Optional[str] = None,
                  return_route: bool = False,
                  max_hops: Optional[int] = None,
                  prefer: Optional[str] = None,
                  max_model_mb: Union[int, None, _Unset] = UNSET,
                  count_cached_as_free: Optional[bool] = None,
                  oversize_fallback: Optional[bool] = None,
                  num_beams: Optional[int] = None,
                  max_new_tokens: Optional[int] = None,
                  target_token: Optional[str] = None
                  ) -> Union[str, Tuple[str, Route]]:
        """Translate ``text``.

        Three ways to say *how*, in order of precedence:

        ``model=``
            Pin one registry model and bypass routing entirely.
        ``route=``
            Execute a caller-supplied :class:`Route` verbatim, even when it is
            not the top-ranked one. This is the escape hatch for a caller who
            disagrees with the cost model.
        ``src=``/``tgt=``
            The normal path: score the candidate routes and take the winner.

        ``return_route=True`` returns ``(text, route)`` so a multi-hop
        translation is never silent.
        """
        config = self.generation
        if num_beams is not None or max_new_tokens is not None:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens if max_new_tokens is not None
                else config.max_new_tokens,
                num_beams=num_beams if num_beams is not None else config.num_beams,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping)

        # The slot is taken before routing, not just before the hops, because
        # `model=` resolves its route by *loading* the model
        # (`_pinned_route`). Taking the slot after that would let any number
        # of `model=` callers load simultaneously and walk straight past the
        # limit - the one path that would make
        # "max_concurrent_translations bounds peak RSS" a false claim.
        #
        # One slot per translation, not per hop: the hops of a route run in
        # sequence, so a route holds at most one model in flight at a time,
        # and taking the slot per hop would let a two-hop route deadlock
        # against itself at `max_concurrent_translations=1`.
        with self._translation_slot():
            return self._translate_locked(
                text, src, tgt, route=route, model=model,
                return_route=return_route, max_hops=max_hops, prefer=prefer,
                max_model_mb=max_model_mb,
                count_cached_as_free=count_cached_as_free,
                oversize_fallback=oversize_fallback, config=config,
                target_token=target_token)

    def _translate_locked(self, text, src, tgt, *, route, model, return_route,
                          max_hops, prefer, max_model_mb,
                          count_cached_as_free, oversize_fallback, config,
                          target_token):
        """Route and run, with this translation's slot already held."""
        if model is not None:
            chosen = self._pinned_route(model, src, tgt)
        elif route is not None:
            # A caller-supplied route is the documented escape hatch from the
            # scoring policy, not from correctness. Without validation a
            # hand-built hop can run a one-directional model backwards --
            # IndicTrans2 en->indic, the nos-coda pairs, liv4ever -- and both
            # tags resolve in the model's code map, so it translates the wrong
            # way round fluently, with nothing raised.
            #
            # Selection is checked first, and deliberately: the graph can only
            # report an excluded model as unknown, while this layer knows *why*
            # it is absent and can name the flag that would allow it.
            for hop in route.hops:
                if hop.model_id not in self._entries:
                    self._reject_unselected(hop.model_id)
            chosen = self.graph.validate_route(route)
        else:
            if src is None or tgt is None:
                raise ValueError("give src= and tgt=, or a route=, or a model=")
            chosen = self.route(src, tgt, max_hops=max_hops, prefer=prefer,
                                max_model_mb=max_model_mb,
                                count_cached_as_free=count_cached_as_free,
                                oversize_fallback=oversize_fallback)

        # The route is the single thing an operator needs to explain a bad
        # translation: which models ran, in which order, and through which
        # pivot.
        LOG.info("translating %s -> %s via %s", chosen.src, chosen.tgt,
                 " | ".join(f"{hop.model_id}:{hop.src}->{hop.tgt}"
                            for hop in chosen.hops))

        out = text
        for hop in chosen.hops:
            out = self.model(hop.model_id).translate(
                out, hop.src, hop.tgt, config=config,
                target_token=target_token)
        return (out, chosen) if return_route else out

    def _pinned_route(self, model_id: str, src: Optional[str],
                      tgt: Optional[str]) -> Route:
        """A one-hop Route through a named model, with no scoring at all."""
        loaded = self.model(model_id)
        capability = loaded.capability
        if capability.pair is not None:
            hop_src, hop_tgt = capability.pair
        else:
            if src is None or tgt is None:
                raise ValueError(
                    f"{model_id} is multilingual; src= and tgt= are required")
            hop_src, hop_tgt = normalize_tag(src), normalize_tag(tgt)
            if not capability.covers(hop_src, hop_tgt):
                raise NoRouteError(
                    f"{model_id} does not cover {hop_src!r} -> {hop_tgt!r}")
        hop = Hop(model_id=model_id, src=hop_src, tgt=hop_tgt,
                  arch=capability.arch, license=capability.license,
                  license_tier=capability.license_tier,
                  size_mb=capability.size_mb, dedicated=capability.dedicated)
        return Route(hop_src, hop_tgt, (hop,), prefer="pinned", max_hops=1)


def load_translator(models: Optional[Sequence[str]] = None,
                    model: Optional[str] = None,
                    prefer: str = "fewest_hops",
                    max_hops: int = 2,
                    precision: Optional[str] = "int8",
                    include_noncommercial: bool = False,
                    exclude_flagged: bool = False,
                    min_chrf: Optional[float] = None,
                    pivot_preference: Sequence[str] = DEFAULT_PIVOT_PREFERENCE,
                    max_routes: int = 10,
                    pivot_ranking: str = "auto",
                    max_model_mb: Union[int, None, _Unset] = UNSET,
                    count_cached_as_free: Union[bool, _Unset] = UNSET,
                    oversize_fallback: bool = False,
                    num_beams: int = 4,
                    max_new_tokens: int = 128,
                    length_penalty: float = 1.0,
                    no_repeat_ngram_size: int = 0,
                    model_cache_size: int = DEFAULT_MODEL_CACHE_SIZE,
                    max_loaded_mb: Optional[int] = DEFAULT_MAX_LOADED_MB,
                    max_concurrent_translations: Optional[int] =
                    DEFAULT_MAX_CONCURRENT_TRANSLATIONS,
                    fetch_on_demand: bool = True) -> Translator:
    """Build a :class:`Translator` over the registry.

    The default graph is **every permissive-licensed int8 model**: M2M100-418M
    for general coverage plus the opus-mt bilingual pairs, which also supply
    Basque - M2M100 has no Basque at all, so ``eu`` is routed through
    ``opus-mt-*-eu`` instead of being silently dropped.

    NLLB-200 is in the registry but **out of the default graph**, because it is
    CC-BY-NC-4.0 and this library will not put a non-commercial licence into a
    caller's output without being asked. Pass
    ``include_noncommercial=True`` to add it.

    :param models: exact registry ids to route over; overrides the filters.
    :param model: pin a single model. Shorthand for ``models=[model]``.
    :param prefer: ``"fewest_hops"`` (default) or ``"dedicated"``.
    :param max_hops: 1 = direct models only, 2 = default, 3+ = allowed but see
        the README on diminishing returns.
    :param precision: ``"int8"`` (default), ``"fp32"``, or ``None`` for both.
    :param exclude_flagged: drop any model :func:`~linguonnx.translate.quality.is_quality_flagged`
        flags - either precision scoring below 40 chrF against the FLORES-200
        reference, or int8 trailing fp32 by more than 2 chrF. Combine with
        ``precision=None`` to fall back to fp32 wherever int8 alone is
        flagged (a pair whose fp32 side is not flagged still routes through
        it) instead of losing the pair entirely. See ``docs/routing.md`` for
        what the numbers mean and where they came from. Unmeasured models are
        never excluded by this - "not measured" is not a flag.
    :param min_chrf: drop any model whose measured chrF-vs-reference falls
        below this. Only applies to models actually measured; a model with no
        ``quality`` data is left alone; a missing measurement must never be
        read as a bad one.
    :param pivot_ranking: how to order pivot candidates for a 2-hop route.
        ``"auto"`` (default) ranks them by phonological distance when the
        optional ``orthography2ipa`` package is installed - ``pip install
        linguonnx[distance]`` - and by the curated table otherwise.
        ``"phonological"`` demands the package and raises without it;
        ``"table"`` ignores it. ``Route.pivot_basis`` reports which was used.
    :param max_model_mb: largest single model routing may use, in MB. Unset
        reads ``LINGUONNX_MAX_MODEL_MB``; ``None`` is no budget. A pair that a
        big multilingual model served in one hop is then served by a chain of
        small ones, so coverage survives a budget that latency does not.
    :param count_cached_as_free: whether a model already in the local cache is
        exempt from ``max_model_mb``. True reads the budget as "do not
        **download** more than this", which is the metered-connection reading;
        False reads it as "do not **use** a model bigger than this", which is
        the small-disk and the load-latency one. Unset picks False when
        ``oversize_fallback`` is on and True otherwise, because on a warm
        cache the exemption waives the budget for every model there is.
    :param oversize_fallback: whether ``max_model_mb`` is a preference rather
        than a filter. False (default) drops an oversized model from routing
        outright. True keeps the cap for every pair a smaller model can serve,
        and falls back to the *smallest* oversized model that covers a pair
        nothing under the cap does - so a cap tuned for latency does not also
        delete the long tail of languages that lives only inside MADLAD, NLLB
        and M2M100. ``Route.waived_size_cap`` says when the exception was
        used. See ``linguonnx.translate.graph`` for the full semantics.
    :param fetch_on_demand: whether a model that is not in the local cache may
        be downloaded on the request path. ``True`` (default) is today's
        behaviour: the first request for an uncached pair pays the download,
        which is the right trade on a well-provisioned server and keeps the
        full registry reachable. ``False`` is the embedded/metered/small-disk
        stance: routing sees only what is already on disk, so a pair a cached
        chain can serve is served by it, and a pair *nothing* cached covers
        raises :class:`NoRouteError` naming the models to prefetch instead of
        stalling the request for minutes. Orthogonal to ``max_model_mb``,
        which bounds a model's *size* rather than its presence: with
        ``fetch_on_demand=False`` an absent model is unavailable however small
        it is, and ``count_cached_as_free`` becomes moot for it because the
        only models left are cached ones. The default is deliberately the
        permissive one - silently narrowing an existing deployment's coverage
        on upgrade would be a worse failure than a slow first request.
    :param num_beams: 4 by default; 1 is greedy and about 4x faster.
    :param max_loaded_mb: byte budget for the cache, in MB, or ``None``.
        Bounds what the cache *retains*, not peak RSS - see
        ``max_concurrent_translations`` and ``docs/models.md``.
    :param max_concurrent_translations: how many translations may run at
        once, or ``None`` for no limit. The only setting that bounds peak
        RSS on a threaded server.
    :param model_cache_size: how many loaded models to keep alive at once,
        least-recently-used evicted first. The whole default graph is ~25 GB,
        so an unbounded cache is an OOM kill in any long-lived server that
        routes over many language pairs.
    """
    if model is not None:
        models = [model] if models is None else list(models) + [model]
    entries = _select_entries(precision, include_noncommercial, models,
                              exclude_flagged=exclude_flagged,
                              min_chrf=min_chrf)
    if not entries:
        raise ValueError("no translation models matched the given filters")
    waive_budget = models is not None and isinstance(max_model_mb, _Unset) \
        and not operator_budget_is_set()
    if waive_budget:
        # `models=`/`model=` is a caller naming exactly what they want, and it
        # already overrides every entry filter. The size budget has to follow,
        # or a graph built from one named model routes nothing: every fp32
        # model above the library default (`aina-translator-ca-zh`, 9294 MB)
        # raised `NoRouteError` for the pair it is the only model for, while
        # its int8 twin (2345 MB) served the same pair.
        #
        # Only the *library's* own defaults are waived. Naming a model
        # overrides library policy; it does not override the operator's, and
        # `UNSET` is not "the default" - it reads `LINGUONNX_MAX_MODEL_MB` and
        # then the download budget. Collapsing it to `None` unconditionally
        # discarded a budget set on a metered link or a small-disk device:
        # under `LINGUONNX_MAX_MODEL_MB=500`, a `models=` list of the 21
        # registry entries over 4 GB routed all 136 GB of them.
        max_model_mb = None
    return Translator(entries, prefer=prefer, max_hops=max_hops,
                      enforce_download_budget=not waive_budget,
                      pivot_preference=pivot_preference, max_routes=max_routes,
                      pivot_ranking=pivot_ranking, max_model_mb=max_model_mb,
                      count_cached_as_free=count_cached_as_free,
                      oversize_fallback=oversize_fallback,
                      num_beams=num_beams, max_new_tokens=max_new_tokens,
                      length_penalty=length_penalty,
                      no_repeat_ngram_size=no_repeat_ngram_size,
                      model_cache_size=model_cache_size,
                      max_loaded_mb=max_loaded_mb,
                      max_concurrent_translations=max_concurrent_translations,
                      fetch_on_demand=fetch_on_demand)
