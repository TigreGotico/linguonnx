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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from linguonnx.model_manager import list_models
from linguonnx.translate.decode import GenerationConfig
from linguonnx.translate.graph import (DEFAULT_PIVOT_PREFERENCE, Capability,
                                       Hop, NoRouteError, Route,
                                       TranslationGraph, normalize_tag)
from linguonnx.translate.models import TranslationModel, capability_from_entry

LOG = logging.getLogger(__name__)

__all__ = ["Translator", "load_translator", "Route", "Hop", "NoRouteError",
           "GenerationConfig"]


def _select_entries(precision: Optional[str], include_noncommercial: bool,
                    models: Optional[Sequence[str]]) -> Dict[str, dict]:
    registry = list_models(kind="translate")
    if models is not None:
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
                 num_beams: int = 4, max_new_tokens: int = 128,
                 length_penalty: float = 1.0, no_repeat_ngram_size: int = 0):
        self._entries = entries
        self.graph = TranslationGraph(
            [capability_from_entry(e) for e in entries.values()],
            pivot_preference=pivot_preference, prefer=prefer,
            max_hops=max_hops, max_routes=max_routes,
            pivot_ranking=pivot_ranking)
        self.generation = GenerationConfig(
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size)
        self._loaded: Dict[str, TranslationModel] = {}

    # -- introspection ----------------------------------------------------

    @property
    def available_languages(self) -> frozenset:
        """Every BCP-47 tag reachable as a source or a target, over all models."""
        return self.graph.languages

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
    def pivot_ranking(self) -> str:
        """The pivot ranking in force, ``"phonological"`` or ``"table"``."""
        return self.graph.pivot_ranking

    def route(self, src: str, tgt: str, max_hops: Optional[int] = None,
              prefer: Optional[str] = None) -> Route:
        """The best route, without translating. Raises :class:`NoRouteError`."""
        return self.graph.route(src, tgt, max_hops=max_hops, prefer=prefer)

    def routes(self, src: str, tgt: str, max_hops: Optional[int] = None,
               prefer: Optional[str] = None,
               limit: Optional[int] = None) -> List[Route]:
        """Every viable route, ranked best-first. Bounded; see the graph docstring."""
        return self.graph.routes(src, tgt, max_hops=max_hops, prefer=prefer,
                                 limit=limit)

    def can_translate(self, src: str, tgt: str,
                      max_hops: Optional[int] = None) -> bool:
        return self.graph.can_translate(src, tgt, max_hops=max_hops)

    # -- models -----------------------------------------------------------

    def model(self, model_id: str) -> TranslationModel:
        if model_id not in self._loaded:
            entry = self._entries.get(model_id)
            self._loaded[model_id] = TranslationModel(model_id, entry)
        return self._loaded[model_id]

    # -- translation ------------------------------------------------------

    def translate(self, text: str, src: Optional[str] = None,
                  tgt: Optional[str] = None, *,
                  route: Optional[Route] = None,
                  model: Optional[str] = None,
                  return_route: bool = False,
                  max_hops: Optional[int] = None,
                  prefer: Optional[str] = None,
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

        if model is not None:
            chosen = self._pinned_route(model, src, tgt)
        elif route is not None:
            chosen = route
        else:
            if src is None or tgt is None:
                raise ValueError("give src= and tgt=, or a route=, or a model=")
            chosen = self.route(src, tgt, max_hops=max_hops, prefer=prefer)

        out = text
        for hop in chosen.hops:
            out = self.model(hop.model_id).translate(
                out, hop.src, hop.tgt, config=config, target_token=target_token)
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
                    pivot_preference: Sequence[str] = DEFAULT_PIVOT_PREFERENCE,
                    max_routes: int = 10,
                    pivot_ranking: str = "auto",
                    num_beams: int = 4,
                    max_new_tokens: int = 128,
                    length_penalty: float = 1.0,
                    no_repeat_ngram_size: int = 0) -> Translator:
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
    :param pivot_ranking: how to order pivot candidates for a 2-hop route.
        ``"auto"`` (default) ranks them by phonological distance when the
        optional ``orthography2ipa`` package is installed - ``pip install
        linguonnx[distance]`` - and by the curated table otherwise.
        ``"phonological"`` demands the package and raises without it;
        ``"table"`` ignores it. ``Route.pivot_basis`` reports which was used.
    :param num_beams: 4 by default; 1 is greedy and about 4x faster.
    """
    if model is not None:
        models = [model] if models is None else list(models) + [model]
    entries = _select_entries(precision, include_noncommercial, models)
    if not entries:
        raise ValueError("no translation models matched the given filters")
    return Translator(entries, prefer=prefer, max_hops=max_hops,
                      pivot_preference=pivot_preference, max_routes=max_routes,
                      pivot_ranking=pivot_ranking,
                      num_beams=num_beams, max_new_tokens=max_new_tokens,
                      length_penalty=length_penalty,
                      no_repeat_ngram_size=no_repeat_ngram_size)
