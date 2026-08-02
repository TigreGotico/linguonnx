"""Routing layer: which model (or chain of models) translates src -> tgt.

The graph is *not* a materialised pair list. Nodes are BCP-47 language tags,
and an edge is a **capability**: a claim by one model that it can translate
some set of directions.

- A **bilingual** model (Marian / opus-mt) is one directed edge, ``en -> pt``.
- A **multilingual** model (M2M100, NLLB) is a *covering set*: it declares the
  languages it knows and is any-to-any inside that set. NLLB's 202 languages
  would be 40602 directed edges if written out; instead the set is stored once
  and expanded lazily when a pair is resolved.

:meth:`TranslationGraph.route` returns a :class:`Route`, an ordered list of
:class:`Hop` objects. Nothing here loads a model or translates anything.

Cost model
----------

Every candidate route is scored by a **tuple**, and the policy decides the
order of the tuple's elements, not which code path runs. Two policies ship:

``prefer="fewest_hops"`` (default)
    ``(n_hops, n_multilingual_hops, licence_tier, total_size_mb, model_ids)``
``prefer="dedicated"``
    ``(n_multilingual_hops, n_hops, licence_tier, total_size_mb, model_ids)``

Under ``fewest_hops`` a single multilingual hop always beats a two-hop
bilingual chain. Under ``dedicated`` the two-hop chain of dedicated bilingual
models wins instead. Which is actually better output is **unmeasured** for this
model set; ``fewest_hops`` is the default because it is the conservative
choice - it never doubles latency or compounds error unless asked to.

For the *same* pair at the *same* hop count a dedicated bilingual model always
beats a multilingual one, under both policies.

Search bounds
-------------

The graph is mostly cliques, so a naive breadth-first search over materialised
nodes would consider ~200 pivots at 2 hops and ~40000 at 3. Two bounds stop
that:

1. Pivot candidates are restricted to the configured pivot-preference list
   plus every language that is an endpoint of a *dedicated* edge. A pivot that
   is only reachable through the same multilingual model that already covers
   the direct pair can never help, so it is pruned.
2. Each leg of a multi-hop route keeps at most its best dedicated candidate and
   its best multilingual candidate, not every model that could serve the leg.

Pivot ranking
-------------

The bounded candidate list is *ordered* by ``pivot_ranking``. ``"table"`` uses
:data:`REGIONAL_PIVOTS` then the preference list; ``"phonological"`` reorders
the same candidates by measured linguistic distance, see
:mod:`linguonnx.translate.distance`. ``"auto"``, the default, picks
phonological when the optional ``orthography2ipa`` package is installed. The
ranking never adds or drops a candidate, so the search bounds above hold either
way, and :attr:`Route.pivot_basis` says which ranking produced a route.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import langcodes

from linguonnx.translate import distance as _distance

__all__ = [
    "NoRouteError",
    "Capability",
    "Hop",
    "Route",
    "TranslationGraph",
    "DEFAULT_PIVOT_PREFERENCE",
    "REGIONAL_PIVOTS",
    "PIVOT_RANKINGS",
    "LICENSE_TIERS",
    "normalize_tag",
]


class NoRouteError(LookupError):
    """No chain of models within ``max_hops`` connects the two languages."""


# Licence tiers, low number = fewer strings attached. A route is only as
# permissive as its most restrictive hop, so the route tier is the max.
LICENSE_TIERS: Dict[str, int] = {
    "permissive": 0,
    "share-alike": 1,
    "non-commercial": 2,
}

#: Ordered pivot preference. English is a good pivot for most of the world
#: because that is where the bilingual training data is, but it is a poor
#: pivot inside the Iberian peninsula, where Spanish keeps far more of the
#: morphology and lexicon. The list is data, not policy baked into the search:
#: pass ``pivot_preference=`` to reorder it, and any language that is an
#: endpoint of a dedicated edge is considered as a pivot anyway, just later.
DEFAULT_PIVOT_PREFERENCE: Tuple[str, ...] = ("es", "en", "fr", "de", "ru", "zh", "ar")

#: Languages that prefer a non-English pivot, and which one. Consulted only to
#: reorder the pivot preference for a specific pair; a pivot that is not in the
#: graph is silently skipped.
REGIONAL_PIVOTS: Dict[str, Tuple[str, ...]] = {
    # Iberian romance + Basque: Spanish keeps more than English does.
    "pt": ("es",), "gl": ("es",), "ca": ("es",), "ast": ("es",),
    "eu": ("es",), "an": ("es",), "oc": ("es", "fr"),
    # Continental romance
    "it": ("es", "fr"), "ro": ("it", "fr"), "co": ("it",), "sc": ("it",),
    # Germanic
    "nl": ("de",), "af": ("nl",), "da": ("sv", "de"), "sv": ("da", "de"),
    "nb": ("sv", "da"), "no": ("sv", "da"), "nn": ("no", "sv"),
    # Slavic
    "uk": ("ru",), "be": ("ru",), "bg": ("ru",), "sr": ("hr", "ru"),
    "hr": ("sr",), "bs": ("hr", "sr"), "sk": ("cs",), "cs": ("sk",),
}


#: Accepted values for ``pivot_ranking``. ``"auto"`` uses phonological
#: distance when :mod:`orthography2ipa` is installed and the curated table
#: order otherwise, so the extra dependency changes quality, never behaviour.
PIVOT_RANKINGS: Tuple[str, ...] = ("auto", "phonological", "table")


#: Model codes that are not valid language tags in any standard. M2M100 calls
#: Northern Sotho ``ns``, which is a two-letter code ISO never assigned.
MODEL_CODE_ALIASES: Dict[str, str] = {"ns": "nso"}


def normalize_tag(tag: str) -> str:
    """Normalise any language tag shape to the graph's BCP-47 node name.

    Accepts FLORES/GlotLID ``por_Latn`` as readily as ``pt``, ``pt-BR`` or
    ``POR``, and returns the same node name for all of them. Unparseable input
    is returned lowercased rather than raising, so an exotic registry code
    still gets *a* node instead of crashing the graph.
    """
    from linguonnx.detect.labels import to_bcp47

    tag = tag.strip()
    if not tag:
        raise ValueError("empty language tag")
    tag = MODEL_CODE_ALIASES.get(tag, tag)
    try:
        if "_" in tag:
            return to_bcp47(tag)
        return to_bcp47(tag) if len(tag) <= 3 and tag.isalpha() else \
            langcodes.standardize_tag(tag)
    except Exception:
        return tag.lower()


@dataclass(frozen=True)
class Capability:
    """What one model claims it can translate.

    ``pair`` set  -> a bilingual model, exactly one directed edge.
    ``pair`` None -> a multilingual model, any-to-any across ``languages``.
    """

    model_id: str
    arch: str
    license: str
    license_tier: str
    size_mb: int
    languages: FrozenSet[str] = frozenset()
    pair: Optional[Tuple[str, str]] = None

    @property
    def dedicated(self) -> bool:
        """True when the model *is* the pair, rather than covering it."""
        return self.pair is not None

    @property
    def tier_rank(self) -> int:
        return LICENSE_TIERS.get(self.license_tier, max(LICENSE_TIERS.values()))

    def covers(self, src: str, tgt: str) -> bool:
        if self.pair is not None:
            return self.pair == (src, tgt)
        return src != tgt and src in self.languages and tgt in self.languages

    def endpoints(self) -> FrozenSet[str]:
        if self.pair is not None:
            return frozenset(self.pair)
        return self.languages


@dataclass(frozen=True)
class Hop:
    """One model doing one directed language pair."""

    model_id: str
    src: str
    tgt: str
    arch: str
    license: str
    license_tier: str
    size_mb: int
    dedicated: bool

    def __str__(self) -> str:
        kind = "dedicated" if self.dedicated else "multilingual"
        return f"{self.src}->{self.tgt} via {self.model_id} ({kind}, {self.license})"


@dataclass(frozen=True)
class Route:
    """An ordered chain of hops from ``src`` to ``tgt``, plus the policy that chose it.

    A Route is always handed back to the caller, so a multi-hop translation is
    never silent: ``len(route.hops) > 1`` says a pivot happened and
    ``route.pivots`` says through what.
    """

    src: str
    tgt: str
    hops: Tuple[Hop, ...]
    prefer: str = "fewest_hops"
    max_hops: int = 2
    #: Position of each pivot in the graph's preference list. Empty for a
    #: direct route. Used to break ties between equally-costed pivots, so a
    #: linguistically closer pivot wins over an alphabetically earlier model id.
    pivot_rank: Tuple[int, ...] = ()
    #: What produced the pivot ordering behind ``pivot_rank``:
    #: ``"phonological"`` (orthography2ipa distances) or ``"table"`` (the
    #: curated :data:`REGIONAL_PIVOTS` plus the preference list). Reported even
    #: on direct routes, where it simply says what *would* have been used.
    pivot_basis: str = "table"

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    @property
    def pivots(self) -> Tuple[str, ...]:
        return tuple(hop.tgt for hop in self.hops[:-1])

    @property
    def model_ids(self) -> Tuple[str, ...]:
        return tuple(hop.model_id for hop in self.hops)

    @property
    def licenses(self) -> Tuple[str, ...]:
        return tuple(hop.license for hop in self.hops)

    @property
    def license_tier(self) -> str:
        """The most restrictive tier on the route - a chain is only as free as its worst hop."""
        worst = max(self.hops, key=lambda h: LICENSE_TIERS.get(h.license_tier, 99))
        return worst.license_tier

    @property
    def total_size_mb(self) -> int:
        return sum(hop.size_mb for hop in self.hops)

    @property
    def n_multilingual_hops(self) -> int:
        return sum(0 if hop.dedicated else 1 for hop in self.hops)

    def __str__(self) -> str:
        chain = " | ".join(str(h) for h in self.hops)
        return f"[{self.src}->{self.tgt}, {self.n_hops} hop(s), prefer={self.prefer}] {chain}"


def _route_key(route: Route, prefer: str) -> tuple:
    """Score a route. Lower sorts first. The policy reorders the tuple only."""
    hops = route.n_hops
    multi = route.n_multilingual_hops
    tier = max(LICENSE_TIERS.get(h.license_tier, 99) for h in route.hops)
    size = route.total_size_mb
    tail = (tier, route.pivot_rank, size, route.model_ids)
    if prefer == "dedicated":
        return (multi, hops) + tail
    if prefer == "fewest_hops":
        return (hops, multi) + tail
    raise ValueError(
        f"unknown routing policy {prefer!r}; use 'fewest_hops' or 'dedicated'")


class TranslationGraph:
    """Resolve ``src -> tgt`` into a :class:`Route` over the registered models."""

    def __init__(
        self,
        capabilities: Iterable[Capability],
        pivot_preference: Sequence[str] = DEFAULT_PIVOT_PREFERENCE,
        prefer: str = "fewest_hops",
        max_hops: int = 2,
        max_routes: int = 10,
        pivot_ranking: str = "auto",
    ):
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        _route_key(  # validate the policy name eagerly, not on first route()
            Route("a", "b", (Hop("x", "a", "b", "marian", "MIT", "permissive", 1, True),)),
            prefer,
        )
        self.capabilities: Tuple[Capability, ...] = tuple(capabilities)
        self.pivot_preference = tuple(normalize_tag(p) for p in pivot_preference)
        self.prefer = prefer
        self.max_hops = max_hops
        self.max_routes = max_routes
        if pivot_ranking not in PIVOT_RANKINGS:
            raise ValueError(
                f"unknown pivot_ranking {pivot_ranking!r}; "
                f"use one of {', '.join(PIVOT_RANKINGS)}")
        if pivot_ranking == "auto":
            pivot_ranking = "phonological" if _distance.available() else "table"
        elif pivot_ranking == "phonological" and not _distance.available():
            raise ValueError(
                "pivot_ranking='phonological' needs orthography2ipa; "
                "install linguonnx[distance], or use pivot_ranking='auto'")
        #: The ranking actually in force, never ``"auto"``.
        self.pivot_ranking = pivot_ranking

        self._bilingual: Dict[Tuple[str, str], List[Capability]] = {}
        self._multilingual: List[Capability] = []
        self._dedicated_endpoints: set = set()
        for cap in self.capabilities:
            if cap.dedicated:
                self._bilingual.setdefault(cap.pair, []).append(cap)
                self._dedicated_endpoints.update(cap.pair)
            else:
                self._multilingual.append(cap)

        self._languages = frozenset().union(
            *(cap.endpoints() for cap in self.capabilities)) if self.capabilities else frozenset()

    # -- introspection ----------------------------------------------------

    @property
    def languages(self) -> FrozenSet[str]:
        """Every language any registered model can read or write."""
        return self._languages

    def capabilities_for(self, src: str, tgt: str) -> List[Capability]:
        """Every model that can do this exact pair in one hop, best first."""
        cands = list(self._bilingual.get((src, tgt), ()))
        cands += [c for c in self._multilingual if c.covers(src, tgt)]
        return sorted(cands, key=self._cap_key)

    @staticmethod
    def _cap_key(cap: Capability) -> tuple:
        # Within one leg: dedicated first, then permissive, then small.
        return (0 if cap.dedicated else 1, cap.tier_rank, cap.size_mb, cap.model_id)

    # -- pivots -----------------------------------------------------------

    def _pivot_candidates(self, src: str, tgt: str) -> List[str]:
        """Bounded, ordered pivot list. Never "every language in the graph".

        Order: pair-specific regional pivots, then the configured global
        preference, then any remaining language that is an endpoint of a
        dedicated edge. Languages reachable *only* through a multilingual model
        are excluded - pivoting through one of those uses the same model that
        already covers the direct pair, so it can never be an improvement.
        """
        ordered: List[str] = []
        seen = {src, tgt}

        def add(lang: str) -> None:
            if lang not in seen and lang in self._languages:
                seen.add(lang)
                ordered.append(lang)

        for lang in REGIONAL_PIVOTS.get(tgt, ()):
            add(lang)
        for lang in REGIONAL_PIVOTS.get(src, ()):
            add(lang)
        for lang in self.pivot_preference:
            add(lang)
        for lang in sorted(self._dedicated_endpoints):
            add(lang)
        if self.pivot_ranking == "phonological":
            ordered = self._rank_phonologically(src, tgt, ordered)
        return ordered

    @staticmethod
    def _rank_phonologically(src: str, tgt: str,
                             ordered: List[str]) -> List[str]:
        """Reorder an existing candidate list by ``src->pivot->tgt`` distance.

        The score is ``(max(leg, leg), sum(legs), table_index)``, worst leg
        first. Output through a pivot is bottlenecked by the **worse** of the
        two legs, so a candidate that is very close to the source cannot buy
        its way past a bad second leg: ranking on the sum alone would put
        ``gl`` ahead of ``en`` for ``es->ru``, because ``es->gl`` is tiny. The
        sum breaks ties as total path cost, and the table position breaks the
        rest, so the order stays stable.

        The candidate *set* is untouched - this only changes the order, so the
        search stays exactly as bounded as it was. A pivot that
        :mod:`orthography2ipa` does not know scores ``None``, which keeps its
        table position among the other unknowns and sorts it after every
        candidate that does have a distance. Unknown is not zero.
        """
        def key(item: Tuple[int, str]) -> tuple:
            index, lang = item
            first = _distance.pair_distance(src, lang)
            second = _distance.pair_distance(lang, tgt)
            if first is None or second is None:
                return (1, float(index), 0.0, index)
            return (0, max(first, second), first + second, index)

        return [lang for _, lang in sorted(enumerate(ordered), key=key)]

    # -- route enumeration ------------------------------------------------

    def _legs(self, src: str, tgt: str) -> List[Capability]:
        """At most one dedicated and one multilingual candidate for a leg."""
        best: List[Capability] = []
        dedicated = [c for c in self._bilingual.get((src, tgt), ())]
        if dedicated:
            best.append(min(dedicated, key=self._cap_key))
        multi = [c for c in self._multilingual if c.covers(src, tgt)]
        if multi:
            best.append(min(multi, key=self._cap_key))
        return best

    @staticmethod
    def _hop(cap: Capability, src: str, tgt: str) -> Hop:
        return Hop(model_id=cap.model_id, src=src, tgt=tgt, arch=cap.arch,
                   license=cap.license, license_tier=cap.license_tier,
                   size_mb=cap.size_mb, dedicated=cap.dedicated)

    def _enumerate(self, src: str, tgt: str, max_hops: int, prefer: str,
                   short_circuit: bool) -> List[Route]:
        routes: List[Route] = []

        for cap in self.capabilities_for(src, tgt):
            routes.append(Route(src, tgt, (self._hop(cap, src, tgt),),
                                prefer=prefer, max_hops=max_hops,
                                pivot_basis=self.pivot_ranking))

        # Under fewest_hops a direct route can never be beaten by a longer one,
        # so enumerating pivots would be pure waste on the hot path.
        if routes and short_circuit and prefer == "fewest_hops":
            return routes
        if max_hops < 2:
            return routes

        direct_multi = {c.model_id for c in self._multilingual if c.covers(src, tgt)}
        pivots = self._pivot_candidates(src, tgt)

        rank_of = {lang: i for i, lang in enumerate(pivots)}
        for path in self._pivot_paths(pivots, max_hops - 1):
            nodes = (src,) + path + (tgt,)
            pivot_rank = tuple(rank_of[p] for p in path)
            legs = [self._legs(a, b) for a, b in zip(nodes, nodes[1:])]
            if any(not leg for leg in legs):
                continue
            for combo in itertools.product(*legs):
                ids = {c.model_id for c in combo}
                # Pivoting inside a single multilingual model that already does
                # the pair directly is strictly worse. Prune it.
                if len(ids) == 1 and ids & direct_multi:
                    continue
                hops = tuple(self._hop(cap, a, b)
                             for cap, a, b in zip(combo, nodes, nodes[1:]))
                routes.append(Route(src, tgt, hops, prefer=prefer,
                                    max_hops=max_hops, pivot_rank=pivot_rank,
                                    pivot_basis=self.pivot_ranking))
        return routes

    @staticmethod
    def _pivot_paths(pivots: Sequence[str], depth: int):
        """All ordered pivot tuples of length 1..depth, in preference order."""
        for n in range(1, depth + 1):
            for combo in itertools.permutations(pivots, n):
                yield combo

    # -- public API -------------------------------------------------------

    def routes(self, src: str, tgt: str, max_hops: Optional[int] = None,
               prefer: Optional[str] = None,
               limit: Optional[int] = None) -> List[Route]:
        """Every viable route, ranked best-first under the active policy.

        Bounded on purpose: returns at most ``limit`` routes (default
        ``max_routes``, 10). Each leg of a multi-hop route contributes at most
        its best dedicated and best multilingual candidate, so this is a
        curated ranking rather than an exhaustive product of every model
        combination. Returns ``[]`` rather than raising - use :meth:`route`
        when you want ``NoRouteError``.
        """
        src, tgt = normalize_tag(src), normalize_tag(tgt)
        prefer = prefer or self.prefer
        max_hops = self.max_hops if max_hops is None else max_hops
        limit = self.max_routes if limit is None else limit
        if src == tgt:
            return []
        found = self._enumerate(src, tgt, max_hops, prefer, short_circuit=False)
        found.sort(key=lambda r: _route_key(r, prefer))
        # Dedupe identical hop chains that different pivot orders produced.
        out, seen = [], set()
        for route in found:
            sig = tuple((h.model_id, h.src, h.tgt) for h in route.hops)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(route)
            if len(out) >= limit:
                break
        return out

    def route(self, src: str, tgt: str, max_hops: Optional[int] = None,
              prefer: Optional[str] = None) -> Route:
        """The single best route, or raise :class:`NoRouteError`."""
        raw_src, raw_tgt = src, tgt
        src, tgt = normalize_tag(src), normalize_tag(tgt)
        prefer = prefer or self.prefer
        max_hops = self.max_hops if max_hops is None else max_hops
        if src == tgt:
            raise NoRouteError(
                f"source and target are the same language ({src!r}); nothing to translate")
        found = self._enumerate(src, tgt, max_hops, prefer, short_circuit=True)
        if not found:
            hint = ""
            if max_hops == 1:
                hint = " (max_hops=1: only direct models were considered)"
            raise NoRouteError(
                f"no route from {raw_src!r} to {raw_tgt!r} within {max_hops} hop(s){hint}")
        return min(found, key=lambda r: _route_key(r, prefer))

    def can_translate(self, src: str, tgt: str, max_hops: Optional[int] = None) -> bool:
        try:
            self.route(src, tgt, max_hops=max_hops)
            return True
        except NoRouteError:
            return False
