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

Coverage and runnability
------------------------

A capability claims languages; it also says whether this library can *run* the
model at all. Some registered architectures need a preprocessing pipeline that
is not vendored, and ``translate()`` raises for them. Those capabilities are
excluded from routing, so :meth:`TranslationGraph.can_translate`,
:meth:`TranslationGraph.route` and :attr:`TranslationGraph.languages` answer
for the same models ``translate()`` will use - a ``True`` followed by
``NotImplementedError`` is worse than a ``False``, because the caller asked
first. Which models those are is registry data, never a list of architecture
names here: see :func:`entry_runnability`.

Size budget
-----------

``max_model_mb`` bounds the size of a single model on a route. A capability
over the budget is excluded the same way an unrunnable one is - kept for the
error message, absent from :meth:`TranslationGraph.route`,
:meth:`TranslationGraph.routes`, :meth:`TranslationGraph.can_translate` and
:attr:`TranslationGraph.languages` - so a pair that one 4.9 GB multilingual
model served in one hop is served by a chain of small bilingual models instead
of failing.

The budget is on the **download**, so ``count_cached_as_free=True`` (the
default) exempts a model already in the local cache: refusing something that
costs no fetch would lose coverage and save nothing. Set it false to bound the
model rather than the fetch. Both are per-call overridable, like ``max_hops``.

Every structure the search reads is derived in :func:`_build_index`, from one
capability list, so the budget cannot be applied at some points of the search
and forgotten at others - which is how ``endpoints()`` and ``covers()`` once
came to disagree.

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
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (Dict, FrozenSet, Iterable, List, Optional, Sequence,
                    Tuple, Union)

import langcodes

from linguonnx.limits import MAX_MODEL_MB
from linguonnx.translate import distance as _distance

LOG = logging.getLogger(__name__)

__all__ = [
    "NoRouteError",
    "MalformedTagError",
    "InvalidRouteError",
    "Capability",
    "Hop",
    "Route",
    "TranslationGraph",
    "DEFAULT_PIVOT_PREFERENCE",
    "REGIONAL_PIVOTS",
    "PIVOT_RANKINGS",
    "LICENSE_TIERS",
    "UNSET",
    "normalize_tag",
    "entry_runnability",
]


class NoRouteError(LookupError):
    """No chain of models within ``max_hops`` connects the two languages."""


class MalformedTagError(ValueError):
    """A caller passed something that is not a language tag at all.

    Distinct from :class:`NoRouteError` on purpose. ``"kea"`` with no model to
    serve it and ``"!!!"`` from an unvalidated HTTP query parameter are
    different failures, and answering both with "no route" tells the caller
    their language is unsupported when in fact their input was junk.
    """


class InvalidRouteError(ValueError):
    """A caller-supplied :class:`Route` is not executable as written."""


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
MODEL_CODE_ALIASES: Dict[str, str] = {
    "ns": "nso",
    # Projecte Aina added a token for Aranese to NLLB and called it
    # "arn_Latn" - which ISO already assigns to Mapudungun. Mapped here so the
    # routing graph sees Occitan and never offers the model for Mapudungun.
    "arn_Latn": "oc",
}


def normalize_tag(tag: str, strict: bool = False) -> str:
    """Normalise any language tag shape to the graph's BCP-47 node name.

    Accepts FLORES/GlotLID ``por_Latn`` as readily as ``pt`` or ``POR``, and
    returns the same node name for all of them.

    Two callers with opposite needs share this function, so unparseable input
    has two answers:

    ``strict=False`` (default), the *registry* path
        Return the tag lowercased and log a warning. An exotic model code that
        no standard knows still gets *a* node, which keeps the model routable
        instead of crashing graph construction over one entry.
    ``strict=True``, the *caller-input* path
        Raise :class:`MalformedTagError`. A lowercased pseudo-node matches
        nothing in the graph, so the lenient answer would reach the caller as
        ``NoRouteError`` - "this language is unsupported" - when the truth is
        "this was not a language tag".
    """
    from linguonnx.detect.labels import to_bcp47

    tag = tag.strip()
    if not tag:
        raise MalformedTagError("empty language tag")
    tag = MODEL_CODE_ALIASES.get(tag, tag)
    try:
        if "_" in tag:
            return to_bcp47(tag)
        return to_bcp47(tag) if len(tag) <= 3 and tag.isalpha() else \
            langcodes.standardize_tag(tag)
    except Exception as err:
        if strict:
            raise MalformedTagError(
                f"{tag!r} is not a usable language tag: {err}") from err
        LOG.warning("%r is not a parseable language tag; using %r as a graph "
                    "node as-is", tag, tag.lower())
        return tag.lower()


def entry_runnability(entry: Dict) -> Tuple[bool, Optional[str]]:
    """``(runnable, reason)`` for one ``translate.json`` entry.

    Runnability is **data**, not a list of architecture names kept in this
    module. A model is runnable unless its registry entry says otherwise, so a
    preprocessing pipeline that lands later removes the flag from the registry
    and the model starts routing again with no change here.

    One rule is enforced on top of the flag, because it guards a failure that
    produces fluent output instead of an error: a Marian model that serves
    several targets from one decoder selects the target with a prefix token,
    and with no token it answers in whichever language it likes. Such an entry
    is refused rather than trusted.
    """
    if entry.get("runnable") is False:
        return False, entry.get("unrunnable_reason") or "the registry marks it unrunnable"
    multi_target = entry["arch"] == "marian" and not entry.get("pair")
    if multi_target and not (entry.get("target_token")
                             or entry.get("target_token_template")):
        LOG.warning(
            "%s is a multi-target Marian model with no target_token and no "
            "target_token_template; it cannot select a target language and "
            "would translate into an arbitrary one. Excluded from routing.",
            entry["model_id"])
        return False, ("multi-target Marian model with no target token; it "
                       "cannot select which language it translates into")
    return True, None


@lru_cache(maxsize=None)
def _registry_runnability(model_id: str) -> Tuple[bool, Optional[str]]:
    """Runnability of a model the caller did not state it for.

    A :class:`Capability` built outside the registry (a test, a caller's own
    model) is runnable: only the registry can say otherwise, and it says so on
    the entry.
    """
    from linguonnx.model_manager import list_models

    entry = list_models(kind="translate").get(model_id)
    if entry is None:
        return True, None
    return entry_runnability(entry)


@dataclass(frozen=True)
class Capability:
    """What one model claims it can translate.

    ``pair`` set  -> a bilingual model, exactly one directed edge.
    ``pair`` None, ``src_languages``/``tgt_languages`` both None
                  -> a multilingual model, any-to-any across ``languages``.
    ``pair`` None, ``src_languages``/``tgt_languages`` set
                  -> a *directional* multilingual model: it only translates
                  from the source set into the target set, never backwards.
                  IndicTrans2 ships three such models - ``en-indic`` only goes
                  ``eng_Latn -> {22 Indic tags}``, ``indic-en`` only the
                  reverse, and ``indic-indic`` is symmetric (both sets equal).
                  Treating a one-directional model as any-to-any would let the
                  router propose an impossible hop that fails at runtime.

    A capability also says whether it can be **run**, not only what it covers.
    Coverage is what the weights know; runnability is whether this library can
    drive them end to end. The two are separate because a model can be a
    correct claim about languages and still have no inference pipeline here.
    ``runnable=None``, the default, means "not stated" and is resolved from the
    registry entry, so :meth:`TranslationGraph.route` and ``translate()`` agree
    without every caller having to look the entry up.
    """

    model_id: str
    arch: str
    license: str
    license_tier: str
    size_mb: int
    languages: FrozenSet[str] = frozenset()
    pair: Optional[Tuple[str, str]] = None
    src_languages: Optional[FrozenSet[str]] = None
    tgt_languages: Optional[FrozenSet[str]] = None
    #: ``None`` = not stated, ask the registry. ``True``/``False`` = stated.
    runnable: Optional[bool] = None
    #: Why it cannot run, for the error the caller finally sees.
    unrunnable_reason: Optional[str] = None

    @property
    def is_cached(self) -> bool:
        """Whether every file this model needs is already on disk.

        Read through the module rather than imported by name, so a caller (or
        a test) that replaces :func:`linguonnx.model_manager.is_cached` is
        honoured.
        """
        from linguonnx import model_manager

        return model_manager.is_cached(self.model_id, kind="translate")

    def within_size_cap(self, max_model_mb: Optional[int],
                        count_cached_as_free: bool = True) -> bool:
        """Whether this model may be used under a per-model size budget.

        The budget is on the **download**, so a model already in the cache
        passes however big it is when ``count_cached_as_free`` is true. Set it
        false to bound the model itself rather than the fetch.
        """
        if max_model_mb is None or self.size_mb <= max_model_mb:
            return True
        return count_cached_as_free and self.is_cached

    @property
    def is_runnable(self) -> bool:
        """Whether :meth:`TranslationModel.translate` can actually execute this."""
        if self.runnable is not None:
            return self.runnable
        return _registry_runnability(self.model_id)[0]

    @property
    def unrunnable_because(self) -> Optional[str]:
        """The reason :attr:`is_runnable` is false, or ``None`` when it is true."""
        if self.is_runnable:
            return None
        if self.runnable is False:
            return self.unrunnable_reason or "no reason recorded"
        return _registry_runnability(self.model_id)[1] or "no reason recorded"

    @property
    def dedicated(self) -> bool:
        """True when the model *is* the pair, rather than covering it."""
        return self.pair is not None

    @property
    def directional(self) -> bool:
        """True for a covering-set model that cannot be run backwards."""
        return self.pair is None and (
            self.src_languages is not None or self.tgt_languages is not None)

    @property
    def tier_rank(self) -> int:
        return LICENSE_TIERS.get(self.license_tier, max(LICENSE_TIERS.values()))

    def covers(self, src: str, tgt: str) -> bool:
        if self.pair is not None:
            return self.pair == (src, tgt)
        if src == tgt:
            return False
        if self.directional:
            return src in self._sources and tgt in self._targets
        return src in self.languages and tgt in self.languages

    def endpoints(self) -> FrozenSet[str]:
        if self.pair is not None:
            return frozenset(self.pair)
        if self.directional:
            # The same fallback :meth:`covers` uses: a capability that declares
            # one side only takes the other from ``languages``. Without the
            # fallback here the graph would serve a hop into a language it does
            # not list, and `available_languages` would disagree with `route`.
            return self._sources | self._targets
        return self.languages

    @property
    def _sources(self) -> FrozenSet[str]:
        return self.src_languages if self.src_languages is not None else self.languages

    @property
    def _targets(self) -> FrozenSet[str]:
        return self.tgt_languages if self.tgt_languages is not None else self.languages


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
    def models_size_mb(self) -> int:
        """Disk cost of the *distinct* models on the route.

        Differs from :attr:`total_size_mb` when one model serves two hops: it
        is downloaded once, so it is counted once here.
        ``total_size_mb`` stays a per-hop sum because it is a cost proxy in
        the route ranking, where a model used twice really is used twice.
        """
        return sum(size for _, size, _ in self._models())

    @property
    def cached_size_mb(self) -> int:
        """How much of :attr:`models_size_mb` is already on disk.

        Read at the moment it is asked for, so a route inspected after a
        ``prefetch`` reports the new state. Each check is a handful of
        ``stat`` calls; nothing is downloaded and the hub is never contacted.
        """
        return sum(size for _, size, cached in self._models() if cached)

    @property
    def download_size_mb(self) -> int:
        """What running this route would have to fetch, in MB.

        The number a caller comparing two routes from :meth:`routes` actually
        needs: a three-hop chain of models it already holds costs nothing,
        while a one-hop route through a 4.9 GB model it does not costs 4.9 GB.
        Size alone cannot see that difference.
        """
        return sum(size for _, size, cached in self._models() if not cached)

    def _models(self) -> Tuple[Tuple[str, int, bool], ...]:
        """``(model_id, size_mb, cached)`` once per distinct model, in order."""
        from linguonnx import model_manager

        seen, out = set(), []
        for hop in self.hops:
            if hop.model_id in seen:
                continue
            seen.add(hop.model_id)
            out.append((hop.model_id, hop.size_mb,
                        model_manager.is_cached(hop.model_id, kind="translate")))
        return tuple(out)

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


class _Unset:
    """Type of :data:`UNSET`."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


#: Sentinel for "the caller said nothing", where ``None`` is already an
#: answer. ``max_model_mb=None`` means *no cap*, so it cannot double as
#: "inherit", the way ``max_hops=None`` can.
UNSET = _Unset()


@dataclass(frozen=True)
class _Index:
    """The searchable form of one capability set.

    Every structure the search reads is derived here, from one list, so a
    capability cannot be visible to :meth:`TranslationGraph.route` and invisible
    to :attr:`TranslationGraph.languages` (or the reverse). A size cap produces
    a *different index*, not a filter applied at some points of the search and
    forgotten at others.
    """

    bilingual: Dict[Tuple[str, str], List[Capability]]
    multilingual: Tuple[Capability, ...]
    dedicated_endpoints: Tuple[str, ...]
    languages: FrozenSet[str]
    #: Left out by the size cap. Kept, like the unrunnable ones, so the error
    #: can name them.
    oversized: Tuple[Capability, ...]
    #: The budget this index was built under, so an error can quote the cap
    #: that actually applied rather than the graph's own default.
    max_model_mb: Optional[int]
    count_cached_as_free: bool


def _build_index(capabilities: Iterable[Capability],
                 max_model_mb: Optional[int],
                 count_cached_as_free: bool) -> _Index:
    bilingual: Dict[Tuple[str, str], List[Capability]] = {}
    multilingual: List[Capability] = []
    endpoints: set = set()
    kept: List[Capability] = []
    oversized: List[Capability] = []
    for cap in capabilities:
        if not cap.within_size_cap(max_model_mb, count_cached_as_free):
            oversized.append(cap)
            continue
        kept.append(cap)
        if cap.dedicated:
            bilingual.setdefault(cap.pair, []).append(cap)
            endpoints.update(cap.pair)
        else:
            multilingual.append(cap)
    languages = frozenset().union(*(c.endpoints() for c in kept)) if kept \
        else frozenset()
    return _Index(bilingual=bilingual, multilingual=tuple(multilingual),
                  dedicated_endpoints=tuple(sorted(endpoints)),
                  languages=languages, oversized=tuple(oversized),
                  max_model_mb=max_model_mb,
                  count_cached_as_free=count_cached_as_free)


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
        max_model_mb: Union[int, None, "_Unset"] = UNSET,
        count_cached_as_free: bool = True,
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

        # A capability that cannot be executed is kept for error messages and
        # excluded from everything else. `can_translate` exists so a caller can
        # ask "will this work" before committing, so it has to answer for the
        # same thing `translate` will do, not for what the registry lists.
        self._runnable: Tuple[Capability, ...] = tuple(
            cap for cap in self.capabilities if cap.is_runnable)
        #: Registered but not executable; named in :class:`NoRouteError`.
        self.unrunnable_capabilities: Tuple[Capability, ...] = tuple(
            cap for cap in self.capabilities if not cap.is_runnable)

        self._by_model: Dict[str, Capability] = {
            cap.model_id: cap for cap in self.capabilities}
        self.max_model_mb = self._check_max_model_mb(max_model_mb)
        self.count_cached_as_free = count_cached_as_free
        self._indexes: Dict[Tuple[Optional[int], bool], _Index] = {}
        self._index = self._index_for(self.max_model_mb,
                                      self.count_cached_as_free)
        # Every tag any runnable model names, cap or no cap. Used to resolve a
        # caller's tag onto a node, which must not change with the size budget:
        # `pt-BR` means `pt` whether or not `pt` is currently routable.
        self._addressable = self._index_for(None, True).languages

    @staticmethod
    def _check_max_model_mb(value: Union[int, None, "_Unset"]) -> Optional[int]:
        """Resolve the size cap, with the environment as the default.

        ``UNSET`` (the default) reads ``LINGUONNX_MAX_MODEL_MB``; an explicit
        ``None`` is "no cap" and overrules the environment, which is what a
        caller who passes it means.
        """
        if isinstance(value, _Unset):
            value = MAX_MODEL_MB
        if value is None:
            return None
        value = int(value)
        if value < 1:
            raise ValueError("max_model_mb must be at least 1, or None for no cap")
        return value

    def _index_for(self, max_model_mb: Optional[int],
                   count_cached_as_free: bool) -> _Index:
        key = (max_model_mb, count_cached_as_free)
        index = self._indexes.get(key)
        if index is None:
            index = _build_index(self._runnable, max_model_mb,
                                 count_cached_as_free)
            self._indexes[key] = index
        return index

    def _resolve(self, max_model_mb: Union[int, None, "_Unset"],
                 count_cached_as_free: Optional[bool]) -> _Index:
        """The index a call runs against, after its per-call overrides."""
        cap = self.max_model_mb if isinstance(max_model_mb, _Unset) \
            else self._check_max_model_mb(max_model_mb)
        free = self.count_cached_as_free if count_cached_as_free is None \
            else count_cached_as_free
        return self._index_for(cap, free)

    # -- introspection ----------------------------------------------------

    @property
    def languages(self) -> FrozenSet[str]:
        """Every language a *runnable* model within the size cap can read or write.

        A language only an unrunnable or over-budget model reaches is not
        listed, because a caller reads this as "these are the languages I can
        ask for", and both filters apply to the models :meth:`route` will use.
        """
        return self._index.languages

    def languages_under(self, max_model_mb: Union[int, None, "_Unset"] = UNSET,
                        count_cached_as_free: Optional[bool] = None
                        ) -> FrozenSet[str]:
        """:attr:`languages` for a size cap this graph was not built with.

        The per-call counterpart of the per-call ``max_model_mb=`` on
        :meth:`route`, so a caller who overrides the budget for one call can
        still ask what that budget covers.
        """
        return self._resolve(max_model_mb, count_cached_as_free).languages

    @property
    def oversized_capabilities(self) -> Tuple[Capability, ...]:
        """Runnable models the size cap left out. Named in :class:`NoRouteError`."""
        return self._index.oversized

    @staticmethod
    def _check_max_hops(max_hops: int) -> int:
        """The constructor's invariant, enforced on the per-call override too.

        ``max_hops=0`` used to run the direct-hop loop anyway and behave as
        ``max_hops=1``, which contradicted the constructor rejecting it.
        """
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        return max_hops

    def _node(self, tag: str) -> str:
        """Caller-supplied tag -> graph node. Strict about junk; see :func:`normalize_tag`."""
        try:
            node = normalize_tag(tag, strict=True)
        except MalformedTagError:
            # The lenient registry path can mint a node no standard knows, and
            # a node that exists must stay addressable by the name it has.
            # Only a tag that is neither parseable nor a node is refused.
            lowered = tag.strip().lower()
            if lowered in self._addressable:
                return lowered
            raise
        if node in self._addressable or "-" not in node:
            return node
        # `pt-BR` is well-formed and unsupported as written, but the graph is
        # keyed on the language subtag, and refusing a region the models simply
        # do not distinguish would be a worse answer than serving `pt`.
        base = node.split("-")[0]
        if base in self._addressable:
            LOG.debug("%r is not a graph node; routing it as %r", node, base)
            return base
        return node

    def capabilities_for(self, src: str, tgt: str,
                         index: Optional[_Index] = None) -> List[Capability]:
        """Every model that can do this exact pair in one hop, best first."""
        index = index or self._index
        cands = list(index.bilingual.get((src, tgt), ()))
        cands += [c for c in index.multilingual if c.covers(src, tgt)]
        return sorted(cands, key=self._cap_key)

    @staticmethod
    def _cap_key(cap: Capability) -> tuple:
        # Within one leg: dedicated first, then permissive, then small.
        return (0 if cap.dedicated else 1, cap.tier_rank, cap.size_mb, cap.model_id)

    # -- pivots -----------------------------------------------------------

    def _pivot_candidates(self, src: str, tgt: str,
                          index: Optional[_Index] = None) -> List[str]:
        """Bounded, ordered pivot list. Never "every language in the graph".

        Order: pair-specific regional pivots, then the configured global
        preference, then any remaining language that is an endpoint of a
        dedicated edge. Languages reachable *only* through a multilingual model
        are excluded - pivoting through one of those uses the same model that
        already covers the direct pair, so it can never be an improvement.
        """
        index = index or self._index
        ordered: List[str] = []
        seen = {src, tgt}

        def add(lang: str) -> None:
            if lang not in seen and lang in index.languages:
                seen.add(lang)
                ordered.append(lang)

        for lang in REGIONAL_PIVOTS.get(tgt, ()):
            add(lang)
        for lang in REGIONAL_PIVOTS.get(src, ()):
            add(lang)
        for lang in self.pivot_preference:
            add(lang)
        for lang in index.dedicated_endpoints:
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

    def _legs(self, src: str, tgt: str,
              index: Optional[_Index] = None) -> List[Capability]:
        """At most one dedicated and one multilingual candidate for a leg."""
        index = index or self._index
        best: List[Capability] = []
        dedicated = [c for c in index.bilingual.get((src, tgt), ())]
        if dedicated:
            best.append(min(dedicated, key=self._cap_key))
        multi = [c for c in index.multilingual if c.covers(src, tgt)]
        if multi:
            best.append(min(multi, key=self._cap_key))
        return best

    @staticmethod
    def _hop(cap: Capability, src: str, tgt: str) -> Hop:
        return Hop(model_id=cap.model_id, src=src, tgt=tgt, arch=cap.arch,
                   license=cap.license, license_tier=cap.license_tier,
                   size_mb=cap.size_mb, dedicated=cap.dedicated)

    def _enumerate(self, src: str, tgt: str, max_hops: int, prefer: str,
                   short_circuit: bool, index: _Index) -> List[Route]:
        routes: List[Route] = []

        for cap in self.capabilities_for(src, tgt, index):
            routes.append(Route(src, tgt, (self._hop(cap, src, tgt),),
                                prefer=prefer, max_hops=max_hops,
                                pivot_basis=self.pivot_ranking))

        # Under fewest_hops a direct route can never be beaten by a longer one,
        # so enumerating pivots would be pure waste on the hot path.
        if routes and short_circuit and prefer == "fewest_hops":
            return routes
        if max_hops < 2:
            return routes

        direct_multi = {c.model_id for c in index.multilingual
                        if c.covers(src, tgt)}
        pivots = self._pivot_candidates(src, tgt, index)

        rank_of = {lang: i for i, lang in enumerate(pivots)}
        for path in self._pivot_paths(pivots, max_hops - 1):
            nodes = (src,) + path + (tgt,)
            pivot_rank = tuple(rank_of[p] for p in path)
            legs = [self._legs(a, b, index) for a, b in zip(nodes, nodes[1:])]
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
               limit: Optional[int] = None,
               max_model_mb: Union[int, None, "_Unset"] = UNSET,
               count_cached_as_free: Optional[bool] = None) -> List[Route]:
        """Every viable route, ranked best-first under the active policy.

        Bounded on purpose: returns at most ``limit`` routes (default
        ``max_routes``, 10). Each leg of a multi-hop route contributes at most
        its best dedicated and best multilingual candidate, so this is a
        curated ranking rather than an exhaustive product of every model
        combination. Returns ``[]`` rather than raising - use :meth:`route`
        when you want ``NoRouteError``.
        """
        src, tgt = self._node(src), self._node(tgt)
        prefer = prefer or self.prefer
        max_hops = self._check_max_hops(
            self.max_hops if max_hops is None else max_hops)
        limit = self.max_routes if limit is None else limit
        index = self._resolve(max_model_mb, count_cached_as_free)
        if src == tgt:
            return []
        found = self._enumerate(src, tgt, max_hops, prefer,
                                short_circuit=False, index=index)
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
              prefer: Optional[str] = None,
              max_model_mb: Union[int, None, "_Unset"] = UNSET,
              count_cached_as_free: Optional[bool] = None) -> Route:
        """The single best route, or raise :class:`NoRouteError`.

        ``max_model_mb`` and ``count_cached_as_free`` override the graph's own
        size budget for this call, the way ``max_hops`` and ``prefer`` override
        the hop cap and the policy. Omit them to inherit; pass
        ``max_model_mb=None`` to lift the budget for one call.
        """
        raw_src, raw_tgt = src, tgt
        src, tgt = self._node(src), self._node(tgt)
        prefer = prefer or self.prefer
        max_hops = self._check_max_hops(
            self.max_hops if max_hops is None else max_hops)
        if src == tgt:
            raise NoRouteError(
                f"source and target are the same language ({src!r}); nothing to translate")
        index = self._resolve(max_model_mb, count_cached_as_free)
        found = self._enumerate(src, tgt, max_hops, prefer, short_circuit=True,
                                index=index)
        if not found:
            raise NoRouteError(
                f"no route from {raw_src!r} to {raw_tgt!r} within "
                f"{max_hops} hop(s)"
                f"{self._why_not(src, tgt, max_hops, prefer, index)}")
        return min(found, key=lambda r: _route_key(r, prefer))

    def _why_not(self, src: str, tgt: str, max_hops: int, prefer: str,
                 index: _Index) -> str:
        """Name the constraint that actually blocked the route.

        Two caps can each turn a servable pair into a failure, they are fixed
        in opposite directions, and they interact: a size budget removes the
        one multilingual model that covered a pair directly, and the chain that
        replaces it needs a hop the caller did not allow. "No route" alone
        sends that caller to the wrong knob, or to the wrong conclusion - that
        the language is unsupported.

        So each constraint is probed on its own, on the failure path only,
        where the search has already come back empty:

        * the **hop cap**, if one more hop finds a route under the same budget;
        * the **size cap**, if lifting the budget finds a route within the same
          hops.

        Both are reported when both are true, because both are. When neither
        is, the pair is genuinely uncovered and the licence and runnability
        hints say what covers it elsewhere.

        Raising ``max_hops`` on the caller's behalf is deliberately not done.
        Both caps were set by the same caller, and trading one for the other
        silently is not a decision this layer gets to make.

        The hop probe runs from ``max_hops=1`` only. The two-hop search is the
        expensive one, and speculating a third hop on every miss would make
        failure cost more than success.
        """
        hint = ""
        if max_hops == 1:
            hint += " (max_hops=1: only direct models were considered)"
            if self._enumerate(src, tgt, 2, prefer, short_circuit=True,
                               index=index):
                hint += "; a 2-hop route exists, so raise max_hops to use it"
        if index.oversized:
            uncapped = self._index_for(None, True)
            if self._enumerate(src, tgt, max_hops, prefer, short_circuit=True,
                               index=uncapped):
                blocking = sorted({
                    f"{cap.model_id} ({cap.size_mb} MB)"
                    for cap in index.oversized
                    if self._could_serve(cap, src, tgt, uncapped, max_hops)})
                return hint + (
                    f" -- the {index.max_model_mb} MB size cap (max_model_mb) "
                    f"excluded the model(s) that would serve this: "
                    f"{', '.join(blocking)}. Raise the cap, allow more hops so "
                    f"smaller models can be chained, or prefetch the model so "
                    f"the download is already paid for")
        if hint:
            return hint
        return (self._excluded_licence_hint(src, tgt)
                + self._unrunnable_hint(src, tgt))

    def _could_serve(self, cap: Capability, src: str, tgt: str,
                     uncapped: _Index, max_hops: int) -> bool:
        """Whether an excluded model is on a route that lifting the cap allows.

        A model that covers the pair outright always qualifies; so does one
        that serves a leg of a route the uncapped search found. Listing every
        oversized model instead would name models that had nothing to do with
        this pair.
        """
        if cap.covers(src, tgt):
            return True
        found = self._enumerate(src, tgt, max_hops, self.prefer,
                                short_circuit=False, index=uncapped)
        return any(cap.model_id in route.model_ids for route in found)

    def _excluded_licence_hint(self, src: str, tgt: str) -> str:
        """Say so when the only thing blocking a route is the licence filter.

        Non-commercial models are left out of the default graph so nobody
        inherits CC-BY-NC output without asking for it. That is the right
        default, but silence is the wrong way to enforce it: a language only
        NLLB covers (Kabuverdianu, for one) would otherwise look simply
        unsupported. Name the models and the flag instead.
        """
        excluded = getattr(self, "excluded_capabilities", None)
        if not excluded:
            return ""
        covering = sorted({
            cap.model_id for cap in excluded
            if cap.covers(src, tgt)
        })
        if not covering:
            return ""
        return (f" -- excluded non-commercial model(s) cover this pair: "
                f"{', '.join(covering)}; pass include_noncommercial=True to use them")

    def _unrunnable_hint(self, src: str, tgt: str) -> str:
        """Say when the pair is covered, but only by a model nothing can run.

        Without this the caller reads "no route" as "this language pair is not
        in the registry", which is the wrong thing to go and fix.
        """
        covering = sorted({cap.model_id for cap in self.unrunnable_capabilities
                           if cap.covers(src, tgt)})
        if not covering:
            return ""
        reasons = sorted({cap.unrunnable_because
                          for cap in self.unrunnable_capabilities
                          if cap.covers(src, tgt)})
        return (f" -- model(s) cover this pair but cannot be run: "
                f"{', '.join(covering)} ({'; '.join(reasons)})")

    def can_translate(self, src: str, tgt: str, max_hops: Optional[int] = None,
                      max_model_mb: Union[int, None, "_Unset"] = UNSET,
                      count_cached_as_free: Optional[bool] = None) -> bool:
        """Whether :meth:`route` would succeed *and* the route would execute.

        Answers for the same models ``translate()`` will use: a pair served
        only by a model this library cannot run is ``False``, not ``True``
        followed by ``NotImplementedError`` two calls later.
        """
        try:
            self.route(src, tgt, max_hops=max_hops,
                       max_model_mb=max_model_mb,
                       count_cached_as_free=count_cached_as_free)
            return True
        except NoRouteError:
            return False

    # -- validation of a caller-supplied route ----------------------------

    def validate_route(self, route: Route) -> Route:
        """Check a :class:`Route` built outside the graph before it is executed.

        ``translate(route=...)`` is the escape hatch for a caller who disagrees
        with the cost model, and it runs the hops verbatim. Nothing in a
        hand-built :class:`Hop` is checked by construction, and the failure it
        invites is the one this whole module exists to prevent: a directional
        model - IndicTrans2 ``en-indic``, the one-way ``nos-coda`` pairs,
        liv4ever - accepts both tags of a backwards hop, because both are in
        its code map, and translates in the direction it was trained in while
        the caller is told it got the other one.

        Raises :class:`InvalidRouteError`; returns the route so it can be used
        inline.
        """
        if not route.hops:
            raise InvalidRouteError("route has no hops")
        for hop in route.hops:
            cap = self._by_model.get(hop.model_id)
            if cap is None:
                raise InvalidRouteError(
                    f"{hop.model_id!r} is not a model in this graph")
            if not cap.is_runnable:
                raise InvalidRouteError(
                    f"{hop.model_id} cannot be run: {cap.unrunnable_because}")
            if not cap.covers(hop.src, hop.tgt):
                raise InvalidRouteError(
                    f"{hop.model_id} does not translate "
                    f"{hop.src!r} -> {hop.tgt!r}; running it anyway would "
                    f"produce fluent text in the wrong language")
        for first, second in zip(route.hops, route.hops[1:]):
            if first.tgt != second.src:
                raise InvalidRouteError(
                    f"hop {first} ends in {first.tgt!r} but the next hop "
                    f"starts from {second.src!r}")
        if route.hops[0].src != route.src or route.hops[-1].tgt != route.tgt:
            raise InvalidRouteError(
                f"route says {route.src!r} -> {route.tgt!r} but its hops go "
                f"{route.hops[0].src!r} -> {route.hops[-1].tgt!r}")
        return route
