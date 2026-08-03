# Routing

No single translation model covers every pair anyone wants. Some pairs have a
dedicated bilingual model, some are only reachable inside a big multilingual
one, and some need two models chained through a third language. The routing
graph decides which, and hands the decision back to the caller.

Nothing in this document loads a model. Routing reads the registry, so you can
inspect coverage over thousands of pairs for the cost of parsing one JSON file.

## Capabilities, not edges

Nodes are BCP-47 language tags. An edge is a **capability**: a claim by one
model about what it can translate.

- A **bilingual** model (opus-mt / Marian) is one directed edge, `en -> pt`.
- A **multilingual** model (M2M100, NLLB, MADLAD) declares the set of
  languages it covers and is any-to-any inside it.
- A **directional** multilingual model covers a source set and a target set
  that differ, and cannot be run backwards. IndicTrans2 ships three of these:
  `en-indic` only goes English into 25 Indic tags, `indic-en` only the
  reverse, `indic-indic` is symmetric.

Storing the sets rather than the pairs is what makes this tractable. NLLB's 202
languages would be 40602 directed edges written out; the set is stored once and
expanded only when a pair is resolved. Directionality is stored too, because
treating a one-way model as any-to-any would let the router propose a hop that
fails at runtime.

## Coverage and runnability are separate

A capability says what a model's weights know. It also says whether this
library can **run** them, which is a different question: an entry can be a
correct claim about languages and still have no inference pipeline here.
IndicTrans2 needs the IndicProcessor preprocessing chain and ProxectoNos'
OpenNMT exports need Moses plus subword-nmt BPE; neither is vendored, so
`TranslationModel.translate` raises `NotImplementedError` for them.

The routing layer agrees with that. A model the registry marks unrunnable is
excluded from `route()`, `routes()`, `can_translate()` and
`available_languages`, so a caller who asks "will this work" before committing
gets the same answer `translate()` will give:

```python
from linguonnx import load_translator

tx = load_translator()
print(tx.can_translate("en", "sat"))    # False — no runnable model has Santali
```

The pair is still covered by an entry, and the error says so instead of
implying the language is unknown:

```
no route from 'en' to 'sat' within 2 hop(s) -- model(s) cover this pair but
cannot be run: indictrans2-en-indic-dist-200M-int8 (IndicTrans2 needs the
IndicProcessor preprocessing pipeline ...)
```

Runnability is registry data, not a list of architecture names in the router.
`scripts/sync_registry.py` writes `"runnable": false` and an
`"unrunnable_reason"` onto the entries of an architecture it knows has no
pipeline. When a pipeline lands, that flag stops being written and the models
route again with no change to `graph.py`. A caller building capabilities by
hand can state it directly with `Capability(runnable=False, ...)`; the default,
`None`, means "ask the registry".

One rule is applied on top of the flag, because it guards a failure with no
error at all: a Marian model that serves several targets from one decoder picks
its target with a `>>xxx<<` or `<2xx>` prefix token, and with no token it
answers in whichever of them it likes. An entry like that with no
`target_token` and no `target_token_template` is refused and logged rather than
routed.

## Routes

`tx.route(src, tgt)` returns a `Route`: an ordered list of `Hop`s, each naming
the model and the exact `(src, tgt)` it handles.

```python
from linguonnx import load_translator

tx = load_translator()
route = tx.route("pt", "eu")

print(route.n_hops)         # 1
print(route.model_ids)      # ('madlad400-3b-mt-int8',)
print(route.pivots)         # () — no pivot on a direct route
print(route.licenses)       # ('Apache-2.0',)
print(route.license_tier)   # 'permissive' — the worst tier on the chain
print(route.prefer)         # 'fewest_hops' — the policy that produced this
print(route.pivot_basis)    # 'phonological' or 'table'
print(route)
# [pt->eu, 1 hop(s), prefer=fewest_hops] pt->eu via madlad400-3b-mt-int8
# (multilingual, Apache-2.0)
```

A `Route` is always available to the caller, and `translate(...,
return_route=True)` hands it back alongside the text. That is the point: a
pivot through a third language changes the quality of the output, so it must
never happen silently.

## Cost ordering

Every candidate route is scored by a tuple, lowest first. The policy reorders
the tuple's first two elements; it does not run a different search.

| `prefer=` | key |
|---|---|
| `"fewest_hops"` (default) | `(hops, multilingual_hops, non_specialist_hops, recency, licence_tier, pivot_rank, size, model_ids)` |
| `"dedicated"` | `(multilingual_hops, hops, non_specialist_hops, recency, licence_tier, pivot_rank, size, model_ids)` |

Read in order, that means:

1. **Hop count or dedication**, depending on the policy.
2. **A dedicated bilingual model beats a multilingual one for the same pair.**
   `en -> pt` picks `opus-mt-en-pt` over M2M100, under both policies.
3. **Specialist provenance**: the institution that owns the language beats a
   generalist that merely covers it. See "Specialist provenance" below.
4. **Recency**: the more recently published upstream model, as a last resort.
5. **Licence tier**: permissive before share-alike before non-commercial.
6. **Pivot preference**: a linguistically closer pivot before a further one.
7. **Smaller model**, then model id, so the ranking is deterministic and the
   same registry always produces the same answer.

**The hop-count-versus-dedication tradeoff is unmeasured for this model set.**
Whether two strong Marian hops (`pt -> en -> ru`) beat one MADLAD or M2M100 hop
(`pt -> ru`) is genuinely pair-dependent, and settling it needs a benchmark
rather than intuition. So it is a parameter, not a baked-in assumption:

```python
tx = load_translator(prefer="fewest_hops")   # the default
tx = load_translator(prefer="dedicated")     # 2 dedicated hops > 1 multilingual
print(tx.route("pt", "ru", prefer="fewest_hops").model_ids)  # or per call
```

`fewest_hops` is the default because it is the conservative choice: it never
doubles latency or compounds error unless it is asked to. It is not a claim
that it produces better text.

The two policies really do diverge. Under the default, `pt -> eu` is one MADLAD
hop; under `prefer="dedicated"` it is a two-hop Marian chain through Catalan.

```python
tx = load_translator()
print(tx.route("pt", "eu").model_ids)
# ('madlad400-3b-mt-int8',)
print(tx.route("pt", "eu", prefer="dedicated").model_ids)
# ('opus-mt-pt-ca-int8', 'mt-hitz-ca-eu-int8')
```

Basque is a good illustration of why the default graph is not flat: **M2M100
does not support Basque at all**, `eu` is absent from its 100 codes. Leaving
Basque to a general multilingual model would have dropped it entirely, so the
dedicated `opus-mt-*-eu` and `mt-hitz-*-eu` models are what make it reachable
under `prefer="dedicated"`, and MADLAD is what makes it a single hop by
default.

## Specialist provenance

Being *covered* by a model and being *owned* by an institution that studies
the language are different facts, and the plain cost-ordering tiers above
cannot see the difference: "dedicated beats multilingual" only asks whether a
model *is* the pair, not who built it or how well. A big multilingual model
that lists Basque as one of a hundred codes is, in that sense, on equal
footing with HiTZ - the institute whose entire remit is Basque - the moment
both happen to be dedicated (or both multilingual) for the same pair.

`SPECIALIST_MAP` in `linguonnx/translate/graph.py` names, for a handful of
languages, the institution that is *the* specialist for it:

| Language(s) | Institution |
|---|---|
| Basque (`eu`) | HiTZ |
| Galician (`gl`), Asturian (`ast`) | Proxecto Nós |
| Catalan (`ca`), Aragonese (`an`), Occitan/Aranese (`oc`) | Projecte AINA |
| Indic block (`hi`, `bn`, `ta`, `te`, `ml`, `mr`, `gu`, `pa`, `ur`, `sa`, and the rest of the IndicTrans2 set) | AI4Bharat |
| West-African pairs (`bam`, `bbj`, `fon`, `ewe`, `mos`) | Masakhane |
| Estonian and Finno-Ugric minority languages (`et`, `vro`, `sma`) | TartuNLP |
| Livonian (`liv`) | liv4ever |

A model's registry entry carries `provenance_org` when it is a fine-tune or
export from one of these institutions (see `PROVENANCE_BY_PREFIX` in
`scripts/sync_registry.py`, derived from the model_id prefix, not guessed
from the card). `TranslationGraph` checks that against `SPECIALIST_MAP` for
the specific `src`/`tgt` being routed - `is_specialist_for(org, src, tgt)` -
so a specialist institution's own model into some third pivot language does
not count as "specialist" for that leg; only its claim on the language(s) the
table names does.

This is a **default policy, not a wall** - consistent with the rest of this
module's design (see "Choosing the path yourself" below). It only ever breaks
a tie between routes that already tied on hop count and dedicated-vs-
multilingual status. A generalist still wins, and is still used, whenever no
specialist model covers the pair at all - Basque is served by MADLAD or
M2M100 just fine when no HiTZ model is in the graph. And a hand-built `Route`
passed to `translate(route=...)` or resolved from an explicit `models=[...]`
runs exactly as given: nothing here second-guesses a caller who already knows
which model they want. The ranking is inspectable, not just applied: every
`Hop` carries `.specialist`, `.provenance_org`, and `.release_date`, and
`Route.specialist_hops` / `Route.n_non_specialist_hops` summarise them, so a
caller can see *why* a route won without re-deriving it from the registry.

### The rule behind the table, not just the table

Two languages in the table are shipped by **two** of these institutions, and
which one wins looks arbitrary until you read it as one rule, not two special
cases:

> **Ties between specialist institutions break on the geographical proximity
> of the institution to the language.**

- **`ast` (Asturian) -> Proxecto Nós, not Projecte AINA.** Both institutions
  publish an `es -> ast` model. Asturian sits on the Galician-Portuguese
  continuum - Eonavian (Galician-Asturian) is spoken across the Navia-Eo
  border strip between Asturias and Galicia - so the Galician institute is
  the geographically adjacent one, and it wins.
- **`an` (Aragonese) -> Projecte AINA, not Proxecto Nós.** Both institutions
  ship an `es -> an` model too (`aina-translator-es-an`, `nos-mt-es-arg`).
  Aragonese borders Catalonia and shares the Aragon-Catalan transitional
  dialect zone (the *Franja Oriental*), so the Catalan institute is the
  adjacent one here instead, and it wins.

Without this written down, a future maintainer who notices "Nós ships an
Aragonese model and it's losing to AINA" - or the mirror image for Asturian -
will read it as a routing bug and "fix" it into an inconsistency: whichever
institution ships a pair should just win, symmetrically. It is deliberately
*not* symmetric. Proximity is the rule; `eu -> HiTZ`, the Indic block ->
AI4Bharat, and the rest of the table are the unambiguous cases where only one
institution is geographically anywhere near the language at all. `ast` and
`an` are where the rule actually has to do work, which is exactly why they
are the two entries worth writing a rule down for.

### Recency: a last resort, and only ever upstream

Below specialist provenance sits one more tie-break: the more recently
published model wins, consulted only once hop count, dedication, and
specialist provenance have all already tied.

"Recently published" means the **upstream** model's own creation date -
`Capability.release_date`, sourced by hand from each institution's *source*
repository on the Hub (`HiTZ/mt-hitz-es-eu`, `projecte-aina/...`,
`ai4bharat/...`, and so on), never this project's own `TigreGotico` mirror.
That distinction matters because it is a trap otherwise: every mirror this
project exports gets a fresh `lastModified` the day it is exported, so if the
mirror's own timestamp were used, re-uploading *any* model - fixing a
licence tag, adding an int8 variant, nothing to do with quality - would
silently promote it over everything else on recency alone. The registry
field is deliberately populated from the upstream repo's `createdAt`, once,
by hand, and the sync script documents exactly where each date came from
(`PROVENANCE_BY_PREFIX` in `scripts/sync_registry.py`).

Not every institution's date has been verified yet - Proxecto Nós's
`proxectonos` org repos and TartuNLP's `smugri` model are undated as of this
writing, rather than guessed. An undated model sorts as if it were
infinitely old, so it never wins a recency comparison and never blocks a
route for a reason it has no evidence for; it just stops competing on this
one tier and falls through to licence and size, same as before this feature
existed.

## Hop cap

```python
tx = load_translator(max_hops=2)             # the default
print(tx.route("pt", "en", max_hops=1))      # per-call override
```

`max_hops=1` means direct models only: it raises `NoRouteError` rather than
pivoting, which is the strict mode for when quality matters more than coverage.
`max_hops=2` is the default. Anything higher is allowed but not recommended —
translation error compounds multiplicatively per hop while latency only adds
up, so a third hop costs a lot and buys little. It is not forbidden, because
the caller may know something the registry does not.

`max_hops` must be at least 1, and the per-call override is held to the same
rule as the constructor: `route(src, tgt, max_hops=0)` raises `ValueError`. A
route with no hops translates nothing, so there is no reading of `0` worth
guessing at.

## Size budget

`max_model_mb` is the largest single model routing may put on a route. A model
over the budget is left out, and the router looks for a chain of smaller ones
instead of failing.

<!-- doc-check: norun the second line's answer depends on the local cache -->
```python
tx = load_translator(max_model_mb=500)
print(tx.route("pt", "ru").model_ids)
# ('opus-mt-pt-en-int8', 'opus-mt-en-ru-int8')
```

Without the budget that pair is one hop through `m2m100-418M-int8`, 1207 MB.
With it, two Marian models of 172 MB and 169 MB do the same work through
English: 341 MB fetched instead of 1207 MB, at two encoder-decoder passes
instead of one. That is the trade the budget exists to make, and it is the
reason the budget does not simply refuse the pair.

The budget is per **model**, not per route. Six 100 MB hops pass a 500 MB
budget; one 600 MB model does not. The number bounds what a single fetch can
cost, which is the thing that fails on a slow link.

### Download cost, not memory cost

By default `count_cached_as_free=True`, and a model already in
`~/.cache/linguonnx` passes the budget however big it is. Nothing is
downloaded to find that out — the cache directory is inspected, the hub is
never contacted.

The two settings answer two different questions:

| | reading | set |
|---|---|---|
| `count_cached_as_free=True` (default) | "do not **download** more than this" | metered or slow connection |
| `count_cached_as_free=False` | "do not **use** a model bigger than this" | small disk, or a memory ceiling |

So `prefetch("madlad400-3b-mt-int8", kind="translate")` at startup keeps MADLAD
routable under a 500 MB budget: the download it was excluded for has already
been paid. Under `count_cached_as_free=False` it stays excluded, because there
the number is about the model, not the fetch.

The default is `True` because the budget is a *download* budget: refusing a
model already on disk costs quality and saves nothing.

### What a budget costs in coverage

Over the default selection — permissive, int8, 73 models — with an empty cache:

| `max_model_mb` | models kept | languages routable |
|---|---|---|
| none | 73 | 459 |
| 2000 | 71 | 121 |
| 1000 | 69 | 57 |
| 500 | 62 | 54 |
| 300 | 56 | 29 |

The model column and the language column tell different stories, and the second
one is the one to plan against. A 500 MB budget drops 11 of 73 models, but
those 11 include `madlad400-3b-mt-int8` (4945 MB), `m2m100-1.2B-int8` (2344 MB)
and `m2m100-418M-int8` (1207 MB), and the big multilingual models are where the
long tail of languages lives. MADLAD is the only model in the registry with
Chuvash at all; no chain of small models replaces it, because there is no small
model on either side of it.

What survives is what has bilingual models: the ~50 languages opus-mt, mt-hitz
and the Iberian pairs cover between them. `pt -> ru`, `nl -> fi`, `pt -> eu`
all still route under 500 MB, as chains. `de -> ja` does not, because nothing
under 500 MB has Japanese.

So a budget is not a way to shrink the registry evenly. It keeps the
well-served pairs and drops the tail, and `available_languages` says which is
which:

```python
tx = load_translator(max_model_mb=500, count_cached_as_free=False)
print(len(tx.available_languages))       # 54
print(tx.can_translate("pt", "ru"))      # True — via a chain
print(tx.can_translate("de", "ja"))      # False — no small model has Japanese
```

`available_languages` is filtered by the budget for the same reason it is
filtered by runnability: a caller reads it as "these are the languages I can
ask for", and it has to answer for the models `route()` will actually use.

### The budget and the hop cap interact

A budget can make a one-hop route impossible where one existed, and the chain
that replaces it needs a hop. Both caps came from the caller, so neither is
raised on the caller's behalf — `NoRouteError` names the one that bound
instead:

```
no route from 'pt' to 'ru' within 1 hop(s) (max_hops=1: only direct models
were considered); a 2-hop route exists, so raise max_hops to use it -- the 500
MB size cap (max_model_mb) excluded the model(s) that would serve this:
m2m100-418M-int8 (1207 MB). Raise the cap, allow more hops so smaller models
can be chained, or prefetch the model so the download is already paid for
```

Both constraints are reported when both are true. When neither is — when the
pair is simply not covered — the budget is not mentioned at all, and the
licence and runnability hints say what covers the pair elsewhere.

### Per call, and from the environment

`max_model_mb` and `count_cached_as_free` override per call, the way `max_hops`
and `prefer` do. Omitting a keyword inherits the translator's value; passing
`max_model_mb=None` lifts the budget for that one call, because `None` already
means "no budget" and cannot also mean "inherit".

<!-- doc-check: norun the answers depend on what this host has cached -->
```python
tx = load_translator(max_model_mb=500)
tx.route("pt", "ru")                        # inherits the 500 MB budget
tx.route("pt", "ru", max_model_mb=2000)     # this call only
tx.route("pt", "ru", max_model_mb=None)     # no budget for this call
tx.route("pt", "ru", count_cached_as_free=False)
```

`LINGUONNX_MAX_MODEL_MB` sets the default, alongside the other bounds in
`linguonnx/limits.py`, so an operator can impose it on a deployment without
patching the caller. An explicit `max_model_mb=` argument, including
`max_model_mb=None`, overrules it.

Routing is the only thing the budget governs. `translate(model=...)` pins a
model and bypasses routing entirely, and it stays pinned; the download bound
that protects that path is `LINGUONNX_MAX_DOWNLOAD_MB`, in
`linguonnx/model_manager.py`.

### What a route costs to fetch

A `Route` reports the download it implies, so a caller choosing between routes
can see it rather than infer it from `size_mb`:

<!-- doc-check: norun the split depends on what this host has cached -->
```python
for route in tx.routes("pt", "ru")[:3]:
    print(route.model_ids, route.download_size_mb, route.cached_size_mb)
# ('m2m100-418M-int8',)                        1207   0
# ('opus-mt-pt-en-int8', 'opus-mt-en-ru-int8')    0 341
```

`download_size_mb` is what is missing, `cached_size_mb` is what is already on
disk, and `models_size_mb` is their sum. All three count each **model** once,
so a route that uses one multilingual model for two hops counts it once —
unlike `total_size_mb`, which is a per-hop sum because it is a cost proxy in
the route ranking, where a model used twice is used twice.

Fetch cost does not enter the ranking. It changes with the cache, and a
translator that reordered its routes as models were downloaded would give two
hosts in one fleet different answers with nothing to explain it. The number is
reported so the caller can decide; it is not decided for them.

## Choosing the path yourself

The cost model is a default, not a verdict. There are three ways to overrule
it:

```python
tx = load_translator()
for route in tx.routes("pt", "ru")[:3]:   # every viable route, ranked
    print(route)
```

Take one and hand it back, or skip routing altogether by naming a model:

<!-- doc-check: skip runs a 1.2 GB M2M100 download -->
```python
chosen = tx.routes("pt", "ru")[1]
tx.translate("olá", route=chosen)                                  # verbatim
tx.translate("olá", model="m2m100-418M-int8", src="pt", tgt="ru")  # pinned
```

A `Route` that came from `routes()` is already valid. One built by hand is not
checked by construction, and the mistake it invites is the expensive one: a
directional model accepts both tags of a backwards hop, because both are in its
code map, and translates in the direction it was trained in while reporting the
other. `TranslationGraph.validate_route` re-checks every hop against the
capability that will execute it — direction, chain continuity, endpoints, and
runnability — and raises `InvalidRouteError`:

<!-- doc-check: skip constructs a deliberately invalid route -->
```python
from linguonnx.translate.graph import Hop, Route

backwards = Route("hi", "en", (Hop("indictrans2-en-indic-dist-200M-int8",
                                   "hi", "en", "indictrans2", "MIT",
                                   "permissive", 480, False),))
tx.graph.validate_route(backwards)      # InvalidRouteError
```

`routes()` is bounded on purpose. It returns the top 10 by default (`limit=`),
and each leg of a multi-hop route contributes at most its best dedicated and
its best multilingual candidate. It is a curated ranking, not the full product
of every model combination. It returns `[]` for an unroutable pair rather than
raising; use `route()` when you want the `NoRouteError`.

## Language tags

Nodes are BCP-47, and `normalize_tag` maps every shape onto one node name:
`por_Latn`, `POR` and `pt` all resolve to `pt`. A tag carrying a region the
models do not distinguish routes as its language, so `route("pt-BR", "en")`
serves `pt`.

Junk gets two different answers, because two callers with opposite needs share
the function. Building the graph is lenient: an exotic registry code that no
standard knows is lowercased, logged as a warning, and kept as a node, which
leaves the model routable instead of failing construction over one entry.
Caller input is strict: `route()`, `routes()` and `can_translate()` raise
`MalformedTagError` for a tag that cannot be parsed.

<!-- doc-check: norun one of the two lines is meant to raise -->
```python
tx.can_translate("en", "kea")   # False — well-formed, no runnable model
tx.route("en", "!!!")           # MalformedTagError, not NoRouteError
```

The distinction is the point. `NoRouteError` tells the caller their language is
not served, which is the wrong thing to go and fix when what actually arrived
was an unvalidated query parameter. A node the lenient path minted stays
addressable by the name it was given.

## Why routing stays fast

The graph is mostly cliques — NLLB alone is a 202-language clique — so a naive
breadth-first search over materialised nodes would weigh around 200 pivots at
two hops and 40000 at three, nearly all of them pointless. Three bounds prevent
that:

- Pivot candidates are a **bounded, ordered list**: the pair's regional pivots,
  then the configured global preference, then every language that is an
  endpoint of a *dedicated* edge. A language reachable only through a
  multilingual model is never a useful pivot, because the same model already
  covers the pair directly, so it is pruned.
- Under `fewest_hops`, if any one-hop route exists, no two-hop route is
  enumerated at all.
- A two-hop route whose hops are the same multilingual model that already does
  the pair directly is dropped.

## Pivot choice

The pivot preference is data, not `en` hardcoded in the search. English is the
global default because that is where the bilingual training data is, but it is
a poor pivot inside the Iberian peninsula, where Spanish keeps far more of the
morphology and lexicon than English does.

`linguonnx/translate/graph.py` holds a global `DEFAULT_PIVOT_PREFERENCE` and a
per-language `REGIONAL_PIVOTS` table, and both are overridable:

```python
tx = load_translator(pivot_preference=("es", "en", "fr"))
print(tx.route("gl", "ca", prefer="dedicated").pivots)   # ('pt',)
```

### Ranking pivots by measured distance

A hand-written table only knows the pairs somebody thought of. Install the
optional extra and the same candidates get ordered by a measured linguistic
distance instead:

```bash
pip install linguonnx[distance]     # adds orthography2ipa
```

```python
tx = load_translator(pivot_ranking="auto")   # the default
print(tx.pivot_ranking)                      # 'phonological' or 'table'
print(tx.route("pt", "eu").pivot_basis)
```

| `pivot_ranking=` | behaviour |
|---|---|
| `"auto"` (default) | phonological when `orthography2ipa` imports, table otherwise |
| `"phonological"` | demand the package; raise `ValueError` without it |
| `"table"` | curated order only, even when the package is installed |

Each candidate gets two leg distances, `src -> pivot` and `pivot -> tgt`, from
`orthography2ipa.distance.phonological_distance(...).combined`. They are ranked
by **the worse of the two legs first**, then by the total, then by table
position:

```
(max(first_leg, second_leg), first_leg + second_leg, table_index)
```

Output through a pivot is bottlenecked by the worse leg, so a candidate that
happens to sit very close to the source cannot buy its way past a bad second
leg. Ranking on the total alone would pick Galician as the `es -> ru` pivot,
purely because `es -> gl` is 0.10; the worst-leg rule picks Ukrainian, whose
leg into Russian is 0.23. For `pt -> eu` Spanish wins either way — its worst
leg is 0.25 against English's 0.45.

The total is a tiebreak rather than the primary key, and both are judgement
calls a benchmark could overturn.

Three limits are deliberate:

- The candidate **set** does not change. `REGIONAL_PIVOTS` plus the preference
  list plus the dedicated-edge endpoints still decide who is considered; the
  distance only reorders them, so the search stays as bounded as before.
- A language `orthography2ipa` does not know scores nothing, not zero. It keeps
  its table position and sorts after every candidate that does have a distance.
- Every pair is memoised, so a routing call never recomputes a distance.

`Route.pivot_basis` reports which ranking produced the path, so a route is
never ambiguous about where its pivot came from.

### CLDR distance was measured and rejected

`langcodes.tag_distance` looks like the obvious answer and is not. It is a
*locale-matching* score, built to pick which translation file to serve a user,
not to say how alike two languages are. It scores `pt -> es` and `pt -> en`
identically (84 each) and rates `pt -> gl` as distant as `pt -> en`. It cannot
rank pivots.

`orthography2ipa`'s `full_distance` and `ancestry_similarity` are also unused.
Their ancestry component is incomplete — every Romance medieval stage in the
dataset carries an empty ancestry list, so Ibero-Romance languages never meet
at a shared ancestor and `full_distance` ends up rating English closer to
Catalan than Spanish is. That data gap has to close before the switch is worth
making.
