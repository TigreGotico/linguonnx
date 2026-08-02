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
| `"fewest_hops"` (default) | `(hops, multilingual_hops, licence_tier, pivot_rank, size, model_ids)` |
| `"dedicated"` | `(multilingual_hops, hops, licence_tier, pivot_rank, size, model_ids)` |

Read in order, that means:

1. **Hop count or dedication**, depending on the policy.
2. **A dedicated bilingual model beats a multilingual one for the same pair.**
   `en -> pt` picks `opus-mt-en-pt` over M2M100, under both policies.
3. **Licence tier**: permissive before share-alike before non-commercial.
4. **Pivot preference**: a linguistically closer pivot before a further one.
5. **Smaller model**, then model id, so the ranking is deterministic and the
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

`routes()` is bounded on purpose. It returns the top 10 by default (`limit=`),
and each leg of a multi-hop route contributes at most its best dedicated and
its best multilingual candidate. It is a curated ranking, not the full product
of every model combination. It returns `[]` for an unroutable pair rather than
raising; use `route()` when you want the `NoRouteError`.

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
