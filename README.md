# linguonnx

CPU-first language technology models on ONNX Runtime, in the spirit of
[`onnx-asr`](https://github.com/istupakov/onnx-asr) (speech recognition) and
[`phoonnx`](https://github.com/TigreGotico/phoonnx) (text-to-speech).

Two packages under one namespace:

- `linguonnx.detect` - language identification, built on the TigreGotico ONNX
  exports of four fastText classifiers: GlotLID, fastText's classic lid.176,
  OpenLID and OpenLID-v2.
- `linguonnx.translate` - machine translation over Marian (opus-mt), M2M100 and
  NLLB-200 ONNX exports, with a routing graph that decides which model, or
  which chain of models, connects a language pair.

Runtime dependencies are `onnxruntime`, `numpy`, `sentencepiece`,
`huggingface_hub` and `langcodes`. **No torch, at any point.** The
encoder-decoder generation loop, beam search and KV cache included, is written
against the raw ONNX graphs.

## Install

```bash
pip install linguonnx
```

Models download from HuggingFace on first use and are cached under
`~/.cache/linguonnx/models/<model_id>/`.

## Language identification: how the model split works

Every model here is a fastText classifier. fastText's actual math - averaging
embedding rows and multiplying by the output matrix, then softmax - is
representable as an ONNX graph, so that part runs as a normal ONNX Runtime
session:

```
input_ids: int64[num_features]  ->  probs: float32[num_labels]
```

(lid.176 is the exception, and gets its own section below.)

What ONNX *can't* do portably is fastText's own preprocessing: tokenizing
text and hashing character n-grams into `input_ids`. That half stays in
Python, in `linguonnx/detect/hashing.py`, vendored from the reference
implementation published alongside the ONNX weights
(`glotlid_hash.py` in the HuggingFace repo; every model in the registry uses
the same hashing, differing only in the `nwords`/`minn`/`maxn`/`bucket`
values read from its `config.json`). It is a straight port of
fastText's `Dictionary` class, including the detail that trips up most
from-scratch reimplementations: fastText hashes each UTF-8 byte as a
**signed** `int8`, so bytes >= `0x80` get sign-extended before the FNV-1a
XOR. Every non-ASCII n-gram (i.e. every non-Latin-script language) hashes to
the wrong bucket if you "fix" that.

## Language identification API

```python
from linguonnx import load_detector

det = load_detector()                 # glotlid-int8; downloads on first use
det = load_detector("lid176-int8")    # or openlid-int8, openlid-v2-int8, ...

det.detect("bo dia, como estas?")        # -> "gl"              BCP-47 best guess
det.detect_probs("...", top_k=5)         # -> {"gl": 0.82, "pt": 0.11, ...}
det.detect_raw("...")                    # -> ("glg_Latn", 0.82)  native GlotLID label
det.available_languages                  # -> set of BCP-47 tags it can emit
det.loss                                 # -> 'softmax' or 'hs'
```


## Language varieties

GlotLID labels individual varieties, not macrolanguages. Casual Arabic comes
back as `ajp-Arab` (South Levantine) or `ars` (Najdi/Saudi) rather than `ar`,
and Chinese may come back as `yue-Hani` (Cantonese). There are 11 Arabic and
8 Chinese varieties in the label set.

That is useful on its own — it is free text-side dialect identification — but
most callers want a tag they can act on (a TTS voice, a translation target).
Both are available:

```python
det.detect_raw("وش لونك يا خوي")               # ('ars_Arab', 0.87)  Najdi/Saudi
det.detect("وش لونك يا خوي")                    # 'ars'
det.detect("وش لونك يا خوي", collapse_varieties=True)   # 'ar'
```

## Models

| model_id           | file                    | size    | labels | precision | licence      |
|--------------------|-------------------------|---------|--------|-----------|--------------|
| `glotlid-int8`     | `glotlid.int8.onnx`     | 419 MB  | 2102   | int8      | Apache-2.0   |
| `glotlid`          | `glotlid.onnx`          | 1.68 GB | 2102   | fp32      | Apache-2.0   |
| `lid176-int8`      | `lid176.int8.onnx`      | 31 MB   | 176    | int8      | CC-BY-SA-3.0 |
| `lid176`           | `lid176.onnx`           | 125 MB  | 176    | fp32      | CC-BY-SA-3.0 |
| `openlid-int8`     | `openlid.int8.onnx`     | 293 MB  | 201    | int8      | GPL-3.0      |
| `openlid`          | `openlid.onnx`          | 1.15 GB | 201    | fp32      | GPL-3.0      |
| `openlid-v2-int8`  | `openlid-v2.int8.onnx`  | 290 MB  | 200    | int8      | GPL-3.0      |
| `openlid-v2`       | `openlid-v2.onnx`       | 1.13 GB | 200    | fp32      | GPL-3.0      |

The families:

- **[GlotLID](https://huggingface.co/TigreGotico/glotlid-onnx)** - 2102
  language-script labels, by far the widest coverage. The default.
- **[lid.176](https://huggingface.co/TigreGotico/lid176-onnx)** - Meta's
  original fastText language identifier. 176 languages, and only 31 MB in
  int8, so it is the one to reach for on a small device. It is a 16-dimension
  model and leans on diacritics to separate close Romance languages; feed it
  properly accented text.
- **[OpenLID](https://huggingface.co/TigreGotico/openlid-onnx)** and
  **[OpenLID-v2](https://huggingface.co/TigreGotico/openlid-v2-onnx)** - 201
  and 200 curated FLORES-200 varieties, trained on audited data. Prefer v2
  unless you need parity with the original OpenLID paper.

### Licences

`glotlid-int8` is the default, and stays the default, because GlotLID is the
only Apache-2.0 model here.

**OpenLID and OpenLID-v2 are GPL-3.0**, inherited from the upstream models.
If your project cares about licence compatibility, do not use them - stay on
GlotLID. lid.176 is CC-BY-SA-3.0, which has its own share-alike condition.
`linguonnx` itself is Apache-2.0 and downloads no model you did not name.

`glotlid-int8` is the default precision as well as the default family: the
HuggingFace repo's own parity testing found identical top-1 decisions against
the fp32 graph on a 78-sample multilingual set, so there is no accuracy reason
to prefer the larger file. The same holds for `openlid`; `lid176.int8.onnx`
does lose a little (96.6% vs 100% top-1 agreement with fastText on the
publisher's 59-language set), so use `lid176` fp32 when 125 MB is affordable.

### Label shapes

GlotLID, OpenLID and OpenLID-v2 emit `iso3_Script` labels (`eng_Latn`,
`glg_Latn`, `zho_Hans`). lid.176 emits bare ISO codes with no script
(`en`, `gl`, `pt`). `linguonnx` maps both to BCP-47 tags via
`linguonnx.detect.labels`; see that module's docstring for the exact rules.

## Hierarchical softmax

lid.176 was trained with fastText's hierarchical softmax (`loss=hs`). Under
`hs` the rows of the output matrix are **not** per-label scores: each row is a
binary classifier for one internal node of a Huffman tree built over the label
frequencies, and a label's probability is the product of the sigmoid (or
1-sigmoid) values along its root-to-leaf path. Running the ordinary
`MatMul -> Softmax` recipe on such a model produces well-formed numbers and
0% agreement with fastText.

So the lid.176 ONNX graph ends in `Sigmoid` over the Huffman nodes, and
`linguonnx.detect.hs` walks the paths in Python (the tree is precomputed into
`hs_tree.json` at export time, and `build_tree()` reproduces the construction
from label counts). Which path a model takes is decided by the `loss` field on
its registry entry, falling back to the `loss` in its own `config.json` and
then to `softmax` - never by the model's name.

## Label output notes

GlotLID's Arabic labels are dialect-specific (`arb`/Standard, `ars`/Najdi,
`aeb`/Tunisian, `ajp`/South Levantine, ...); short colloquial text is often
- correctly - tagged with a dialect rather than `ar`. Its Chinese labels
distinguish `Hans`/`Hant`/`Hani` (simplified / traditional / script-neutral);
text using only characters shared between scripts resolves to `zh-Hani`, not
`zh-Hans`.

## Translation

```python
from linguonnx import load_translator

tx = load_translator()

tx.translate("bom dia", src="pt", tgt="eu")                 # -> 'Egun on'
tx.translate("bom dia", src="pt", tgt="eu", return_route=True)  # -> (str, Route)
tx.route("pt", "eu")                                        # inspect, translate nothing
tx.routes("pt", "eu")                                       # all viable routes, ranked
tx.available_languages                                      # union over the graph
tx.translate("bom dia", model="opus-mt-pt-en-int8")         # pin one model
```

Models download on first use and cache under
`~/.cache/linguonnx/models/<model_id>/`. Sessions are built lazily, so
constructing a translator over the whole registry costs nothing.

### The graph

Nodes are BCP-47 language tags. Edges are **capabilities**, not materialised
pairs:

- A **bilingual** model (opus-mt / Marian) is one directed edge, `en -> pt`.
- A **multilingual** model (M2M100, NLLB) declares the set of languages it
  covers and is any-to-any inside it. NLLB's 202 languages would be 40602
  directed edges written out; the set is stored once and expanded only when a
  pair is resolved.

`tx.route(src, tgt)` returns a `Route`: an ordered list of `Hop`s, each naming
the model and the exact `(src, tgt)` it handles. A `Route` is always available
to the caller, so a pivot is never silent:

```python
route = tx.route("pt", "eu")
route.n_hops        # 2
route.pivots        # ('en',)
route.model_ids     # ('opus-mt-pt-en-int8', 'opus-mt-en-eu-int8')
route.licenses      # ('Apache-2.0', 'Apache-2.0')
route.license_tier  # 'permissive' - the most restrictive tier on the chain
route.prefer        # 'fewest_hops' - the policy that produced this route
route.max_hops      # 2
print(route)
# [pt->eu, 2 hop(s), prefer=fewest_hops] pt->en via opus-mt-pt-en-int8
# (dedicated, Apache-2.0) | en->eu via opus-mt-en-eu-int8 (dedicated, Apache-2.0)
```

### Cost ordering

Every candidate route is scored by a tuple. The policy reorders the tuple's
first two elements; it does not run a different search.

| `prefer=` | key |
|---|---|
| `"fewest_hops"` (default) | `(hops, multilingual_hops, licence_tier, pivot_rank, size, model_ids)` |
| `"dedicated"` | `(multilingual_hops, hops, licence_tier, pivot_rank, size, model_ids)` |

Read in order, that means:

1. **Hop count or dedication**, depending on the policy (below).
2. **A dedicated bilingual model beats a multilingual one for the same pair.**
   `en -> pt` picks `opus-mt-en-pt` over M2M100, under both policies.
3. **Licence tier**: permissive before share-alike before non-commercial.
4. **Pivot preference**: a linguistically closer pivot before a further one.
5. **Smaller model**, then model id, so the ranking is deterministic.

**The hop-count-versus-dedication tradeoff is unmeasured for this model set.**
Whether two strong Marian hops (`pt -> en -> ru`) beat one distilled-NLLB or
M2M100 hop (`pt -> ru`) is genuinely pair-dependent, and settling it needs a
benchmark rather than intuition. So it is a policy parameter, not a baked-in
assumption:

```python
tx = load_translator(prefer="fewest_hops")   # DEFAULT
tx = load_translator(prefer="dedicated")     # 2 dedicated hops > 1 multilingual hop
tx.route("pt", "ru", prefer="dedicated")     # or per call
```

`fewest_hops` is the default because it is the conservative choice: it never
doubles latency or compounds error unless it is asked to. It is not a claim
that it produces better text.

### Hop cap

```python
tx = load_translator(max_hops=2)   # DEFAULT
tx.route("pt", "eu", max_hops=1)   # per-call override
```

- `max_hops=1` - direct models only. Raises `NoRouteError` rather than pivoting.
  This is the strict mode for when quality matters more than coverage.
- `max_hops=2` - the default.
- `max_hops>=3` - allowed, not recommended. Translation error compounds
  multiplicatively per hop while latency adds up, so the third hop costs a lot
  and buys little. It is not forbidden, because the caller may know something
  the registry does not.

### Choosing the path yourself

The cost model is a default, not a verdict. Three ways to overrule it:

```python
for route in tx.routes("pt", "ru"):     # every viable route, ranked, with licences
    print(route)

tx.translate("ola", route=my_route)     # execute a route verbatim, no scoring
tx.translate("ola", model="m2m100-418M-int8", src="pt", tgt="ru")   # pin a model
```

`routes()` is bounded on purpose: it returns the top 10 by default (`limit=`),
and each leg of a multi-hop route contributes at most its best dedicated and
best multilingual candidate. It is a curated ranking, not the full product of
every model combination.

### Why routing stays fast

The graph is mostly cliques - NLLB alone is a 202-language clique - so a naive
breadth-first search over materialised nodes would weigh ~200 pivots at two
hops and ~40000 at three, nearly all of them pointless. Three bounds prevent
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

### Pivot choice

The pivot preference is data, not `en` hardcoded in the search. English is the
global default because that is where the bilingual training data is, but it is
a poor pivot inside the Iberian peninsula, where Spanish keeps far more of the
morphology and lexicon. `linguonnx/translate/graph.py` holds a global
`DEFAULT_PIVOT_PREFERENCE` and a per-language `REGIONAL_PIVOTS` table, and both
are overridable:

```python
tx = load_translator(pivot_preference=("es", "en", "fr"))
tx.route("gl", "ca").pivots      # ('es',) - not English
```

#### Ranking pivots by measured distance

A hand-written table only knows the pairs somebody thought of. Install the
optional extra and the same candidates get ordered by a measured linguistic
distance instead:

```bash
pip install linguonnx[distance]     # adds orthography2ipa
```

```python
tx = load_translator(pivot_ranking="auto")   # DEFAULT
tx.route("pt", "eu").pivot_basis             # 'phonological', or 'table'
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

```python
(max(first_leg, second_leg), first_leg + second_leg, table_index)
```

Output through a pivot is bottlenecked by the worse leg, so a candidate that
happens to sit very close to the source cannot buy its way past a bad second
leg. Ranking on the total alone would pick Galician as the `es -> ru` pivot,
purely because `es->gl` is 0.10; the worst-leg rule picks Ukrainian, whose leg
into Russian is 0.23. For `pt -> eu` Spanish wins either way: its worst leg is
0.25 against English's 0.45.

The total is a tiebreak rather than the primary key, and both are judgement
calls that a benchmark could overturn.

Three limits are deliberate:

- The candidate **set** does not change. `REGIONAL_PIVOTS` plus the preference
  list plus the dedicated-edge endpoints still decide who is considered; the
  distance only reorders them, so the search stays as bounded as before.
- A language `orthography2ipa` does not know scores nothing, not zero. It keeps
  its table position and sorts after every candidate that does have a distance.
- Every pair is memoised, so a routing call never recomputes a distance.

`Route.pivot_basis` reports which ranking produced the path.

**CLDR distance was measured and rejected.** `langcodes.tag_distance` looks
like the obvious answer and is not: it is a *locale-matching* score, built to
pick which translation file to serve a user, not to say how alike two languages
are. It scores `pt->es` and `pt->en` identically (84 each) and rates `pt->gl`
as distant as `pt->en`. It cannot rank pivots.

`orthography2ipa`'s `full_distance` and `ancestry_similarity` are also unused,
for now. Their ancestry component is incomplete - every Romance medieval stage
in the dataset carries an empty ancestry list, so Ibero-Romance languages never
meet at a shared ancestor and `full_distance` ends up rating English closer to
Catalan than Spanish is. Fix that data gap before switching.

### Selecting the target language, per architecture

This is the part that fails **silently**. Get it wrong and the model does not
raise; it returns fluent text in the wrong language. Each architecture does it
differently:

| arch | how the target is chosen |
|---|---|
| `marian` | Nothing to choose. The model **is** the pair. |
| `m2m100` | Source language is the first token of the *input*; target language is forced as the decoder's first generated token, `forced_bos_token_id = lang_id("pt")`. Codes are plain `en`, `pt`, `gl`. |
| `nllb` | Same mechanism, but the codes are FLORES-200 (`por_Latn`), so language and script are chosen together. |

`linguonnx` handles all three behind one call and converts BCP-47 to whatever
codes the model wants, so `tgt="pt"` means Portuguese whichever model serves
the hop. The per-architecture tests assert the target actually changes the
output language, because nothing else would catch it.

### Basque, and why the default is not flat

**M2M100 does not support Basque.** `eu` is absent from its 100 codes. A single
flat default model would therefore drop Basque silently. The default graph is
M2M100-418M for general coverage *plus* the opus-mt bilingual pairs, and `eu`
routes through `opus-mt-en-eu` / `opus-mt-eu-en`:

```python
tx.route("pt", "eu").model_ids
# ('opus-mt-pt-en-int8', 'opus-mt-en-eu-int8')
```

### Licences

The default graph is **permissive only**. NLLB-200 is in the registry but out
of the default graph, because it is CC-BY-NC-4.0 and this library will not put
a non-commercial licence into a caller's output without being asked:

```python
tx = load_translator(include_noncommercial=True)   # adds NLLB-200
```

`Route.license_tier` reports the most restrictive tier on the chain - a route
is only as free as its worst hop.

### Models

`linguonnx/model_index/translate.json` holds the registry: 64 entries, fp32 and
int8 for every model. `load_translator()` defaults to `precision="int8"`; pass
`precision="fp32"` or `precision=None` for both.

| model_id | arch | languages | size (int8) | licence |
|---|---|---|---|---|
| `m2m100-418M-int8` | M2M100 | 100, any-to-any | 1.2 GB | MIT |
| `m2m100-1.2B-int8` | M2M100 | 100, any-to-any | 2.3 GB | MIT |
| `nllb-600M-int8` | NLLB-200 | 202, any-to-any | 1.8 GB | **CC-BY-NC-4.0** |
| `opus-mt-<src>-<tgt>-int8` | Marian | one pair | 180-1020 MB | Apache-2.0 or CC-BY-4.0 |

28 opus-mt pairs are published: `ar-en`, `ca-en`, `ca-es`, `de-en`, `en-ar`,
`en-ca`, `en-de`, `en-es`, `en-eu`, `en-fr`, `en-gl`, `en-it`, `en-nl`,
`en-pt`, `en-ru`, `en-zh`, `es-ca`, `es-en`, `es-gl`, `eu-en`, `fr-en`,
`gl-en`, `gl-es`, `it-en`, `nl-en`, `pt-en`, `ru-en`, `zh-en`.

### Generation

Greedy and beam search are both implemented over the raw graphs.

```python
tx = load_translator(num_beams=4, max_new_tokens=128)   # defaults
tx.translate(text, src="en", tgt="pt", num_beams=1)     # greedy, ~4x faster
```

`no_repeat_ngram_size` is off by default and available as a loop guard. Beam
search reorders the KV cache by parent-beam index at every step, which is the
one genuinely fiddly part and is tested against an independent numpy beam
search over a toy ONNX graph.

## Development

```bash
uv pip install -e .
pytest test/ -m "not network"     # unit tests: no download, no model
pytest test/                      # also runs the real end-to-end model downloads
```

Routing and decoding are tested without any real model. The graph is pure data,
and the decode loop runs against a few-kilobyte ONNX seq2seq built in the test
file, using the same `past_key_values.*` / `present.*` naming the real exports
use. The network-marked tests then compare the hand-written decoder to
`transformers` and `optimum` on the real graphs, string for string - those two
are test-only dependencies and must never appear in `linguonnx/`.

## Related projects

`linguonnx` answers "what language is this text in?". These siblings answer the
neighbouring questions, and are worth reaching for instead of stretching this
library to cover them:

- **[scriptconv](https://github.com/TigreGotico/scriptconv)** — the *writing
  system* rather than the language: zero-dependency ISO-15924 script detection
  and metadata, plus conversions between phoneme notations (IPA ↔ ARPABET,
  X-SAMPA, Kirshenbaum, Cotovía, RFE), Buckwalter ↔ Arabic, Hangul → jamo and
  kana. A GlotLID label carries a script subtag (`zho_Hans`, `srp_Cyrl`); use
  scriptconv when the script itself is the thing you need to identify or
  transliterate.
- **[ovos-lang-parser](https://github.com/OpenVoiceOS/ovos-lang-parser)** —
  language *names* rather than text: parses a spoken or written language name
  into a BCP-47 code, and renders a BCP-47 code back into a spoken name. Pair it
  with `linguonnx` when a user says or reads a language name ("translate this to
  Brazilian Portuguese") and you need the tag, or when you want to speak a
  detected tag back to them.
- **[phoonnx](https://github.com/TigreGotico/phoonnx)** — text to speech on ONNX
  Runtime, 1000+ languages.
- **[onnx-asr](https://github.com/istupakov/onnx-asr)** — speech to text on ONNX
  Runtime; the structural model this library follows.
