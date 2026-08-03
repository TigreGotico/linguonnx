# Licences

`linguonnx` is Apache-2.0. The models are not all Apache-2.0, and a model's
licence follows its output into whatever you build. So the library treats
licence as a first-class routing fact rather than a footnote: it downloads no
model you did not ask for, keeps the restrictive ones out of the default
translation graph, and reports the tier of every route it returns.

## The tiers

Each registry entry carries a `license` string and a `license_tier`. Three
tiers, in order of how many strings are attached:

| tier | what is in it | in the default graph? |
|---|---|---|
| `permissive` | Apache-2.0, MIT, CC-BY-4.0 | yes |
| `share-alike` | CC-BY-SA-3.0 | LID only, by name |
| `non-commercial` | CC-BY-NC-4.0 | no |

Tier is a routing key, not just metadata. It is the third element of the
[route cost tuple](routing.md#cost-ordering), so between two otherwise equal
routes the more permissive chain wins.

## Language identification

Nothing is filtered here, because a detector is one model and you chose it by
name. But the choice matters:

- **GlotLID** is Apache-2.0. It is the default, and stays the default, because
  it is the only permissive option.
- **OpenLID and OpenLID-v2 are GPL-3.0**, inherited from the upstream models.
  If your project cares about licence compatibility, do not use them.
- **lid.176** is CC-BY-SA-3.0, which has its own share-alike condition. It is
  also the only model small enough for some devices, so it is a real trade
  rather than a trap.

## Translation

`load_translator()` builds its graph from **permissive models only**. Four
registry entries are excluded by default: `nllb-600M` and `aina-es-oc`, in both
precisions. Both are CC-BY-NC-4.0.

That is not a judgement about the models. NLLB-200 is excellent and covers 202
languages, more than anything else here except MADLAD. It is a judgement about
defaults: a library should not put a non-commercial licence into a caller's
output because the caller did not read a table.

Opting in is one argument:

```python
from linguonnx import load_translator

tx = load_translator(include_noncommercial=True)
print(tx.route("pt", "kea").model_ids)   # ('nllb-600M-int8',)
print(tx.route("es", "oc").model_ids)    # ('aina-es-oc-int8',)
```

### The filter holds at load time, not only at routing time

A translator loads only the models it was built over. Naming an excluded model
in a `route=` or a `model=` raises, rather than loading it and translating
through it:

```python
tx = load_translator()                       # permissive models only
try:
    tx.translate("bom dia", src="pt", tgt="kea", model="nllb-600M-int8")
except ValueError as err:
    print(err)
    # 'nllb-600M-int8' is not in this translator's models. It is
    # filtered out (see include_noncommercial=, precision= and models= on
    # load_translator) or it does not exist.
```

This is what makes `include_noncommercial=False` a guarantee about the output
rather than a preference about routing. If you want the model, ask for it —
with `include_noncommercial=True`, or by naming it in `models=`.

## Vendored code

`linguonnx/translate/_indic_processor.py` is AI4Bharat's `IndicProcessor`,
copied from [`IndicTransToolkit`](https://github.com/VarunGumma/IndicTransToolkit)
rather than depended on, because that package declares `transformers` as a hard
dependency and this library keeps `transformers` out of the runtime. The
changes are mechanical: Cython declarations dropped, the progress bar removed,
and the placeholder map returned to the caller instead of parked in a
module-level queue. Why a copy and not a reimplementation is explained in
[translate.md](translate.md#indictrans2) — a reimplementation could diverge
from the training-time preprocessing and translate fluently into the wrong
words.

The upstream licence is MIT:

```
MIT License

Copyright (c) 2024 Varun Gumma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MIT is a permissive licence, so this changes nothing about a route's
`license_tier`. It is recorded here because the file is someone else's work.

## What a route tells you

Every `Route` carries the licence of every hop, and a tier for the chain as a
whole:

```python
tx = load_translator()
route = tx.route("pt", "eu", prefer="dedicated")
print(route.licenses)       # one licence per hop
print(route.license_tier)   # the most restrictive of them
```

`license_tier` is the **most restrictive** tier on the chain, because a route
is only as free as its worst hop. A permissive first hop does not launder a
non-commercial second one.

## When a licence is what blocks a pair

Excluding models by default has one bad failure mode: a language that only a
non-commercial model covers looks simply unsupported. Kabuverdianu is one —
only NLLB has it — and "no route" would be a misleading thing to say about a
model that is sitting in the registry.

So the error says which models were excluded and how to get them:

```python
from linguonnx.translate import NoRouteError

tx = load_translator()
try:
    tx.route("pt", "kea")
except NoRouteError as err:
    print(err)
# no route from 'pt' to 'kea' within 2 hop(s) -- excluded non-commercial
# model(s) cover this pair: nllb-600M-int8; pass include_noncommercial=True
# to use them
```

The hint only appears when the licence filter is genuinely the thing in the
way. A pair no model covers at all gets the plain message, and `max_hops=1`
adds its own note saying only direct models were considered, so the three
reasons a route can fail are never confused with each other.
