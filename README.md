# linguonnx

CPU-first language technology on ONNX Runtime, in the spirit of
[`onnx-asr`](https://github.com/istupakov/onnx-asr) (speech recognition) and
[`phoonnx`](https://github.com/TigreGotico/phoonnx) (text to speech).

Two packages under one namespace:

- `linguonnx.detect` — language identification over ONNX exports of four
  fastText classifiers: GlotLID, fastText's classic lid.176, OpenLID and
  OpenLID-v2.
- `linguonnx.translate` — machine translation over Marian (opus-mt), M2M100,
  NLLB-200 and MADLAD exports, with a routing graph that decides which model,
  or which chain of models, connects a language pair.

Runtime dependencies are `onnxruntime`, `numpy`, `sentencepiece`,
`huggingface_hub` and `langcodes`. **No torch, at any point.** The
encoder-decoder generation loop, beam search and KV cache included, is written
against the raw ONNX graphs, because pulling in `torch` to run a 150 MB
quantised model is a trade nobody on a small device wants to make.

## Install

```bash
pip install linguonnx
pip install linguonnx[distance]   # adds orthography2ipa, for pivot ranking
```

Models download from HuggingFace on first use and are cached under
`~/.cache/linguonnx/models/<model_id>/`.

## Identify a language

```python
from linguonnx import load_detector

det = load_detector()            # glotlid-int8, 425 MB on first use

print(det.detect("Egun on, zer moduz?"))              # 'eu'
print(det.detect_probs("Bon dia a tothom", top_k=3))  # {'ca': 0.998, ...}
print(det.detect_raw("وش لونك يا خوي"))                # ('ars_Arab', 0.99)
print(det.detect("وش لونك يا خوي", collapse_varieties=True))   # 'ar'
```

GlotLID labels 2102 *varieties*, not macrolanguages, so colloquial Arabic
comes back as a dialect (`ars`, Najdi) and Chinese may come back as Cantonese.
That is free text-side dialect identification when you want it and a nuisance
when you do not, which is what `collapse_varieties` is for. See
[docs/detect.md](docs/detect.md).

## Translate

```python
from linguonnx import load_translator

tx = load_translator()

print(tx.translate("bom dia, como estás?", src="pt", tgt="en"))
# 'Good morning, how are you?'   via opus-mt-pt-en-int8, 172 MB
```

A pair no single model covers is chained through a third language, and the
`Route` comes back with the translation so a pivot is never silent:

```python
route = tx.route("pt", "eu", prefer="dedicated")
print(route.model_ids)      # ('opus-mt-pt-ca-int8', 'mt-hitz-ca-eu-int8')
print(route.pivots)         # ('ca',) — it went through Catalan
print(route.license_tier)   # 'permissive' — the worst licence on the chain
```

See [docs/translate.md](docs/translate.md) for the API and
[docs/routing.md](docs/routing.md) for how a route is chosen.

## What ships

| | |
|---|---|
| Language identification | 4 models, 176 to 2102 labels, 33 MB to 1.7 GB |
| Translation | 150 registry entries — fp32 and int8 of 75 models |
| Reachable languages | 459, over the default graph |
| Default LID model | `glotlid-int8` — Apache-2.0, the only permissive LID option |
| Default translation graph | every permissive int8 model, fewest hops, capped at 2 |
| Licences | Apache-2.0, MIT and CC-BY-4.0 by default; GPL-3.0 and CC-BY-NC-4.0 must be asked for by name |

`linguonnx` itself is Apache-2.0 and downloads no model you did not ask for.
Some of the models are not: OpenLID is GPL-3.0 and NLLB-200 is CC-BY-NC-4.0,
so non-commercial models are kept out of the default translation graph
entirely. [docs/licences.md](docs/licences.md) explains what that costs and how
to opt in.

## Documentation

- [docs/detect.md](docs/detect.md) — language identification: the four models,
  the variety labels, hierarchical softmax, BCP-47 mapping.
- [docs/translate.md](docs/translate.md) — the translation API, and how each
  architecture picks its target language. Get that wrong and nothing raises.
- [docs/routing.md](docs/routing.md) — capabilities rather than edges, the
  `prefer` policies, hop caps, pivot ranking, pinning a route yourself.
- [docs/models.md](docs/models.md) — the registry, what is in it, and the
  generated `sync_registry.py` workflow that keeps it honest.
- [docs/licences.md](docs/licences.md) — the licence tiers and what
  `NoRouteError` tells you when a licence is what blocks a pair.

Runnable scripts live in [`examples/`](examples/). Each says in its docstring
what it demonstrates and what it downloads.

## Development

```bash
uv pip install -e .[test]
pytest test/ -m "not network"     # unit tests: no download, no model
pytest test/                      # also runs the real model downloads
python scripts/check_docs.py      # execute every code sample in these docs
                                   # (needs network: it downloads real models,
                                   # same as `pytest -m network`; not run in CI)
```

Routing and decoding are tested without any real model. The graph is pure
data, and the decode loop runs against a few-kilobyte ONNX seq2seq built in the
test file, using the same `past_key_values.*` / `present.*` naming the real
exports use. The network-marked tests then compare the hand-written decoder to
`transformers` and `optimum` on the real graphs, string for string — those two
are test-only dependencies and must never appear in `linguonnx/`.

## Related projects

`linguonnx` answers "what language is this text in?" and "say it in another
one". These siblings answer the neighbouring questions, and are worth reaching
for instead of stretching this library to cover them:

- **[scriptconv](https://github.com/TigreGotico/scriptconv)** — the *writing
  system* rather than the language: zero-dependency ISO-15924 script detection
  and metadata, plus conversions between phoneme notations (IPA ↔ ARPABET,
  X-SAMPA, Kirshenbaum, Cotovía, RFE), Buckwalter ↔ Arabic, Hangul → jamo and
  kana. A GlotLID label carries a script subtag (`zho_Hans`, `srp_Cyrl`); use
  scriptconv when the script itself is the thing you need to identify or
  transliterate.
- **[ovos-lang-parser](https://github.com/OpenVoiceOS/ovos-lang-parser)** —
  language *names* rather than text: parses a spoken or written language name
  into a BCP-47 code, and renders a BCP-47 code back into a spoken name. Pair
  it with `linguonnx` when a user says or reads a language name ("translate
  this to Brazilian Portuguese") and you need the tag, or when you want to
  speak a detected tag back to them.
- **[phoonnx](https://github.com/TigreGotico/phoonnx)** — text to speech on
  ONNX Runtime, 1000+ languages.
- **[onnx-asr](https://github.com/istupakov/onnx-asr)** — speech to text on
  ONNX Runtime; the structural model this library follows.
