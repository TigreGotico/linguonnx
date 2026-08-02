# Translation

`linguonnx.translate` runs encoder-decoder translation models on ONNX Runtime,
with numpy doing the generation loop. Nothing here imports torch.

```python
from linguonnx import load_translator

tx = load_translator()

print(tx.translate("bom dia, como estás?", src="pt", tgt="en"))
# 'Good morning, how are you?'
```

Models download on first use and cache under
`~/.cache/linguonnx/models/<model_id>/`. Sessions are built lazily, so building
a translator over the whole registry costs nothing but parsing one JSON file —
only the models a hop actually needs are ever fetched.

## The API

<!-- doc-check: norun a signature summary, not a program -->
```python
tx.translate(text, src="pt", tgt="en")                  # -> str
tx.translate(text, src="pt", tgt="en", return_route=True)  # -> (str, Route)
tx.translate(text, model="opus-mt-pt-en-int8")          # pin one model
tx.translate(text, route=some_route)                    # run a route verbatim

tx.route("pt", "en")            # decide, translate nothing
tx.routes("pt", "en")           # every viable route, ranked
tx.can_translate("pt", "en")    # bool
tx.available_languages          # every tag reachable in the graph
tx.models                       # the registry entries in play
```

`route()` and `routes()` never load a model. They read the registry, so
inspecting coverage across hundreds of language pairs is free. See
[routing.md](routing.md).

There are three ways to say *how* a translation should happen, in order of
precedence. `model=` pins one registry model and skips routing entirely.
`route=` runs a `Route` you built or picked yourself, even when it is not the
top-ranked one. `src=`/`tgt=` is the normal path: score the candidates and take
the winner. Give none of them and you get a `ValueError` rather than a guess.

## Building the translator

```python
tx = load_translator(
    precision="int8",              # "fp32", or None for both
    prefer="fewest_hops",          # or "dedicated"
    max_hops=2,
    include_noncommercial=False,
    num_beams=4,
    max_new_tokens=128,
)
```

`precision="int8"` is the default because these are CPU models and int8 is
where they become usable on one. Nothing forces you to it: pass `"fp32"` for
the full-precision graphs, or `None` to let both into the graph and let the
cost model prefer the smaller file.

`models=["opus-mt-pt-en-int8", ...]` restricts the graph to exactly those
registry ids and overrides every filter, which is the right tool for a
deployment that ships a fixed set of weights. An unknown id raises `ValueError`
listing what it did not recognise, at construction time rather than on the
first translation.

## Generation

Greedy and beam search are both implemented over the raw graphs.

```python
tx = load_translator(num_beams=4, max_new_tokens=128)   # the defaults
tx.translate("bom dia", src="pt", tgt="en", num_beams=1)   # greedy, ~4x faster
```

`no_repeat_ngram_size` is off by default and available as a loop guard.
`length_penalty` and `early_stopping` behave as they do in `transformers`.

Beam search reorders the KV cache by parent-beam index at every step, which is
the one genuinely fiddly part of the loop, and is tested against an independent
numpy beam search over a toy ONNX graph.

The cache is wired **by name**: every `past_key_values.<i>.<attn>.<kv>` input
is matched to the graph output of the same name with `past_key_values` swapped
for `present`. Nothing assumes a layer count, a head layout, or that the two
decoder graphs order their tensors the same way. A graph with a naming scheme
this does not cover raises at load time rather than producing silent garbage.

## Selecting the target language, per architecture

This is the part that fails **silently**. Get it wrong and the model does not
raise: it returns fluent, well-formed text in the wrong language. Each
architecture chooses its target differently, and there is no shared mechanism
to fall back on.

| arch | how the target is chosen |
|---|---|
| `marian` | Nothing to choose for a dedicated pair — the model **is** the pair. A *multilingual* Marian export prepends a `<2xx>` token to the input, built from `target_token_template` in the registry entry. |
| `m2m100` | Source language is the first token of the *input*; target is forced as the decoder's first generated token, `forced_bos_token_id = lang_id(tgt)`. Codes are plain `en`, `pt`, `gl`. |
| `nllb` | The same forced-decoder mechanism, but the codes are FLORES-200 (`por_Latn`), so language and script are chosen together. |
| `madlad` (T5) | A `<2xx>` piece prepended to the input text, exactly like any other SentencePiece piece — not a forced decoder id. |
| `indictrans2` | A custom `IndicProcessor` pipeline: script normalisation and transliteration, then a `<src> <tgt>` prefix. Not implemented, see below. |
| `opennmt-bpe` | Moses tokenisation plus `subword-nmt` BPE over OpenNMT's concatenated source/target vocabulary. Not implemented, see below. |

`linguonnx` handles the implemented architectures behind one call and converts
BCP-47 to whatever codes the model wants, so `tgt="pt"` means Portuguese
whichever model serves the hop. The per-architecture tests assert that changing
the target actually changes the output language, because nothing else would
catch a regression here.

### Marian group models

Some opus-mt repos are exports of *group* models rather than single pairs. A
repo named `opus-mt-en-pl-onnx` is an export of `opus-mt-en-sla`, which serves
several Slavic languages from one decoder and needs a `>>pol<<` prefix token on
the input. Without the token the model picks a Slavic language on its own and
you get Czech, fluently, with nothing raised.

The registry records the `target_token` the export was verified against, and
`TranslationModel.translate` applies it automatically. You can override it per
call with `target_token=`, which is worth knowing about if you pin such a model
directly.

The mirror case is harmless: `opus-mt-pt-en-onnx` is an export of
`opus-mt-ROMANCE-en`, multi-*source*. The source language is inferred from the
text, so `pt -> en` still holds without any token.

### Architectures that are listed but cannot run

`indictrans2` and `opennmt-bpe` models are in the registry, but `translate()`
on one raises `NotImplementedError`. Both need a preprocessing pipeline this
library does not vendor, and vendoring it would mean taking on Moses
tokenisation or IndicNLP as a runtime dependency.

Their entries carry `"runnable": false`, and the router honours it: they are
excluded from `route()`, `routes()`, `can_translate()` and
`available_languages`. Listing them while routing through them would put the
failure in the worst possible place — `can_translate()` answering `True`, then
`NotImplementedError` from the call the caller made on the strength of that
answer. They stay in the registry because the entry is still true about what
the export covers, and `NoRouteError` names them when they are the only cover
for a pair. See [routing.md](routing.md#coverage-and-runnability-are-separate).

## When there is no route

`route()` and `translate()` raise `NoRouteError` — a `LookupError` subclass —
when nothing within `max_hops` connects the pair. The message says what was
tried, and says so specifically when the licence filter is the only thing in
the way:

```python
from linguonnx.translate import NoRouteError

try:
    tx.route("pt", "kea")           # Kabuverdianu
except NoRouteError as err:
    print(err)
# no route from 'pt' to 'kea' within 2 hop(s) -- excluded non-commercial
# model(s) cover this pair: nllb-600M-int8; pass include_noncommercial=True
```

See [licences.md](licences.md) for why that model is out of the default graph.

Asking to translate a language into itself also raises, rather than returning
the input unchanged, because a caller who does that usually has a bug upstream
in their language detection.
