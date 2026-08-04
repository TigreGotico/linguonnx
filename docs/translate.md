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
| `madlad` (T5) | A `<2xx>` piece prepended to the input text, exactly like any other SentencePiece piece — not a forced decoder id. The `<2xx>` spelling belongs to the architecture, not to one export, so the pipeline supplies it and a registry entry cannot switch it off. Encoding without it raises. |
| `indictrans2` | AI4Bharat's `IndicProcessor` pipeline, then a `<src_tag> <tgt_tag>` prefix on the input. Needs the `indic` extra; see below. |
| `opennmt-bpe` | Nothing to choose — the model is the pair. The work is Moses tokenisation plus `subword-nmt` BPE over OpenNMT's concatenated source/target vocabulary. Needs the `opennmt` extra; see below. |

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

## Architectures with their own preprocessing

Two architectures do not translate raw text. They expect the text their
training pipeline produced, and the work on the way out is the exact inverse of
the work on the way in. Both live in `linguonnx/translate/preprocess.py`, one
`Pipeline` subclass each, so that the two halves are written next to each other
and cannot drift apart.

Neither is installed by default. `linguonnx` itself stays on `onnxruntime`,
`numpy` and `sentencepiece`; the extras add the tokenisation these two need.
When an extra is missing the call raises `ImportError` naming it. It never
falls back to a simpler tokenisation, because a wrong tokenisation does not
fail — it translates fluently into the wrong words.

### IndicTrans2

```bash
pip install 'linguonnx[indic]'
```

On the way in:

1. punctuation normalisation;
2. Devanagari, Bengali, Tamil, Perso-Arabic and the other native digits folded
   to ASCII;
3. URLs, emails, numerals and `@handles` replaced by `<ID1>`-style
   placeholders, so the model moves an opaque token instead of trying to
   translate a URL;
4. Moses tokenisation for English, IndicNLP tokenisation for everything else;
5. **transliteration into Devanagari** for every Indic script except
   Perso-Arabic, Ol Chiki, Meetei Mayek and Latin, because the model's shared
   vocabulary is written in Devanagari;
6. the `<src_tag> <tgt_tag> ` prefix — FLORES-style tags such as `hin_Deva` or
   `tam_Taml` — which is what selects the pair.

On the way out, all of that in reverse: script fix-ups for Perso-Arabic and
Oriya, placeholders restored, **transliteration back into the target script**,
then detokenisation.

Step 5 and its inverse are the reason this is not optional work. Skip the
transliteration back and a `tam_Taml` request returns fluent Tamil spelled in
Devanagari. It is correct text in the wrong script, and nothing raises.

The processing code is vendored from AI4Bharat's
[`IndicTransToolkit`](https://github.com/VarunGumma/IndicTransToolkit) rather
than depended on, because that package declares `transformers` as a hard
dependency and `linguonnx` keeps `transformers` out of the runtime. It is a
copy, not a reimplementation — see
[`_indic_processor.py`](../linguonnx/translate/_indic_processor.py) and
[licences.md](licences.md).

**These models accept at most 256 source tokens.** The export bakes in a
256-row sinusoidal position table, so a longer input cannot be embedded at all.
`translate()` raises `InputTooLongError` before anything reaches ONNX Runtime;
split the text into sentences and translate them one at a time.

### OpenNMT-BPE (Proxecto Nós `nos-coda_iacobus-*`)

```bash
pip install 'linguonnx[opennmt]'
```

In: Moses-tokenise with the source language's rules, apply the `*_35k.code` /
`source.bpe` merges shipped in the model repo **constrained by the export's own
source vocabulary**, look the pieces up in that vocabulary, and add
`source_offset`. OpenNMT-py keeps separate source and target vocabularies; the
export concatenates them as `[target | source]`, so encoder input ids carry
that offset and decoder output ids do not. No `</s>` is appended to the source,
because OpenNMT-py does not append one.

The vocabulary constraint is `subword-nmt`'s `--vocabulary`, and it is not
optional. The merge table says *how* to join characters; the vocabulary says
*how far*. Given the vocabulary, `apply_bpe` re-splits any segment the
vocabulary does not contain until every piece has an embedding. Given none, it
applies every merge that fits and hands back a segment the model has never
seen — which becomes `<unk>` on the encoder input, for a word the model knows
perfectly well. On `nos-mt-es-arg`, *duerme* merged to `duer@@ me` and `duer@@`
is not in that export's source vocabulary, while `du@@ er@@ me` is; the model
was handed `Lo <unk> <unk> me` and answered accordingly.

Out: map through the target vocabulary, strip the `@@` merge markers, Moses-
detokenise. The markers are removed with the upstream `sed 's/@\s*//g'` rule,
not `replace("@@ ", "")` — the two differ on a word-final `@@` before
punctuation, where the naive form glues two words together.

**`<unk>` is kept in the output.** Upstream `onmt_translate` hides it with
`-replace_unk`, which copies the aligned source word using the decoder's
cross-attention weights. Those weights are not outputs of the exported graph,
so the substitution cannot be reproduced, and inventing a replacement would be
a guess presented as a translation. A visible `<unk>` says where the model
failed; the original `onmt_translate` emits one in the same places.

That is true of the **target** side only. A `<unk>` on the *source* side is
always a linguonnx bug, and is asserted against per model — see
`TestEveryOpenNmtExportSegmentsIntoItsOwnVocabulary` in
`test/test_translate_preprocess.py`. The `nos-coda_iacobus-*` family has the
smallest target vocabularies in the registry (~18k subwords) and emits target
`<unk>` most often; a 5-sentence spot check counted 8 for `en-es`, 6 for
`en-pt`, 2 for `es-gl` and `es-pt`, 1 for `en-gl` and 0 for `pt-gl`, against 0
for every `nos-mt-*` model on the same sentences. Prefer `nos-mt-*` where the
pair exists.

### Verified against the reference implementations

Ten sentences per model, beam 4, exact string match:

| model | reference | match |
|---|---|---|
| `indictrans2-en-indic-dist-200M` | `IndicProcessor` + `transformers` 4.44.2 | 10/10 |
| `indictrans2-indic-en-dist-200M` | as above | 10/10 |
| `indictrans2-indic-indic-dist-320M` | as above | 10/10 |
| `nos-coda_iacobus-en-gl` | the Moses+BPE recipe on the same graph | 10/10 |
| `nos-coda_iacobus-en-es` | as above | 10/10 |
| `nos-coda_iacobus-en-pt` | as above | 10/10 |
| `nos-coda_iacobus-es-gl` | as above | 10/10 |
| `nos-coda_iacobus-es-pt` | as above | 10/10 |
| `nos-coda_iacobus-pt-gl` | as above | 10/10 |

The IndicTrans2 runs cover Hindi, Tamil, Bengali, Marathi and Malayalam, and
include Indic→Indic pairs, which is what proves the transliteration round-trip
rather than assuming it.

### An architecture that cannot run

Nothing in the registry is in this state today, but the mechanism stays. An
entry can carry `"runnable": false` with an `unrunnable_reason`, and the router
honours it: the model is excluded from `route()`, `routes()`, `can_translate()`
and `available_languages`, while remaining listed, licensed and sized. Listing
a model while routing through it would put the failure in the worst possible
place — `can_translate()` answering `True`, then a raise from the call the
caller made on the strength of that answer. `NoRouteError` names such a model
when it is the only cover for a pair. See
[routing.md](routing.md#coverage-and-runnability-are-separate).

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

## Input limits and generation bounds

The source text is bounded before it reaches the encoder. Self-attention is
quadratic in the source length, and beam search then re-gathers the whole
cross-attention cache once per generated token — around 0.4 MB per source
token for M2M100-418M at 4 beams, so a 1,000-token input moves over a gigabyte
per token produced. Past the model's positional table the output is not a
translation anyway.

| limit | default | environment variable |
|---|---|---|
| encoder input tokens | 1,024 | `LINGUONNX_MAX_ENCODER_TOKENS` |
| `num_beams` | 32 | `LINGUONNX_MAX_NUM_BEAMS` |

1,024 is the largest positional table among the registered architectures
(M2M100); Marian stops at 512. Crossing the bound raises
`linguonnx.limits.InputTooLongError` — split long text into sentences or
paragraphs and translate them one at a time.

`GenerationConfig` validates its values on construction, which is the boundary
`Translator(...)` and `load_translator(...)` pass through:

- `num_beams` — positive int, at most 32. Beams past the number of finite
  candidates are padded with PAD tokens and add nothing, but each is still a
  full row of the attention cache, re-gathered every step.
- `max_new_tokens` — positive int.
- `length_penalty` — between `0.0` and `10.0`. Beam scores are log
  probabilities, so they are negative and divided by
  `length ** length_penalty`; a negative exponent rewards longer sequences and
  can rank a PAD-padded beam first.
- `no_repeat_ngram_size` — `0` (off) or `2` and above. **`1` is rejected**: in
  this loop the n-gram prefix is empty, so every token ever emitted would be
  banned from recurring and the output collapses, while in `transformers` the
  same setting silently does nothing because its lookup key is the whole
  sequence. Neither reading is useful, so it raises.

### Decode failure is not empty output

`translate()` returns `""` for empty input. Generation therefore never returns
an empty sequence to mean failure: if beam search finishes no hypothesis and
every beam collapses, or greedy decoding ends before emitting a token, the
decoder raises `linguonnx.limits.DecodeError`. A caller can tell "nothing to
translate" from "this model could not produce anything for that input".
