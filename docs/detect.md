# Language identification

`linguonnx.detect` answers one question: what language is this text in?

```python
from linguonnx import load_detector

det = load_detector()                 # glotlid-int8, the default
# load_detector("lid176-int8") for another model: openlid-int8, openlid-v2-int8

print(det.detect("Egun on, zer moduz?"))          # 'eu' — best-guess BCP-47 tag
print(det.detect_probs("Bon dia a tothom", 3))    # {'ca': 0.998, ...}
print(det.detect_raw("Bon dia a tothom"))         # ('cat_Latn', 0.998)
print(len(det.available_languages))               # 2100 BCP-47 tags
print(det.loss)                                   # 'softmax'
```

`detect` and `detect_probs` give you BCP-47 tags, which is what the rest of a
voice stack wants. `detect_raw` gives you the model's own label untouched, for
when you need the exact variety and script the classifier chose.

## The four models

| model_id | file | size | labels | precision | licence |
|---|---|---|---|---|---|
| `glotlid-int8` | `glotlid.int8.onnx` | 425 MB | 2102 | int8 | Apache-2.0 |
| `glotlid` | `glotlid.onnx` | 1.68 GB | 2102 | fp32 | Apache-2.0 |
| `lid176-int8` | `lid176.int8.onnx` | 33 MB | 176 | int8 | CC-BY-SA-3.0 |
| `lid176` | `lid176.onnx` | 131 MB | 176 | fp32 | CC-BY-SA-3.0 |
| `openlid-int8` | `openlid.int8.onnx` | 310 MB | 201 | int8 | GPL-3.0 |
| `openlid` | `openlid.onnx` | 1.23 GB | 201 | fp32 | GPL-3.0 |
| `openlid-v2-int8` | `openlid-v2.int8.onnx` | 305 MB | 200 | int8 | GPL-3.0 |
| `openlid-v2` | `openlid-v2.onnx` | 1.22 GB | 200 | fp32 | GPL-3.0 |

[**GlotLID**](https://huggingface.co/TigreGotico/glotlid-onnx) has 2102
language-script labels, by far the widest coverage, and is the default. It is
also the only Apache-2.0 model here, which is the other reason it is the
default.

[**lid.176**](https://huggingface.co/TigreGotico/lid176-onnx) is Meta's
original fastText identifier: 176 languages and only 33 MB in int8, so it is
the one to reach for on a small device. It is a 16-dimension model and leans
on diacritics to separate close Romance languages, so feed it properly
accented text or it will guess.

[**OpenLID**](https://huggingface.co/TigreGotico/openlid-onnx) and
[**OpenLID-v2**](https://huggingface.co/TigreGotico/openlid-v2-onnx) cover 201
and 200 curated FLORES-200 varieties, trained on audited data. Prefer v2
unless you need parity with the original OpenLID paper. Both are GPL-3.0.

### Precision

int8 is the default because it is almost free. GlotLID's HuggingFace repo
reports identical top-1 decisions against the fp32 graph on a 78-sample
multilingual set, and the same holds for OpenLID. `lid176.int8.onnx` does lose
a little — 96.6% top-1 agreement with fastText against 100% for fp32 on the
publisher's 59-language set — which is what you would expect from quantising a
16-dimension model that was already small. Use `lid176` fp32 when 131 MB is
affordable.

## How a fastText model becomes an ONNX graph

Every model here is a fastText classifier. fastText's actual math — averaging
embedding rows, multiplying by the output matrix, softmax — is representable
as an ONNX graph, so that part runs as a normal ONNX Runtime session:

```
input_ids: int64[num_features]  ->  probs: float32[num_labels]
```

What ONNX cannot do portably is fastText's *preprocessing*: tokenising text
and hashing character n-grams into `input_ids`. That half stays in Python, in
`linguonnx/detect/hashing.py`, vendored from the reference implementation
published alongside the ONNX weights. Every model in the registry uses the
same hashing and differs only in the `nwords`/`minn`/`maxn`/`bucket` values
read from its `config.json`.

It is a straight port of fastText's `Dictionary` class, including the detail
that trips up most from-scratch reimplementations: fastText hashes each UTF-8
byte as a **signed** `int8`, so bytes `>= 0x80` are sign-extended before the
FNV-1a XOR. Every non-ASCII n-gram — which is to say every non-Latin-script
language — hashes to the wrong bucket if you "fix" that.

## Hierarchical softmax

lid.176 was trained with fastText's hierarchical softmax (`loss=hs`), and that
changes what the output matrix means. Under `hs` the rows are **not** per-label
scores. Each row is a binary classifier for one internal node of a Huffman tree
built over the label frequencies, and a label's probability is the product of
the sigmoid (or one minus the sigmoid) values along its root-to-leaf path.
Running the ordinary `MatMul -> Softmax` recipe on such a model produces
well-formed numbers and 0% agreement with fastText.

So the lid.176 ONNX graph ends in `Sigmoid` over the Huffman nodes, and
`linguonnx.detect.hs` walks the paths in Python. The tree is precomputed into
`hs_tree.json` at export time, and `build_tree()` reproduces the construction
from label counts.

Which path a model takes is decided by the `loss` field on its registry entry,
falling back to the `loss` in its own `config.json` and then to `softmax` —
never by the model's name. `det.loss` reports what was chosen.

## Varieties, and collapsing them

GlotLID labels individual varieties rather than macrolanguages. There are 11
Arabic and 8 Chinese varieties in its label set, so casual Arabic comes back as
`ars` (Najdi) or `ajp-Arab` (South Levantine) rather than `ar`, and Chinese may
come back as `yue-Hani` (Cantonese).

This is useful on its own — it is free, text-side dialect identification. But
most callers want a tag they can act on: a TTS voice, a translation target.
Both are available:

```python
print(det.detect_raw("وش لونك يا خوي"))                      # ('ars_Arab', 0.99)
print(det.detect("وش لونك يا خوي"))                          # 'ars'
print(det.detect("وش لونك يا خوي", collapse_varieties=True)) # 'ar'
```

`collapse_varieties` keeps a script subtag when it still disambiguates, so
`yue-Hani` collapses to `zh-Hani` rather than to a bare `zh`. Chinese labels
distinguish `Hans`/`Hant`/`Hani` — simplified, traditional and script-neutral —
and text written only in characters the two scripts share resolves to
`zh-Hani`, which is the honest answer rather than a coin flip.

## Labels and BCP-47

GlotLID, OpenLID and OpenLID-v2 emit `iso3_Script` labels (`eng_Latn`,
`glg_Latn`, `zho_Hans`). lid.176 emits bare ISO codes with no script (`en`,
`gl`, `pt`). `linguonnx.detect.labels` maps both shapes to BCP-47.

Most of the work is `langcodes.standardize_tag`: ISO 639-3 to 639-1 where a
two-letter code exists (`eng` → `en`), deprecated codes repaired (`iw` → `he`),
canonical replacements applied (`tgl` → `fil`), and a redundant script subtag
dropped (`eng-Latn` → `en`) while an informative one is kept (`srp-Cyrl` →
`sr-Cyrl`).

Two things it does not do, which the module owns:

- **Redundant scripts CLDR has no data about.** `standardize_tag` only drops a
  script subtag for a language CLDR's likely-subtags table covers, and that is
  a minority of GlotLID's labels — it leaves `ast-Latn`, `yo-Latn`, `ig-Latn`.
  Correct, but noisy. So the script is dropped when `Language.maximize()`
  guesses the same script anyway. `_ALWAYS_KEEP_SCRIPT` protects the languages
  this must not happen to: readers of Serbian and Mandarin need Cyrl/Latn and
  Hans/Hant kept apart even though CLDR calls one of each the likely default.
- **Macrolanguage varieties.** `arb`, `cmn` and `zho` always resolve to `ar`
  and `zh`, because GlotLID never emits a bare `ara` and downstream consumers
  expect the macrolanguage. Everything else is left alone unless you ask for
  `collapse_varieties`.

Because several raw labels can map onto the same BCP-47 tag,
`available_languages` is smaller than the label count: GlotLID's 2102 labels
yield 2100 distinct tags. `detect_probs` collapses duplicates the same way,
keeping the highest probability for a tag.

A handful of GlotLID labels are not valid language tags at all — `und_Kawi`,
`und_Nagm`, `tok_Latn` (Toki Pona), `eml_Latn` — and are emitted as-is with a
warning on the logger. Better a code you can look up than a silent guess.

## Short text is unreliable

Accuracy depends strongly on how much text you give the model. GlotLID scores
character n-grams, so a handful of words carries little signal and closely
related languages collapse into each other. Galician shows this clearly:

| input | length | detected | confidence |
|---|---|---|---|
| `Bo día` | 6 | `pap` (Papiamento) | 0.77 |
| `Bo día, como estás?` | 19 | `es` | 0.53 |
| `Bo día, o meu nome é Miro` | 25 | `gl` | 1.00 |
| `A lingua galega é unha lingua románica...` | 113 | `gl` | 1.00 |

The first two are wrong, and the second is wrong *confidently enough to look
plausible* — Galician and Spanish share most of their character n-grams, so a
short greeting genuinely does not distinguish them. Around 25 characters the
model becomes reliable for this pair.

Two practical consequences:

- **Check the probability, not just the label.** A top-1 score near 0.5 on a
  short string means the model is guessing between neighbours. `detect_probs`
  shows you what it was choosing between.
- **Orthography matters.** Stripping diacritics costs accuracy: the same
  greeting written `bo dia, como estas?` is detected as Kimbundu. If your input
  is ASCII-folded or lowercased upstream, expect worse results, and prefer a
  language hint over detection where you have one.

This is a property of n-gram language identification rather than a defect in
this export — the same behaviour is present in the original fastText models.
