# lingonnx

CPU-first language technology models on ONNX Runtime, in the spirit of
[`onnx-asr`](https://github.com/istupakov/onnx-asr) (speech recognition) and
[`phoonnx`](https://github.com/TigreGotico/phoonnx) (text-to-speech).

This release covers language identification (`lingonnx.detect`), built on the
TigreGotico ONNX exports of four fastText classifiers: GlotLID, fastText's
classic lid.176, OpenLID and OpenLID-v2. Translation
(`lingonnx.translate`) is planned as a sibling package under the same
`lingonnx` namespace; it does not exist yet.

## Install

```bash
pip install lingonnx
```

Models download from HuggingFace on first use and are cached under
`~/.cache/lingonnx/models/<model_id>/`.

## How the model split works

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
Python, in `lingonnx/detect/hashing.py`, vendored from the reference
implementation published alongside the ONNX weights
(`glotlid_hash.py` in the HuggingFace repo; every model in the registry uses
the same hashing, differing only in the `nwords`/`minn`/`maxn`/`bucket`
values read from its `config.json`). It is a straight port of
fastText's `Dictionary` class, including the detail that trips up most
from-scratch reimplementations: fastText hashes each UTF-8 byte as a
**signed** `int8`, so bytes >= `0x80` get sign-extended before the FNV-1a
XOR. Every non-ASCII n-gram (i.e. every non-Latin-script language) hashes to
the wrong bucket if you "fix" that.

## API

```python
from lingonnx import load_detector

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
`lingonnx` itself is Apache-2.0 and downloads no model you did not name.

`glotlid-int8` is the default precision as well as the default family: the
HuggingFace repo's own parity testing found identical top-1 decisions against
the fp32 graph on a 78-sample multilingual set, so there is no accuracy reason
to prefer the larger file. The same holds for `openlid`; `lid176.int8.onnx`
does lose a little (96.6% vs 100% top-1 agreement with fastText on the
publisher's 59-language set), so use `lid176` fp32 when 125 MB is affordable.

### Label shapes

GlotLID, OpenLID and OpenLID-v2 emit `iso3_Script` labels (`eng_Latn`,
`glg_Latn`, `zho_Hans`). lid.176 emits bare ISO codes with no script
(`en`, `gl`, `pt`). `lingonnx` maps both to BCP-47 tags via
`lingonnx.detect.labels`; see that module's docstring for the exact rules.

## Hierarchical softmax

lid.176 was trained with fastText's hierarchical softmax (`loss=hs`). Under
`hs` the rows of the output matrix are **not** per-label scores: each row is a
binary classifier for one internal node of a Huffman tree built over the label
frequencies, and a label's probability is the product of the sigmoid (or
1-sigmoid) values along its root-to-leaf path. Running the ordinary
`MatMul -> Softmax` recipe on such a model produces well-formed numbers and
0% agreement with fastText.

So the lid.176 ONNX graph ends in `Sigmoid` over the Huffman nodes, and
`lingonnx.detect.hs` walks the paths in Python (the tree is precomputed into
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

## Translation (planned)

`lingonnx.translate` is not implemented yet. The package layout leaves room
for it beside `lingonnx.detect` without restructuring anything already
shipped.

## Development

```bash
uv pip install -e .
pytest test/                      # unit tests only need the mocked ONNX session
pytest test/ -m network           # also runs the real end-to-end model downloads
```
