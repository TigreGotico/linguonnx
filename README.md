# lingonnx

CPU-first language technology models on ONNX Runtime, in the spirit of
[`onnx-asr`](https://github.com/istupakov/onnx-asr) (speech recognition) and
[`phoonnx`](https://github.com/TigreGotico/phoonnx) (text-to-speech).

This release covers language identification (`lingonnx.detect`), built on
the [`TigreGotico/glotlid-onnx`](https://huggingface.co/TigreGotico/glotlid-onnx)
export of [GlotLID](https://github.com/cisnlp/GlotLID). Translation
(`lingonnx.translate`) is planned as a sibling package under the same
`lingonnx` namespace; it does not exist yet.

## Install

```bash
pip install lingonnx
```

Models download from HuggingFace on first use and are cached under
`~/.cache/lingonnx/models/<model_id>/`.

## How the model split works

GlotLID is a fastText classifier. fastText's actual math - averaging
embedding rows and multiplying by the output matrix, then softmax - is
representable as an ONNX graph, so that part runs as a normal ONNX Runtime
session:

```
input_ids: int64[num_features]  ->  probs: float32[2102]
```

What ONNX *can't* do portably is fastText's own preprocessing: tokenizing
text and hashing character n-grams into `input_ids`. That half stays in
Python, in `lingonnx/detect/hashing.py`, vendored from the reference
implementation published alongside the ONNX weights
(`glotlid_hash.py` in the HuggingFace repo). It is a straight port of
fastText's `Dictionary` class, including the detail that trips up most
from-scratch reimplementations: fastText hashes each UTF-8 byte as a
**signed** `int8`, so bytes >= `0x80` get sign-extended before the FNV-1a
XOR. Every non-ASCII n-gram (i.e. every non-Latin-script language) hashes to
the wrong bucket if you "fix" that.

## API

```python
from lingonnx import load_detector

det = load_detector("glotlid-int8")   # or "glotlid"; downloads on first use

det.detect("bo dia, como estas?")        # -> "gl"              BCP-47 best guess
det.detect_probs("...", top_k=5)         # -> {"gl": 0.82, "pt": 0.11, ...}
det.detect_raw("...")                    # -> ("glg_Latn", 0.82)  native GlotLID label
det.available_languages                  # -> set of BCP-47 tags it can emit
```

## Models

| model_id       | file               | size   | precision | license    |
|----------------|--------------------|--------|-----------|------------|
| `glotlid-int8` | `glotlid.int8.onnx`| 419 MB | int8      | Apache-2.0 |
| `glotlid`      | `glotlid.onnx`     | 1.68 GB| fp32      | Apache-2.0 |

`glotlid-int8` is the default: the HuggingFace repo's own parity testing
found identical top-1 decisions against the fp32 graph on a 78-sample
multilingual set, so there is no accuracy reason to prefer the larger file.

Both cover the same 2102 GlotLID language-script labels (e.g. `eng_Latn`,
`zho_Hans`, `arb_Arab`). `lingonnx` maps these to BCP-47 tags
(`eng_Latn` -> `en`, `zho_Hans` -> `zh-Hans`) via `lingonnx.detect.labels`;
see that module's docstring for the exact script-subtag rules.

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
pytest test/ -m network           # also runs the real end-to-end model download
```
