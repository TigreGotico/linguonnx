# Install and run
<!-- agent-hint: extras, GPU providers, the model cache and the development commands. Verify: `grep -n "optional-dependencies" -A40 pyproject.toml` lists each extra named here. -->

```bash
pip install linguonnx
pip install linguonnx[distance]   # adds orthography2ipa, for pivot ranking
pip install linguonnx[rules]      # adds pt2barranquenho and pt2mirandese
```

The runtime dependencies are `onnxruntime`, `numpy`, `huggingface_hub`,
`linguatag` and `sentencepiece`. The library does not use torch. The
encoder-decoder generation loop, with beam search and the KV cache, runs on the
raw ONNX graphs.

## Extras

| Extra | Adds | Use |
| --- | --- | --- |
| `distance` | `orthography2ipa` | Pivot ranking by phonological distance |
| `rules` | `pt2barranquenho`, `pt2mirandese` | The rule-based translators |
| `indic` | `regex`, `sacremoses`, `indic-nlp-library-itt` | IndicTrans2 preprocessing |
| `opennmt` | `sacremoses`, `subword-nmt` | The Proxecto Nós models |
| `fast-tokenizers` | `tokenizers`, `regex` | AfroLID, the Softcatalà Pegasus pairs and AraT5 (`tokenizers`), Plume (`regex`) |
| `test` | the test tools | Development |

## Run on a GPU

Each session runs on `CPUExecutionProvider` by default. `onnxruntime` and
`onnxruntime-gpu` use the same import name, and you must not install both. For
this reason the `gpu` extra installs nothing. To use a GPU, replace the runtime
in the same environment:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
export LINGUONNX_ONNX_PROVIDERS=auto   # or a CSV list, for example CUDAExecutionProvider
```

`auto` selects the best provider that the installed ONNX Runtime build has, and
falls back to CPU. When a requested provider does not start, for example a CUDA
build on a host with no CUDA device, ONNX Runtime does not raise. It runs the
next provider in the list. `linguonnx` compares the active provider with the
requested provider after it builds each session, and logs a warning when they
differ. The code is in `linguonnx/providers.py`.

A model architecture runs on a non-CPU provider only when the provider supports
each op in the exported graph. Test each architecture on the real hardware.

## The model cache

Models download from Hugging Face on first use, into
`~/.cache/linguonnx/models/<model_id>/`. On a server the weights can be tens of
gigabytes, and `$HOME` is frequently on a small volume. Set `LINGUONNX_CACHE`
to move the cache:

```bash
export LINGUONNX_CACHE=/mnt/bulk/linguonnx
```

## Use it from OpenVoiceOS

[`ovos-plugin-linguonnx`](https://github.com/OpenVoiceOS/ovos-plugin-linguonnx)
gives two OVOS plugins from one install:

- a language detector: `opm.lang.detect`, id `ovos-lang-detect-plugin-linguonnx`
- a translator: `opm.lang.translate`, id `ovos-translate-plugin-linguonnx`

Both plugins load their models on first use. Each argument of `load_detector`
and `load_translator` is available in `mycroft.conf`.

## Development

```bash
uv pip install -e .[test]
pytest test/ -m "not network"     # unit tests: no download, no model
pytest test/                      # also the tests that download real models
python scripts/check_docs.py      # run each docs example and compare each printed value
```

The routing and decoding tests use no real model. The graph is data. The decode
loop runs on a small ONNX seq2seq that the test file builds, with the same
`past_key_values.*` and `present.*` names as the real exports. The tests with
the `network` mark compare the decoder with `transformers` and `optimum` on the
real graphs, string for string. Those two packages are test dependencies only.
`linguonnx/` must not import them.

`scripts/check_docs.py` runs the Python blocks of the README and of each docs
page, in order. A `print(...)` line with a `# ` comment is a check: the printed
value must equal the comment, and `...` in the comment matches any text. The
"Docs check" workflow (`.github/workflows/docs-check.yml`) runs it with a
cached model directory.

---
[← Prev](index.md) · [Home](index.md) · [Next →](detect.md)
