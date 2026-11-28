# linguonnx

Language identification and machine translation on ONNX Runtime. The runtime
does not use torch.

## Repo map

| Path | Contents |
| --- | --- |
| `linguonnx/detect/` | The fastText LID models on ONNX |
| `linguonnx/translate/graph.py` | The routing graph: capabilities, policies, size budget, pivots |
| `linguonnx/translate/decode.py` | The encoder-decoder loop: greedy, beam search, KV cache |
| `linguonnx/translate/models.py`, `tokenizers.py`, `preprocess.py` | One translation model, its tokenizer and its pre- and post-processing |
| `linguonnx/model_index/*.json` | The registry. `scripts/sync_registry.py` writes it |
| `linguonnx/model_manager.py` | Downloads and the model cache (`LINGUONNX_CACHE`) |
| `benchmarks/results.json` | The measured numbers. `scripts/benchmark.py` writes it |
| `docs/` | The user docs. `docs/index.md` gives the reading order |
| `docs/img/plot_*.py` | One script for each figure, from `benchmarks/results.json` |
| `scripts/check_docs.py` | Runs each Python block of the docs and compares each printed value with its comment |

## Setup and tests

```bash
uv venv && uv pip install --prerelease=allow -e ".[test]"
pytest test/ -m "not network"            # offline, no model download
pytest test/test_docs.py                 # the docs tables and figure data equal benchmarks/results.json
python scripts/check_docs.py             # downloads glotlid-int8 and opus-mt-pt-en-int8
```

## Rules

- `linguonnx/` never imports `torch`, `transformers` or `optimum`. They are
  test dependencies.
- Runnability and coverage are registry data. Do not put a list of
  architecture names in `graph.py`.
- A number in the docs comes from `benchmarks/results.json` or from the
  registry, and a test compares it. A printed value in a docs example has its
  output in a `# ` comment, and `scripts/check_docs.py` compares it.
- After a registry change, update the constants in
  `test/test_translate_registry_numbers.py`. `test/test_docs.py` then checks
  the "What ships" table in `docs/index.md`.
