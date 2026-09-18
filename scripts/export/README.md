# IndicTrans2 → ONNX export

`export_indictrans2_onnx.py` converts an AI4Bharat IndicTrans2 checkpoint
(`ai4bharat/indictrans2-*`) to ONNX via `optimum`, producing the full graph
triple (`encoder_model.onnx`, `decoder_model.onnx`,
`decoder_with_past_model.onnx`) in both fp32 and dynamic int8.

This is a one-off conversion tool, not a runtime dependency of `linguonnx` —
it is kept here so the recipe is not lost again (it previously lived only in a
worktree that got cleaned up, and had to be rediscovered from scratch).

## Why this needs a custom recipe

IndicTrans2 ships a **custom `trust_remote_code` architecture**
(`IndicTransForConditionalGeneration`, `model_type: "IndicTrans"` in
`config.json`). `optimum.exporters.tasks.TasksManager` does not know this
model type, and will refuse to export it ("custom or unsupported
architecture") unless a matching `OnnxConfig` is registered first.

Architecturally, IndicTrans2 is an M2M100 clone: pre-norm transformer
encoder-decoder, separate source/target SentencePiece vocabularies,
`decoder_start_token_id != bos`. `configuration_indictrans.py` uses the same
field names as M2M100/BART (`encoder_layers`, `decoder_layers`,
`encoder_attention_heads`, `encoder_ffn_dim`, etc.), so the export config can
be derived directly from `optimum.exporters.onnx.model_configs.M2M100OnnxConfig`
— verify this against `configuration_indictrans.py` for whichever checkpoint
you're exporting before trusting it; do not assume it holds without checking
if AI4Bharat ever changes the custom modelling code.

### The registration gotcha (cost real debugging time — read this before touching the script)

`TasksManager._SUPPORTED_MODEL_TYPE` is looked up by **two different call
sites with two different casing conventions**, against the very same dict:

- `optimum.exporters.onnx.convert.onnx_export_from_model` builds
  `model_type = model.config.model_type.replace("_", "-")` — **no
  lowercasing**. IndicTrans2's `config.json` has `"model_type": "IndicTrans"`,
  so this path needs the key `"IndicTrans"` (exact case) registered.
- `optimum.exporters.tasks.TasksManager.get_supported_tasks_for_model_type`
  (called from `main_export`'s early task-validation) does
  `model_type.lower().replace("_", "-")` before its lookup into the **same**
  dict — this path needs the key `"indictrans"` (lowercase).

Register the config under **both** keys, or one of the two call sites raises
a `KeyError`/`ValueError` even though the other succeeds. See `register()` in
the script.

## Pin `optimum`/`transformers`/`torch` -- the defaults `pip`/`uv` resolve are broken

**Symptom: `KeyError: 'm2m-100'`**, or later a total unrelated-looking
`ModuleNotFoundError: No module named 'onnxscript'`, or a `RuntimeError` deep
in `onnx.version_converter` about `LayerNormalization` having no earlier
opset. All three come from installing whatever `optimum`/`torch` versions are
current instead of the versions this script's registration code was written
against.

`optimum>=2.0` restructured `TasksManager._SUPPORTED_MODEL_TYPE` entirely (the
`"m2m-100"` key this script reads to derive its own registration no longer
exists in that form). Separately, `torch>=2.5`-ish made the dynamo-based ONNX
exporter the default, which needs `onnxscript` and does not behave like the
TorchScript-tracing exporter `optimum==1.24` assumes -- version-converting the
traced graph down to an old opset chokes on ops like `LayerNormalization`.

Pin: `optimum==1.24.0`, `transformers==4.45.2`, `torch==2.4.1`,
`huggingface_hub<1.0`. Re-check these against whatever is current before
reusing this script; the whole point of this note is that "just install the
latest" quietly breaks it.

## `NORMALIZED_CONFIG_CLASS` must remap `vocab_size`, not just layer/head counts

**Symptom: `IndexError: index out of range in self`** raised from deep inside
`torch.embedding`, during the encoder tracing step (`== exporting ... (fp32)`
prints, then the traceback lands in `modeling_indictrans.py`'s
`embed_tokens(input_ids)` call).

IndicTrans2 has **separate source/target SentencePiece vocabularies**
(`encoder_vocab_size` vs `vocab_size`/`decoder_vocab_size` in `config.json` --
32322 vs 122672 for `en-indic-dist-200M`). `optimum`'s dummy-input generator
reads a single `NormalizedConfig.vocab_size` and samples random token ids up
to that bound for **both** the encoder and decoder dummy inputs. Left at the
default (`vocab_size`, i.e. the larger decoder vocab), it samples encoder
`input_ids` well past the encoder's embedding table size.

Fix: base `IndicTransOnnxConfig.NORMALIZED_CONFIG_CLASS` on
`NormalizedSeq2SeqConfig` (matching `M2M100OnnxConfig`'s real parent -- not
`NormalizedTextConfig`, which was the first thing tried and silently accepted
the wrong field names via `allow_new`), and remap
`vocab_size="encoder_vocab_size"` (the smaller of the two, always safe for
both embedding tables since real translation quality is unaffected by dummy
tracing inputs never hitting real vocabulary entries).

## `register()` must preserve task/`use_past` binding, not just the class

**Symptom: `RuntimeError: number of output names provided (1) exceeded
number of outputs (0)`**, raised from `torch.onnx.utils._set_input_and_output_names`
during the **decoder** submodel's export step (the encoder step above it
succeeds). This is almost certainly why the two export attempts referenced in
this PR's original description died without a clear error.

Real `optimum` registry entries (e.g. `"m2m-100"`) are NOT the bare config
class -- they are `functools.partial(ConfigClass, task=..., use_past=...)`,
produced by `optimum`'s own registration decorators, which is how each task
string gets bound into the constructed `OnnxConfig`'s `self.task`. A version
of `register()` that maps every task key to the bare `IndicTransOnnxConfig`
class resolves fine through `TasksManager.get_exporter_config_constructor`
(a plain dict lookup succeeds either way), but instantiating the bare class
without an explicit `task=` kwarg falls back to its constructor default,
`"feature-extraction"`, regardless of which task was actually requested. That
silently gives *every* submodel (encoder AND decoder) `self.task ==
"feature-extraction"`, so the decoder submodel's `cfg.outputs` stays
`{"last_hidden_state"}` -- but the decoder submodel is
`IndicTransForConditionalGeneration`, whose real forward returns `{"logits":
...}`. Output filtering in `optimum`'s `ModelPatcher` matches nothing, and the
ONNX tracer ends up with a graph that has zero outputs.

Fix: rebuild the `onnx_configs` dict from the real `"m2m-100"` entries'
`functools.partial` objects, swapping only the class, so every task/`use_past`
combination is preserved exactly as `optimum` itself would register it. See
`register()` in the script.

## `NormalizedConfigManager` is a second, separate registry for the *runtime* load path

**Symptom: `KeyError: 'IndicTrans model type is not supported yet in
NormalizedConfig'`**, raised from `ORTModelForSeq2SeqLM.from_pretrained()` --
notably **after** export has already fully succeeded and the `.onnx` files on
disk are fine. This only shows up when something actually loads the exported
model (`verify_clean_load()`, or any downstream consumer using `optimum.onnxruntime`).

`optimum.utils.normalized_config.NormalizedConfigManager._conf` is a
completely separate dict from `TasksManager._SUPPORTED_MODEL_TYPE`, consulted
by `ORTEncoder`/`ORTDecoder.__init__` at inference-load time, not at export
time. It has the same no-lowercasing key convention (`model_type.replace("_",
"-")`), so it needs the literal `"IndicTrans"` key too. `register()` now also
sets `NormalizedConfigManager._conf["IndicTrans"]`.

## Custom `trust_remote_code` siblings are not copied by `main_export()`

**Symptom: `ValueError: Tokenizer class IndicTransTokenizer does not exist or
is not currently imported`**, raised from `AutoTokenizer.from_pretrained()` --
again after ONNX export itself has already succeeded.

`optimum.exporters.onnx.main_export()` writes only the ONNX graphs and
`generation_config.json`. It does not copy IndicTrans2's custom
`trust_remote_code` siblings: the tokenizer class file
(`tokenization_indictrans.py`), the two SentencePiece models (`model.SRC`,
`model.TGT`), the two source/target vocab dicts (`dict.SRC.json`,
`dict.TGT.json`), `tokenizer_config.json`, or `special_tokens_map.json`.
Without these next to the export, nothing that needs the tokenizer (including
`verify_clean_load()`) can load it. `copy_tokenizer_files()` in the script
pulls every non-weight file from the source repo's HF cache snapshot into the
export directory after the fp32 export.

## Decoder merge post-processing fails at 1B scale -- skip it, it isn't required

**Symptom: `google.protobuf.message.EncodeError: Failed to serialize proto`**,
wrapped by `optimum` as `Exception: Unable to merge decoders. Detailed error:
Failed to serialize proto`, then `Exception: The post-processing of the ONNX
export failed`. Only shows up on the 1B checkpoints -- the 200M/320M models
stay under the limit.

`optimum`'s post-processing step merges `decoder_model.onnx` and
`decoder_with_past_model.onnx` into a single `decoder_model_merged.onnx` for
convenience. At 1B scale that merged proto exceeds the 2 GB protobuf limit,
and `optimum.onnx.graph_transformations.check_and_save_model` does not use
external data when saving it. The merged file is **not** part of the required
graph triple (`encoder_model.onnx`, `decoder_model.onnx`,
`decoder_with_past_model.onnx`) -- the script passes `no_post_process=True`
to `main_export()` and skips it entirely rather than working around the
protobuf limit for a file nothing needs.

## The `.onnx.data` trap (only shows up at real scale, e.g. the 1B checkpoints)

Any ONNX graph whose total tensor payload exceeds the 2 GB protobuf limit
gets its weights written to a companion **external-data** file. `onnx`/ORT's
exporter writes this as `<name>.onnx.data` and points the proto's internal
`location` field at that exact filename — but `transformers`/`optimum`'s
`from_pretrained` loader looks for `<name>.onnx_data` (underscore, single
dot). Get this wrong and:

- a plain `ls` of the export directory looks completely fine (the file is
  there, just under the "wrong" name for `from_pretrained` — a listing check
  is not sufficient), and
- loading fails only when you actually construct `ORTModelForSeq2SeqLM`.

Fix requires **both** halves:

1. Rename the file on disk, `<name>.onnx.data` → `<name>.onnx_data`.
2. Load the `.onnx` proto with `onnx.load(path, load_external_data=False)`,
   rewrite every `initializer[*].external_data[*].location` entry that still
   points at the old filename, and re-save the proto.

`fix_onnx_data()` in the script does this. **Always verify by loading from a
freshly cleared `HF_HOME`** (`verify_clean_load()`) — a load that "succeeds"
against a warm local cache can be hiding a `location` field that still points
at a file that happens to already be sitting next to it from a previous
export attempt.

## Usage

```bash
# fp32 + int8, full graph triple
python export_indictrans2_onnx.py ai4bharat/indictrans2-en-indic-1B /media/data/<workdir>/en-indic-1B

# fp32 only (fast sanity check against a small checkpoint before committing
# real compute to a 1B model)
python export_indictrans2_onnx.py ai4bharat/indictrans2-en-indic-dist-200M /media/data/<workdir>/en-indic-200M --skip-quantize
```

Requires a Python env with `optimum[onnxruntime]`, `transformers`,
`huggingface_hub<1.0` (as of writing, `transformers` 4.45.x hard-requires
`huggingface_hub<1.0`; check this pin against whatever versions are current
before reusing), `sentencepiece`, `onnx`, `onnxruntime`. Deliberately **not**
`IndicTransToolkit` — that package declares `transformers` as a hard
dependency and is not needed at export time; `linguonnx` vendors the
preprocessing it needs at runtime in
`linguonnx/translate/_indic_processor.py` and `preprocess.py`.

## After exporting — not automated here

This script only produces the ONNX graphs and verifies they load. It does
**not**:

- run parity checks (greedy + beam-4) against the original PyTorch model,
- run the differential/language-identity check across target languages,
- write the model card,
- publish to the Hub.

Those steps are gated separately per the campaign this was written for (see
the TigreGotico IndicTrans2 ONNX conversion effort) — do not publish a model
exported with this script without them.
