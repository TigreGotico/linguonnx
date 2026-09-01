#!/usr/bin/env python3
"""
Export an AI4Bharat IndicTrans2 checkpoint (custom `IndicTransForConditionalGeneration`
architecture, trust_remote_code) to ONNX via optimum, fp32 + int8 dynamic quantization,
full graph triple (encoder / decoder / decoder_with_past).

IndicTrans2's custom model class is architecturally an M2M100 clone (pre-norm
transformer encoder-decoder, separate source/target sentencepiece vocabs, decoder_start
token != bos). `optimum.exporters.tasks.TasksManager` does not know the "IndicTrans"
model_type out of the box, so we register a config for it derived from
`M2M100OnnxConfig`, keyed on the `model_type` string found in config.json ("IndicTrans").

Usage:
    python export_indictrans2.py <hf_repo_id> <out_dir> [--skip-quantize]

Example:
    python export_indictrans2.py ai4bharat/indictrans2-en-indic-1B /media/data/it2_export/en-indic-1B
"""
import argparse
import functools
import json
import os
import shutil
import sys

import onnx
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import M2M100OnnxConfig
from optimum.exporters.tasks import TasksManager
from optimum.utils import NormalizedSeq2SeqConfig
from optimum.utils.normalized_config import NormalizedConfigManager


class IndicTransOnnxConfig(M2M100OnnxConfig):
    """Same graph shape as M2M100: encoder-decoder seq2seq with self/cross attention
    + KV cache. IndicTransConfig uses M2M100/BART-style field names already
    (encoder_layers, decoder_layers, encoder_attention_heads, decoder_attention_heads,
    encoder_ffn_dim/decoder_ffn_dim, decoder_start_token_id != bos), which is exactly
    what M2M100OnnxConfig's NormalizedConfig expects -- verified against
    configuration_indictrans.py for the en-indic-1B checkpoint before use."""
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        encoder_num_attention_heads="encoder_attention_heads",
        decoder_num_attention_heads="decoder_attention_heads",
        hidden_size="encoder_embed_dim",
        eos_token_id="eos_token_id",
        # IndicTrans2 has SEPARATE source/target sentencepiece vocabs
        # (encoder_vocab_size=32322 vs decoder_vocab_size/vocab_size=122672 for
        # en-indic-dist-200M). optimum's DummyTextInputGenerator uses a single
        # NormalizedConfig.vocab_size for ALL generated input_ids (encoder AND
        # decoder), sampling token ids up to that bound. Left at the default
        # "vocab_size" (=122672, the decoder/target vocab), it samples encoder
        # input_ids as high as ~122k against an encoder embedding table sized
        # for only 32322 rows -> "IndexError: index out of range in self" deep
        # inside torch.embedding during tracing. Pin it to the SMALLER of the
        # two vocabs (encoder_vocab_size) so generated ids are always in-range
        # for both embedding tables; real translation quality is unaffected,
        # dummy tracing inputs never need to hit real vocabulary entries.
        vocab_size="encoder_vocab_size",
    )


def register():
    # optimum.exporters.onnx.convert.onnx_export_from_model does:
    #     model_type = model.config.model_type.replace("_", "-")
    # with NO lowercasing. IndicTrans2's config.json has "model_type": "IndicTrans"
    # (mixed case, no underscore) -- so the registration key must be the literal
    # string "IndicTrans", not "indictrans". Get this wrong and optimum falls through
    # to its "custom or unsupported architecture" error even though a config is
    # registered, because the dict lookup is case-sensitive and never matches.
    # ... except optimum.exporters.tasks.TasksManager.get_supported_tasks_for_model_type
    # ALSO does `model_type.lower()` before its own dict lookup into the very same
    # `_SUPPORTED_MODEL_TYPE` dict -- so both casings must resolve, or one of the two
    # call sites (onnx_export_from_model vs. main_export's early task-validation) will
    # KeyError even though the "other" one succeeds. Register under both keys.
    # IMPORTANT: the dict values here are not the bare config class -- upstream
    # entries (e.g. 'm2m-100') are functools.partial(ConfigClass, task=..., use_past=...),
    # produced by TasksManager's own registration decorators, which is how each
    # task string gets bound into the constructed OnnxConfig's self.task.
    # A first version of this function did 'task: IndicTransOnnxConfig' (the bare
    # class, same object for every task key). That resolves fine through
    # TasksManager.get_exporter_config_constructor (which just does a dict
    # lookup), but calling the bare class without an explicit task= kwarg falls
    # back to its default ('feature-extraction') regardless of which task was
    # actually requested -- so a 'text2text-generation-with-past' export silently
    # got self.task=='feature-extraction', self.use_past==False, and
    # cfg.outputs=={'last_hidden_state'} for BOTH the encoder AND decoder
    # submodels (the decoder submodel is IndicTransForConditionalGeneration,
    # whose real forward returns {'logits': ...} -- filtering that dict against
    # the wrong expected key 'last_hidden_state' finds nothing, and the ONNX
    # tracer ends up with a graph with zero outputs: 'number of output names
    # provided (1) exceeded number of outputs (0)'). Rebuild the partials from the
    # real 'm2m-100' entries so each task/use_past combination is preserved,
    # swapping only the config class.
    onnx_configs = {}
    for task, ctor in TasksManager._SUPPORTED_MODEL_TYPE["m2m-100"]["onnx"].items():
        kwargs = dict(ctor.keywords) if isinstance(ctor, functools.partial) else {"task": task}
        onnx_configs[task] = functools.partial(IndicTransOnnxConfig, **kwargs)
    for model_type_key in ("IndicTrans", "indictrans"):
        TasksManager._SUPPORTED_MODEL_TYPE[model_type_key] = {"onnx": onnx_configs}

    # Separate registry, separate bug: optimum.onnxruntime.base.ORTEncoder /
    # ORTDecoder look up NormalizedConfigManager._conf[config.model_type] at
    # *inference* load time (ORTModelForSeq2SeqLM.from_pretrained), independent
    # of everything registered above for the export path. Its key lookup does
    # `model_type.replace("_", "-")` with NO lowercasing, so -- same trap as
    # TasksManager -- it needs the literal "IndicTrans" key. Left unregistered,
    # any ORTModelForSeq2SeqLM.from_pretrained() over an exported IndicTrans2
    # graph fails with "IndicTrans model type is not supported yet in
    # NormalizedConfig" even though the ONNX files themselves are fine; this is
    # what verify_clean_load() below actually exercises, so register it too.
    NormalizedConfigManager._conf["IndicTrans"] = NormalizedSeq2SeqConfig.with_args(
        encoder_num_attention_heads="encoder_attention_heads",
        decoder_num_attention_heads="decoder_attention_heads",
        hidden_size="encoder_embed_dim",
    )


def fix_onnx_data(onnx_path: str):
    """optimum/onnxruntime writes external-data tensors to `<name>.onnx.data` and points
    the proto's `location` field at that same string. `from_pretrained` / ORTModel however
    look for `<name>.onnx_data` (underscore, no second dot) -- this is the well known
    optimum/ORT naming mismatch for models above the 2GB protobuf limit. Must rename the
    file on disk AND rewrite `location` inside the .onnx proto, or loading silently fails
    (or worse, silently loads stale/wrong weights if a same-named file exists from a
    previous export)."""
    data_old = onnx_path + ".data"
    data_new = onnx_path.rsplit(".onnx", 1)[0] + ".onnx_data"
    if not os.path.exists(data_old):
        return  # small graph, no external data -- nothing to do
    model = onnx.load(onnx_path, load_external_data=False)
    changed = False
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value == os.path.basename(data_old):
                entry.value = os.path.basename(data_new)
                changed = True
    if os.path.abspath(data_old) != os.path.abspath(data_new):
        shutil.move(data_old, data_new)
    if changed:
        onnx.save(model, onnx_path)
    print(f"  fixed external data: {os.path.basename(data_old)} -> {os.path.basename(data_new)}")


def verify_clean_load(export_dir: str, subfolder: str = None):
    """Load from a CLEARED cache dir to prove the .onnx_data rewrite actually works,
    not just that the files exist on disk."""
    import tempfile
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        kwargs = {"trust_remote_code": True}
        if subfolder:
            kwargs["subfolder"] = subfolder
        tok = AutoTokenizer.from_pretrained(export_dir, **kwargs)
        model = ORTModelForSeq2SeqLM.from_pretrained(export_dir, use_cache=True, **kwargs)
        ids = tok("eng_Latn hin_Deva Hello, how are you?", return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=20, num_beams=1)
        print("  clean-cache load OK, greedy sample decode:", tok.decode(out[0], skip_special_tokens=True))


def copy_tokenizer_files(repo_id: str, dest_dir: str):
    """optimum's main_export only writes the ONNX graphs + generation_config.json --
    it does NOT copy IndicTrans2's custom trust_remote_code siblings (the tokenizer
    class file, the two SentencePiece models, the two source/target vocab dicts, or
    tokenizer_config.json / special_tokens_map.json). Without these,
    AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True) fails with
    "Tokenizer class IndicTransTokenizer does not exist or is not currently imported"
    because the .py defining that class was never copied next to the export -- found
    the hard way, by verify_clean_load() failing after a full successful fp32 export.
    Pull every non-weight file from the HF cache snapshot for repo_id and copy it in."""
    from huggingface_hub import snapshot_download

    src_dir = snapshot_download(repo_id)
    skip_suffixes = (".safetensors", ".bin", ".msgpack", ".h5")
    for name in os.listdir(src_dir):
        if name.endswith(skip_suffixes):
            continue
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(dest_dir, name))
    print(f"  copied tokenizer/custom-code siblings from {src_dir} -> {dest_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("out_dir")
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    register()

    fp32_dir = os.path.join(args.out_dir, "fp32")
    os.makedirs(fp32_dir, exist_ok=True)

    print(f"== exporting {args.repo_id} (fp32) -> {fp32_dir}")
    main_export(
        model_name_or_path=args.repo_id,
        output=fp32_dir,
        task="text2text-generation-with-past",
        trust_remote_code=True,
        # optimum's post-processing step merges decoder_model.onnx and
        # decoder_with_past_model.onnx into a single decoder_model_merged.onnx.
        # At 1B scale that merged proto exceeds the 2GB protobuf limit and
        # optimum.onnx.graph_transformations.check_and_save_model does not use
        # external data for it -> google.protobuf.message.EncodeError: Failed
        # to serialize proto, wrapped by optimum as "Unable to merge decoders"
        # then "The post-processing of the ONNX export failed" (worked fine at
        # 200M scale, where the merged proto stays under 2GB). The merged file
        # is not part of the required graph triple (encoder_model.onnx,
        # decoder_model.onnx, decoder_with_past_model.onnx) -- skip it.
        no_post_process=True,
    )
    for f in os.listdir(fp32_dir):
        if f.endswith(".onnx"):
            fix_onnx_data(os.path.join(fp32_dir, f))
    copy_tokenizer_files(args.repo_id, fp32_dir)

    if not args.skip_quantize:
        int8_dir = os.path.join(args.out_dir, "int8")
        os.makedirs(int8_dir, exist_ok=True)
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        for graph in ["encoder_model", "decoder_model", "decoder_with_past_model"]:
            src = os.path.join(fp32_dir, f"{graph}.onnx")
            if not os.path.exists(src):
                continue
            print(f"== quantizing {graph}")
            quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name=f"{graph}.onnx")
            quantizer.quantize(save_dir=int8_dir, quantization_config=qconfig)
        # copy tokenizer / config / custom code siblings into int8/ too
        for f in os.listdir(fp32_dir):
            if not f.endswith((".onnx", ".onnx_data")):
                shutil.copy(os.path.join(fp32_dir, f), os.path.join(int8_dir, f))
        for f in os.listdir(int8_dir):
            if f.endswith(".onnx"):
                fix_onnx_data(os.path.join(int8_dir, f))

    print("== verifying fp32 load from cleared cache")
    verify_clean_load(fp32_dir)
    if not args.skip_quantize:
        print("== verifying int8 load from cleared cache")
        verify_clean_load(int8_dir)


if __name__ == "__main__":
    sys.exit(main())
