#!/usr/bin/env python3
"""Dynamic int8 quantization of a decoder-only ONNX export.

Usage:
    python quantize_causal_lm_int8.py <fp32 model.onnx> <out dir>
        [--weight-only [--block-size N] [--accuracy-level N] | --per-channel]
        [--quantize-embedding] [--quantize-lm-head] [--exclude REGEX ...]

Example (the Plume128k int8 graph):
    python quantize_causal_lm_int8.py Plume128k-onnx/model.onnx Plume128k-onnx/int8 \
        --per-channel

The default is dynamic quantization: int8 weights, and activations quantized
per tensor at run time. ``--weight-only`` writes ``MatMulNBits`` nodes with
8-bit weights in blocks of ``--block-size``. ``--accuracy-level 4`` lets
ONNX Runtime quantize the activations per block at run time, which is the
fast CPU kernel; without it the activations stay fp32 and the CPU kernel is
several times slower than the fp32 graph.
Only the weight MatMuls are quantized. By default the
token embedding (a ``Gather``) and the output projection (the MatMul that
produces ``logits``) stay fp32: quantizing them with one scale per tensor
makes a large-vocabulary Gemma model repeat itself, because the logits of
close candidates then differ by less than the quantization step. ``--exclude``
keeps every further node whose name matches in fp32.

The graph is written as ``model_quantized.onnx`` with its weights in
``model_quantized.onnx_data``, the names ``optimum`` gives an int8 export.
The tokenizer and config files next to the input graph are copied into the
output directory, so the directory is a complete ``int8/`` folder.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import onnx
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

GRAPH = "model_quantized.onnx"
BLOB = "model_quantized.onnx_data"
SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json")


def lm_head_node(graph: onnx.GraphProto) -> str:
    """The MatMul nearest to the ``logits`` output, walking producers back."""
    producer = {out: node for node in graph.node for out in node.output}
    pending = ["logits"]
    seen = set()
    while pending:
        name = pending.pop(0)
        node = producer.get(name)
        if node is None or node.name in seen:
            continue
        if node.op_type == "MatMul":
            return node.name
        seen.add(node.name)
        pending.extend(node.input)
    raise ValueError("no MatMul produces the logits output")


def rename_blob(graph_path: Path) -> None:
    """Point every external tensor at ``BLOB`` and rename the file to match."""
    model = onnx.load(str(graph_path), load_external_data=False)
    locations = set()
    for tensor in model.graph.initializer:
        if uses_external_data(tensor):
            locations.add(ExternalDataInfo(tensor).location)
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = BLOB
    if len(locations) != 1:
        raise ValueError(f"expected one external data file, found {sorted(locations)}")
    (graph_path.parent / locations.pop()).rename(graph_path.parent / BLOB)
    onnx.save(model, str(graph_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--weight-only", action="store_true")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--accuracy-level", type=int)
    ap.add_argument("--quantize-embedding", action="store_true")
    ap.add_argument("--quantize-lm-head", action="store_true")
    ap.add_argument("--exclude", action="append", default=[])
    args = ap.parse_args()

    graph = onnx.load(str(args.model), load_external_data=False).graph
    exclude = [node.name for node in graph.node
               if any(re.search(pattern, node.name) for pattern in args.exclude)]
    if not args.quantize_lm_head:
        exclude.append(lm_head_node(graph))
    op_types = ["MatMul", "Gather"] if args.quantize_embedding else ["MatMul"]
    print(f"op types {op_types}; {len(exclude)} nodes kept fp32: {exclude[:8]}",
          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / GRAPH
    if args.weight_only:
        quantizer = MatMulNBitsQuantizer(
            str(args.model), bits=8, block_size=args.block_size,
            is_symmetric=True, accuracy_level=args.accuracy_level,
            nodes_to_exclude=exclude,
            op_types_to_quantize=tuple(op_types))
        quantizer.process()
        quantizer.model.save_model_to_file(str(out), use_external_data_format=True)
    else:
        quantize_dynamic(
            args.model, out,
            op_types_to_quantize=op_types,
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=exclude,
            use_external_data_format=True,
            extra_options={"MatMulConstBOnly": True},
        )
    rename_blob(out)
    for name in SIDE_FILES:
        source = args.model.parent / name
        if source.exists():
            shutil.copy(source, args.out_dir / name)
    print(f"wrote {out} and {args.out_dir / BLOB}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
