#!/usr/bin/env python3
"""Quantize an optimum seq2seq ONNX export (encoder, decoder, decoder with
past) to int8, one graph at a time.

ONNX Runtime dynamic quantization, QInt8, per channel, full range, subgraphs
included. Each int8 graph keeps its weights in one external data file named
`<graph>.onnx_data`, the name `optimum` and linguonnx look for.

usage: quantize_seq2seq_int8.py <fp32 dir> <int8 dir>
"""
import sys
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

GRAPHS = ("encoder_model", "decoder_model", "decoder_with_past_model")


def quantize(fp32_dir: Path, int8_dir: Path) -> None:
    int8_dir.mkdir(parents=True, exist_ok=True)
    for graph in GRAPHS:
        with tempfile.TemporaryDirectory(dir=int8_dir) as scratch:
            raw = Path(scratch) / f"{graph}.onnx"
            quantize_dynamic(str(fp32_dir / f"{graph}.onnx"), str(raw),
                             weight_type=QuantType.QInt8, per_channel=True,
                             reduce_range=False, use_external_data_format=True,
                             extra_options={"EnableSubgraph": True})
            # ONNX Runtime names the data file `<graph>.onnx.data`;
            # re-save it under the `<graph>.onnx_data` name.
            model = onnx.load(str(raw), load_external_data=True)
            onnx.save_model(model, str(int8_dir / f"{graph}.onnx"),
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location=f"{graph}.onnx_data", size_threshold=1024)


if __name__ == "__main__":
    quantize(Path(sys.argv[1]), Path(sys.argv[2]))
