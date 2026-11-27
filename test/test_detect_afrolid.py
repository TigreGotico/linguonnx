"""AfroLID v1.5 through linguonnx against the upstream PyTorch model.

``test/data/afrolid_flores_reference.json`` holds FLORES-200 devtest
sentences (CC-BY-SA-4.0, NLLB Team et al. 2022) with what the unmodified
``UBC-NLP/afrolid_1.5`` checkpoint produced for them through ``transformers``:
the input ids of its own SentencePiece tokenizer and its top-5 labels and
probabilities. The fp32 export must match both; int8 must keep the top-1.
"""

import json
from pathlib import Path

import pytest

from linguonnx import load_detector
from linguonnx.detect.labels import to_bcp47

pytestmark = pytest.mark.network

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "afrolid_flores_reference.json")
    .read_text(encoding="utf-8"))
ROWS = REFERENCE["rows"]


def _as_tags(top5):
    out = {}
    for label, p in top5:
        out.setdefault(to_bcp47(label), p)
    return out


@pytest.fixture(scope="module")
def fp32():
    return load_detector("afrolid_1.5")


@pytest.fixture(scope="module")
def int8():
    return load_detector("afrolid_1.5-int8")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['flores']}:{r['line']}")
def test_fp32_matches_the_upstream_model(fp32, row):
    assert fp32._classifier.encode(row["text"])[0].tolist() == row["input_ids"]
    label, confidence = fp32.detect_raw(row["text"])
    assert label == row["top5"][0][0]
    assert confidence == pytest.approx(row["top5"][0][1], abs=1e-3)
    want = _as_tags(row["top5"])
    got = fp32.detect_probs(row["text"], top_k=5)
    assert list(got) == list(want)
    assert list(got.values()) == pytest.approx(list(want.values()), abs=1e-3)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['flores']}:{r['line']}")
def test_int8_keeps_the_upstream_top1(int8, row):
    assert int8.detect_raw(row["text"])[0] == row["top5"][0][0]
