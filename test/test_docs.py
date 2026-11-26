"""The measured numbers in the docs equal benchmarks/results.json, which scripts/benchmark.py writes."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "benchmarks/results.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT / "docs" / "img"))
import plot_budgets  # noqa: E402


def page(name):
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def table_rows(text, first_cell):
    return [[c.strip() for c in line.strip("|").split("|")]
            for line in text.splitlines() if re.match(rf"\| {first_cell} \|", line)]


def test_budget_figure_data():
    committed = json.loads((ROOT / "docs/img/budgets.json").read_text(encoding="utf-8"))
    assert plot_budgets.budgets() == committed


def test_budget_alt_text_table_and_prose():
    text = page("routing.md")
    rows = plot_budgets.budgets()
    assert f'alt="{plot_budgets.alt(rows)}"' in text
    table = table_rows(text, r"(none|\d+)")
    assert table == [[r["budget"].removesuffix(" MB"), str(r["models"]), str(r["languages"])] for r in rows]
    full = rows[0]["models"]
    at500 = next(r for r in rows if r["budget"] == "500 MB")
    assert full == RESULTS["routing"]["selection_models"]
    assert f"non-commercial, {full} models" in text
    assert f"A 500 MB budget drops {full - at500['models']} of {full} models, but\nthose {full - at500['models']} include" in text
    assert f"the {at500['languages']} languages opus-mt" in text


def test_detect_table():
    text = page("detect.md")
    for case in RESULTS["detect"]["galician_short_text"]:
        rows = table_rows(text, re.escape(f"`{case['input']}`"))
        assert len(rows) == 1, f"no table row for {case['input']!r}"
        assert rows[0][1] == str(case["length"]), case["input"]
        assert rows[0][2].split()[0] == f"`{case['detected']}`", case["input"]
        assert rows[0][3] == f"{case['confidence']:.2f}", case["input"]


def test_index_figure_alt_text():
    assert f'alt="{plot_budgets.alt(plot_budgets.budgets())}"' in page("index.md")


def test_index_what_ships_counts_equal_the_registry():
    from collections import Counter

    from linguonnx.model_manager import list_models
    from linguonnx.translate import load_translator

    translate = list_models(kind="translate")
    precision = Counter(entry.get("precision") for entry in translate.values())
    models = {entry["hf_repo"] for entry in translate.values() if "hf_repo" in entry}
    rules = sum(1 for entry in translate.values() if entry.get("callable"))
    rows = dict(row[:2] for row in table_rows(page("index.md"), r"[A-Z][^|]*"))
    assert rows["Translation registry"] == (
        f"{len(translate)} entries: {precision['int8']} int8 and {precision['fp32']} fp32, "
        f"from {len(models)} models, and {rules} rule-based translators")
    assert rows["Language identification registry"] == f"{len(list_models(kind='lid'))} entries"
    languages = len(load_translator(count_cached_as_free=False).available_languages)
    assert rows["Reachable languages"] == f"{languages}, over the default graph"
