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
        if not rows:
            continue
        assert rows[0][1] == str(case["length"]), case["input"]
        assert rows[0][2].split()[0] == f"`{case['detected']}`", case["input"]
        assert rows[0][3] == f"{case['confidence']:.2f}", case["input"]
    shown = [c for c in RESULTS["detect"]["galician_short_text"] if table_rows(text, re.escape(f"`{c['input']}`"))]
    assert len(shown) >= 3
