"""Draw docs/img/budgets.svg: models kept and languages routable for each max_model_mb budget.

Run from the repo root: python docs/img/plot_budgets.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import save  # noqa: E402


def budgets(condition="base"):
    """The budget rows of one install: ``base`` (no extra) or ``all_extras``."""
    routing = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))["routing"]
    return [{"budget": "none" if b["max_model_mb"] is None else f"{b['max_model_mb']} MB",
             "models": b["models_kept"], "languages": b["languages_routable"]}
            for b in routing[condition]["budgets"]]


def draw(fig, theme, rows):
    models, languages = fig.subplots(1, 2, sharey=True)
    y = range(len(rows))
    for ax, key, title in ((models, "models", "Models kept"), (languages, "languages", "Languages routable")):
        values = [r[key] for r in rows]
        ax.barh(y, values, height=0.6, color=theme["series"][0] if key == "languages" else theme["muted"])
        for i, v in zip(y, values):
            ax.text(v + max(values) * 0.02, i, str(v), va="center", fontsize=8, color=theme["text"])
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    models.set_yticks(list(y), [r["budget"] for r in rows])
    models.invert_yaxis()
    models.set_ylabel("max_model_mb")


def alt(rows):
    full = rows[0]
    at = {r["budget"]: r for r in rows}
    return (f"On a base install, a 500 MB budget keeps {at['500 MB']['models']} of {full['models']} models but only "
            f"{at['500 MB']['languages']} of {full['languages']} languages; 300 MB keeps {at['300 MB']['languages']} languages.")


if __name__ == "__main__":
    rows = budgets()
    save(str(ROOT / "docs" / "img" / "budgets"), draw, rows, height=2.8)
    print(alt(rows))
