"""Write docs/reference.md from the ``__all__`` of each public module.

    python scripts/docs_reference.py [--check]

``--check`` exits 1 when the committed page differs from the generated one.
"""
import importlib
import inspect
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "docs" / "reference.md"
MODULES = ["linguonnx", "linguonnx.detect", "linguonnx.translate", "linguonnx.translate.graph",
           "linguonnx.translate.models", "linguonnx.translate.quality", "linguonnx.translate.distance",
           "linguonnx.translate.preprocess", "linguonnx.translate.decode",
           "linguonnx.translate.tokenizers", "linguonnx.model_manager", "linguonnx.providers",
           "linguonnx.limits"]
HEAD = """# Reference
<!-- agent-hint: every public name of linguonnx, from each module's __all__. Regenerate: `python scripts/docs_reference.py`. -->

Each table lists the names in the `__all__` of one module, with the first line of the docstring. A name with no docstring shows its kind only. The task pages explain how to use the names: [detect.md](detect.md), [translate.md](translate.md), [routing.md](routing.md) and [models.md](models.md).
"""
FOOT = "\n---\n[← Prev](licences.md) · [Home](index.md)\n"


def summary(obj) -> str:
    if inspect.isclass(obj) or inspect.isroutine(obj):
        doc = inspect.getdoc(obj)
        kind = "class" if inspect.isclass(obj) else "function"
        return doc.splitlines()[0].replace("|", "\\|") if doc else kind
    return f"constant (`{type(obj).__name__}`)"


def render() -> str:
    parts = [HEAD]
    for name in MODULES:
        module = importlib.import_module(name)
        names = sorted(getattr(module, "__all__", ()))
        if not names:
            continue
        parts.append(f"\n## `{name}`\n\n| Name | Summary |\n| :--- | :--- |")
        parts += [f"| `{n}` | {summary(getattr(module, n))} |" for n in names]
    return "\n".join(parts) + "\n" + FOOT


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if PAGE.exists() and PAGE.read_text(encoding="utf-8") == text else 1)
    PAGE.write_text(text, encoding="utf-8")
