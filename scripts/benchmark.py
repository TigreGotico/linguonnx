"""Write the measured numbers of docs/routing.md and docs/detect.md to benchmarks/results.json.

    python scripts/benchmark.py            # routing budgets only, no download
    python scripts/benchmark.py --detect   # also the short-text detection rows (downloads glotlid-int8)
"""
import json
import sys
from pathlib import Path

from linguonnx.translate import load_translator

ROOT = Path(__file__).parent.parent
BUDGETS = [None, 2000, 1000, 500, 300]
# count_cached_as_free=False makes the counts independent of the local cache.
GALICIAN = ["Bo día", "Bo día, como estás?", "Bo día, o meu nome é Miro",
            "A lingua galega é unha lingua románica que se fala principalmente en Galicia."]


def routing():
    graph = load_translator().graph
    rows = []
    for mb in BUDGETS:
        kept = sum(c.within_size_cap(mb, count_cached_as_free=False) for c in graph.capabilities)
        langs = graph.languages_under(mb, count_cached_as_free=False, oversize_fallback=False)
        rows.append({"max_model_mb": mb, "models_kept": kept, "languages_routable": len(langs)})
    return {"selection_models": len(graph.capabilities), "budgets": rows}


def detection():
    from linguonnx.detect import LanguageDetector
    detector = LanguageDetector()
    rows = []
    for text in GALICIAN:
        probs = detector.detect_probs(text, top_k=1)
        tag, conf = next(iter(probs.items()))
        rows.append({"input": text, "length": len(text), "detected": tag, "confidence": round(conf, 2)})
    return {"model_id": detector.model_id, "galician_short_text": rows}


def main():
    out = ROOT / "benchmarks" / "results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    results["routing"] = routing()
    if "--detect" in sys.argv:
        results["detect"] = detection()
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sections out {len(results)} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
