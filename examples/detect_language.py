#!/usr/bin/env python3
"""Identify the language of a handful of sentences, including Arabic dialects.

Shows the three detection calls side by side — ``detect_raw`` for the model's
own label, ``detect`` for a BCP-47 tag, ``detect_probs`` for the runners-up —
and what ``collapse_varieties`` changes. The Arabic lines are the interesting
part: GlotLID labels varieties, so colloquial Arabic comes back as a dialect
rather than as ``ar``.

Downloads: ``glotlid-int8``, 425 MB, on first run only.

Run it::

    python examples/detect_language.py
"""

from linguonnx import load_detector

SENTENCES = [
    ("Portuguese", "Bom dia, como estás? Hoje está um dia bonito."),
    ("Galician", "Bo día, como estás? Hoxe fai un día bonito."),
    ("Basque", "Egun on, zer moduz? Gaur egun ederra dago."),
    ("Catalan", "Bon dia a tothom, com aneu?"),
    ("Saudi Arabic", "وش لونك يا خوي، كيف الحال اليوم؟"),
    ("Levantine Arabic", "شو أخبارك؟ كيفك اليوم يا صاحبي؟"),
    ("Standard Arabic", "صباح الخير، كيف حالك اليوم؟"),
    ("Mandarin", "早上好，今天天气很好。"),
]


def main() -> None:
    det = load_detector()
    print(f"model: {det.model_id}  loss: {det.loss}  "
          f"languages: {len(det.available_languages)}\n")

    width = max(len(name) for name, _ in SENTENCES)
    for name, text in SENTENCES:
        label, score = det.detect_raw(text)
        tag = det.detect(text)
        collapsed = det.detect(text, collapse_varieties=True)
        runners = ", ".join(
            f"{k} {v:.3f}" for k, v in det.detect_probs(text, top_k=3).items())
        print(f"{name:<{width}}  raw={label:<10} {score:.4f}  "
              f"tag={tag:<10} collapsed={collapsed:<8} top3: {runners}")

    print("\nThe Arabic lines differ in `raw` and agree in `collapsed`: that is")
    print("dialect identification you get for free, folded away when you do not")
    print("want it.")


if __name__ == "__main__":
    main()
