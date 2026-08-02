#!/usr/bin/env python3
"""Translate a few sentences with the default translator.

Shows the plain call, the ``return_route=True`` form that tells you which
models ran, and the difference beam search makes against greedy decoding.

Every pair here resolves to a dedicated opus-mt model, so the downloads stay
small: ``opus-mt-pt-en-int8`` (172 MB), ``opus-mt-en-ca-int8`` (157 MB),
``opus-mt-en-gl-int8`` (157 MB), ``opus-mt-pt-gl-int8`` (84 MB). Pairs with no
dedicated model route through MADLAD instead, which is a 4.9 GB download — see
``inspect_routes.py`` to check a pair before you commit to it.

Run it::

    python examples/translate_basic.py
"""

from linguonnx import load_translator

PAIRS = [
    ("pt", "en", "Bom dia, como estás?"),
    ("pt", "gl", "Bom dia, como estás?"),
    ("en", "ca", "Good morning, how are you today?"),
    ("en", "gl", "Good morning, how are you today?"),
]


def main() -> None:
    tx = load_translator()
    print(f"{len(tx.models)} models in the graph, "
          f"{len(tx.available_languages)} languages reachable\n")

    for src, tgt, text in PAIRS:
        out, route = tx.translate(text, src=src, tgt=tgt, return_route=True)
        print(f"{src} -> {tgt}: {text}")
        print(f"           {out}")
        print(f"           via {', '.join(route.model_ids)} "
              f"({route.license_tier})\n")

    # Beam search is the default and costs about four times greedy decoding.
    # On short, easy sentences the two often agree; the gap opens up on longer
    # input, which is why the default is the slower one.
    text = "O tempo esteve bom durante toda a semana, mas amanhã vai chover."
    print("beam=4:", tx.translate(text, src="pt", tgt="en"))
    print("greedy:", tx.translate(text, src="pt", tgt="en", num_beams=1))


if __name__ == "__main__":
    main()
