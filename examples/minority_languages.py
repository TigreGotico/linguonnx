#!/usr/bin/env python3
"""Reach the small languages: Mirandese, Aragonese, Galician, Basque, Kabuverdianu.

The point of a routing graph is that the long tail stays reachable. Galician
and Basque have dedicated bilingual models. Mirandese and Aragonese exist only
inside MADLAD. Kabuverdianu exists only inside NLLB, which is non-commercial
and therefore out of the default graph — so this script also shows what
``include_noncommercial=True`` unlocks, and what the error says before you
pass it.

By default the script only **inspects** routes, which downloads nothing.

Pass ``--translate`` to actually run them. That is a large download: Mirandese
and Aragonese both go through ``madlad400-3b-mt-int8`` (4.9 GB) and
Kabuverdianu through ``nllb-600M-int8`` (1.9 GB). Galician and Basque stay
small.

Run it::

    python examples/minority_languages.py
    python examples/minority_languages.py --translate
"""

import sys

from linguonnx import load_translator
from linguonnx.translate import NoRouteError

TEXT = "Bom dia, o tempo hoje está muito bonito."

TARGETS = [
    ("mwl", "Mirandese"),
    ("an", "Aragonese"),
    ("gl", "Galician"),
    ("eu", "Basque"),
    ("kea", "Kabuverdianu"),
]


def main() -> None:
    run = "--translate" in sys.argv

    tx = load_translator()
    print(f"default graph: {len(tx.available_languages)} languages, "
          f"permissive licences only\n")

    blocked = []
    for tag, name in TARGETS:
        try:
            route = tx.route("pt", tag)
        except NoRouteError as err:
            blocked.append((tag, name))
            print(f"{name:<14} ({tag:<3})  no route")
            print(f"{'':<14}  {err}\n")
            continue
        print(f"{name:<14} ({tag:<3})  {route.n_hops} hop(s) via "
              f"{', '.join(route.model_ids)}  "
              f"[{route.total_size_mb} MB, {route.license_tier}]")
        if run:
            print(f"{'':<14}  {tx.translate(TEXT, src='pt', tgt=tag)}")

    if not blocked:
        return

    print("\nWith include_noncommercial=True the excluded models join the graph:")
    nc = load_translator(include_noncommercial=True)
    print(f"  {len(nc.available_languages)} languages reachable\n")
    for tag, name in blocked:
        route = nc.route("pt", tag)
        print(f"{name:<14} ({tag:<3})  {route.n_hops} hop(s) via "
              f"{', '.join(route.model_ids)}  "
              f"[{route.total_size_mb} MB, {route.license_tier}]")
        if run:
            print(f"{'':<14}  {nc.translate(TEXT, src='pt', tgt=tag)}")

    print("\nThat output is CC-BY-NC-4.0. Route.license_tier says so on every")
    print("route, and it reports the worst hop on a chain, not the best.")


if __name__ == "__main__":
    main()
