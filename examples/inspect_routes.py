#!/usr/bin/env python3
"""Show the routes for a language pair, and how to pin the one you want.

Routing reads the registry and never loads a model, so this whole script runs
without downloading anything. That makes it the cheap way to find out what a
pair would cost before you commit to fetching several gigabytes.

It prints the ranked routes for a pair, the winner under each ``prefer``
policy, what a hop cap changes, and how to hand a chosen route back to
``translate()``.

Downloads: nothing.

Run it::

    python examples/inspect_routes.py            # pt -> ru
    python examples/inspect_routes.py pt eu
"""

import sys

from linguonnx import load_translator
from linguonnx.translate import NoRouteError


def show(tx, src: str, tgt: str) -> None:
    print(f"=== {src} -> {tgt} " + "=" * 40)

    print("\nranked routes:")
    routes = tx.routes(src, tgt)
    if not routes:
        print("  none")
        return
    for i, route in enumerate(routes[:5]):
        print(f"  [{i}] {route}")

    print("\nthe winner under each policy:")
    for policy in ("fewest_hops", "dedicated"):
        route = tx.route(src, tgt, prefer=policy)
        print(f"  {policy:<13} {route.n_hops} hop(s) via "
              f"{', '.join(route.model_ids)}"
              f"{'  pivots: ' + str(route.pivots) if route.pivots else ''}")

    print("\ndirect models only (max_hops=1):")
    try:
        print(f"  {tx.route(src, tgt, max_hops=1)}")
    except NoRouteError as err:
        print(f"  {err}")

    # Disagree with the ranking? Take a route from the list and hand it back.
    # translate(route=...) runs it verbatim, with no scoring at all.
    chosen = routes[-1]
    print(f"\nto force the lowest-ranked route above:")
    print(f"  route = tx.routes({src!r}, {tgt!r})[-1]   # {chosen}")
    print(f"  tx.translate(text, route=route)")
    print(f"  # would download: {', '.join(chosen.model_ids)}, "
          f"{chosen.total_size_mb} MB total\n")


def main() -> None:
    src, tgt = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("pt", "ru")
    tx = load_translator()
    print(f"pivot ranking in force: {tx.pivot_ranking}\n")
    show(tx, src, tgt)


if __name__ == "__main__":
    main()
