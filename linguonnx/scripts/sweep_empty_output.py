#!/usr/bin/env python3
"""linguonnx#42 registry-wide sweep.

mt-hitz-gl-eu(-int8) returned "" for every input because its decoder ranked
its own decoder_start/pad token above every real word at generation step 1,
and nothing enforced the `bad_words_ids` ban its own `generation_config.json`
already named. That was found only because a human traced one route by hand.
This script runs one short, real translation through every registered
translation model and flags any of:

  * empty output
  * whitespace/format-character-only output (see `linguonnx.limits.has_visible_content`)
  * output identical to the input (the model silently passed text through)

for real input text, prints a verdict per model, and exits non-zero if any
model is flagged.

Usage:
    python scripts/sweep_empty_output.py                  # every registered model
    python scripts/sweep_empty_output.py --cached-only     # skip cold downloads
    python scripts/sweep_empty_output.py --models mt-hitz-gl-eu-int8,opus-mt-en-pt-int8
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout("model took too long (see --per-model-timeout)")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from linguonnx.limits import has_visible_content
from linguonnx.model_manager import MODELS_DIR, list_models
from linguonnx.translate.decode import GenerationConfig
from linguonnx.translate.models import TranslationModel

# One short, real sentence per source language actually in the registry.
# A source language with no entry here has NO fallback: check_model() marks it
# "skipped-no-sample" and does not run the model. Substituting text from a
# different language would make the verdict meaningless in both directions -
# see linguonnx#42 and the retracted rows from the ad-hoc harness that did
# exactly that (32/183 int8 models affected, 0/7 recorded rows survived
# unchanged on re-check). Never add a language here unless the sentence and
# its script are independently confirmed correct.
SAMPLES = {
    "en": "The library opens at nine in the morning.",
    "pt": "Bom dia, o meu nome e o Joao e moro em Lisboa.",
    "gl": "Bos días, o meu nome e Joao e moro en Lisboa.",
    "ca": "Bon dia, em dic Joao i visc a Lisboa.",
    "eu": "Egun on, nire izena Joao da eta Lisboan bizi naiz.",
    "es": "Buenos días, me llamo Joao y vivo en Lisboa.",
    "fr": "Bonjour, je m'appelle Joao et j'habite a Lisbonne.",
    "de": "Guten Morgen, ich heisse Joao und wohne in Lissabon.",
    "it": "Buongiorno, mi chiamo Joao e vivo a Lisbona.",
    "ar": "صباح الخير، اسمي جواو وأعيش في لشبونة.",
    "asm_Beng": "শুভ ৰাতিপুৱা, মোৰ নাম জোৱাও আৰু মই লিছবনত থাকো।",
    "nl": "Goedemorgen, mijn naam is Joao en ik woon in Lissabon.",
    "sv": "God morgon, mitt namn ar Joao och jag bor i Lissabon.",
    "da": "Godmorgen, mit navn er Joao og jeg bor i Lissabon.",
    "no": "God morgen, mitt navn er Joao og jeg bor i Lisboa.",
    "fi": "Huomenta, nimeni on Joao ja asun Lissabonissa.",
    "pl": "Dzien dobry, nazywam sie Joao i mieszkam w Lizbonie.",
    "cs": "Dobre rano, jmenuji se Joao a bydlim v Lisabonu.",
    "sk": "Dobre rano, volam sa Joao a byvam v Lisabone.",
    "hu": "Jo reggelt, a nevem Joao, es Lisszabonban elek.",
    "ro": "Buna dimineata, numele meu este Joao si locuiesc la Lisabona.",
    "bg": "Добро утро, казвам се Жоао и живея в Лисабон.",
    "uk": "Доброго ранку, мене звати Жоао, i я живу в Лiсабонi.",
    "ru": "Доброе утро, меня зовут Жоао, и я живу в Лиссабоне.",
    "el": "Καλημέρα, με λένε Ζοάο και ζω στη Λισαβόνα.",
    "tr": "Gunaydin, benim adim Joao ve Lizbon'da yasiyorum.",
    "he": "בוקר טוב, שמי ז'ואאו ואני גר בליסבון.",
    "fa": "صبح بخیر، اسم من ژوآئو است و در لیسبون زندگی می‌کنم.",
    "hi": "सुप्रभात, मेरा नाम जोआओ है और मैं लिस्बन में रहता हूँ।",
    "bn": "শুভ সকাল, আমার নাম জোয়াও এবং আমি লিসবনে থাকি।",
    "vi": "Chao buoi sang, ten toi la Joao va toi song o Lisbon.",
    "th": "สวัสดีตอนเช้า ฉันชื่อโจอาว และฉันอาศัยอยู่ที่ลิสบอน",
    "id": "Selamat pagi, nama saya Joao dan saya tinggal di Lisbon.",
    "ms": "Selamat pagi, nama saya Joao dan saya tinggal di Lisbon.",
    "sw": "Habari za asubuhi, jina langu ni Joao na ninaishi Lisbon.",
    "ja": "おはようございます、私の名前はジョアンで、リスボンに住んでいます。",
    "ko": "좋은 아침입니다, 제 이름은 조앙이고 리스본에 살고 있습니다.",
    "zh": "早上好，我叫若昂，我住在里斯本。",
}


def _is_cached(model_id: str, entry: dict) -> bool:
    directory = MODELS_DIR / model_id
    graphs = entry.get("graphs", {})
    if not graphs:
        return False
    return all((directory / name).exists() for name in graphs.values())


def _sample_for(src: str) -> str | None:
    """The confirmed real-language sample for `src`, or None if there isn't one.

    Never substitutes text from another language: a missing sample must be a
    loud skip, not a silent, meaningless verdict.
    """
    return SAMPLES.get(src)


def check_model(model_id: str, entry: dict) -> dict:
    """One verdict dict: status in
    {"ok", "empty", "verbatim", "error", "skipped-no-sample"}."""
    pair = entry.get("pair")
    if pair:
        src, tgt = pair
    else:
        src_langs = entry.get("src_languages") or entry.get("languages") or ["en"]
        tgt_langs = entry.get("tgt_languages") or entry.get("languages") or ["pt"]
        src = "en" if "en" in src_langs else next(iter(src_langs))
        candidates = [c for c in tgt_langs if c != src]
        tgt = candidates[0] if candidates else next(iter(tgt_langs))
    text = _sample_for(src)
    if text is None:
        return {"model_id": model_id, "status": "skipped-no-sample", "src": src, "tgt": tgt,
                "detail": f"no confirmed sample sentence for source language {src!r}",
                "seconds": 0.0}
    started = time.monotonic()
    try:
        model = TranslationModel(model_id, entry)
        out = model.translate(text, src, tgt,
                              config=GenerationConfig(num_beams=1, max_new_tokens=64))
    except Exception as exc:  # noqa: BLE001 - a sweep must not die on one model
        return {"model_id": model_id, "status": "error", "src": src, "tgt": tgt,
                "detail": f"{type(exc).__name__}: {exc}",
                "seconds": time.monotonic() - started}
    seconds = time.monotonic() - started
    if not has_visible_content(out):
        return {"model_id": model_id, "status": "empty", "src": src, "tgt": tgt,
                "detail": repr(out), "seconds": seconds}
    if out.strip() == text.strip():
        return {"model_id": model_id, "status": "verbatim", "src": src, "tgt": tgt,
                "detail": repr(out), "seconds": seconds}
    return {"model_id": model_id, "status": "ok", "src": src, "tgt": tgt,
            "detail": repr(out)[:80], "seconds": seconds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="comma-separated model ids; default is every registered model")
    parser.add_argument("--cached-only", action="store_true",
                        help="skip models whose graphs are not already on disk")
    parser.add_argument("--per-model-timeout", type=int, default=180,
                        help="seconds before a single model is abandoned as hung (default 180)")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, _alarm)

    registry = list_models(kind="translate")
    if args.models:
        wanted = args.models.split(",")
        registry = {m: registry[m] for m in wanted}

    results = []
    for i, (model_id, entry) in enumerate(sorted(registry.items()), 1):
        if args.cached_only and not _is_cached(model_id, entry):
            print(f"[{i}/{len(registry)}] {model_id}: SKIP (not cached)")
            continue
        print(f"[{i}/{len(registry)}] {model_id}: running...", end=" ", flush=True)
        signal.alarm(args.per_model_timeout)
        try:
            result = check_model(model_id, entry)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # a corrupt graph must not kill the sweep
            result = {"model_id": model_id, "status": "error", "src": "?", "tgt": "?",
                      "detail": f"{type(exc).__name__}: {exc}", "seconds": 0.0}
        finally:
            signal.alarm(0)
        results.append(result)
        print(f"{result['status']} ({result['seconds']:.1f}s) {result['detail']}")

    skipped = [r for r in results if r["status"] == "skipped-no-sample"]
    flagged = [r for r in results if r["status"] not in ("ok", "skipped-no-sample")]
    run = [r for r in results if r["status"] != "skipped-no-sample"]
    print("\n" + "=" * 72)
    print(f"{len(results)} models checked, {len(run)} run, "
          f"{len(skipped)} skipped for lack of a source sample, {len(flagged)} flagged")
    if skipped:
        print(f"  {len(skipped)} models skipped for lack of a source sample "
              "(no verdict - add a confirmed sample to SAMPLES to cover them):")
        for r in skipped:
            print(f"    {r['model_id']:35s} src={r['src']}")
    for r in flagged:
        print(f"  {r['model_id']:35s} {r['status']:9s} {r['src']}->{r['tgt']:5s} {r['detail']}")
    print("=" * 72)
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
