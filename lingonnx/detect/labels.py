"""fastText LID label <-> BCP-47 mapping.

The models lingonnx ships use two label shapes:

- ``eng_Latn`` / ``__label__eng_Latn`` - an ISO 639-3 code, an underscore, an
  ISO 15924 script code. GlotLID, OpenLID and OpenLID-v2 use this shape.
- ``en`` / ``__label__pt`` - a bare ISO 639-1 or 639-3 code with no script.
  fastText's classic ``lid.176`` uses this shape.

Both are converted to BCP-47 tags by :func:`to_bcp47`.

The conversion is `langcodes.standardize_tag` doing the work: it maps ISO
639-3 to ISO 639-1 where a two-letter code exists (``eng`` -> ``en``), repairs
deprecated codes (``iw`` -> ``he``, ``in`` -> ``id``), applies canonical
replacements (``tgl`` -> ``fil``), normalises case and region form
(``pt-br`` -> ``pt-BR``), and drops a script subtag that CLDR's likely-subtags
data says is redundant (``eng-Latn`` -> ``en``) while keeping one that is not
(``zho-Hans`` -> ``zh-Hans``, ``srp-Cyrl`` -> ``sr-Cyrl``).

Two things `standardize_tag` does not do, which this module owns:

1. **Redundant scripts CLDR has no data to drop.** ``standardize_tag`` only
   drops a script subtag when CLDR's likely-subtags table covers that
   language, and that table covers a minority of GlotLID's 2102 labels. It
   returns ``ast-Latn``, ``yo-Latn``, ``ig-Latn`` and ``sn-Latn`` - correct
   but noisy, since none of those languages is written in a second script
   that anyone needs distinguished. So we drop the script when
   ``Language.maximize()`` guesses the same script anyway, which for a
   language CLDR knows nothing about is a plain "Latn unless told
   otherwise". ``_ALWAYS_KEEP_SCRIPT`` holds the languages this must not
   happen to: it is a *language*, not a script, that decides whether a
   script subtag is informative, and CLDR likelihood tables call Cyrl
   Serbian's most likely script and Hans Mandarin's even though readers of
   both need the two scripts kept apart.
2. **Macrolanguage varieties.** ``standardize_tag`` passes ``arb``, ``ars``,
   ``cmn``, ``yue``, ``pes``, ``swh`` and friends through unchanged. See
   ``_MACRO_OVERRIDES`` (always applied - GlotLID never emits a bare ``ara``)
   and :func:`collapse_variety` (opt-in) below.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import langcodes

LOG = logging.getLogger(__name__)

LABEL_PREFIX = "__label__"

# ISO 639-3 codes that name an *individual* language under a macrolanguage
# whose two-letter code is what every downstream consumer (translators, TTS
# voice pickers, ...) actually expects. GlotLID/OpenLID label Standard Arabic
# as "arb" and never emit a bare "ara", so "arb_Arab" must resolve to "ar".
_MACRO_OVERRIDES: Dict[str, str] = {
    "arb": "ar",   # Standard Arabic -> Arabic macrolanguage
    "cmn": "zh",   # Mandarin -> Chinese macrolanguage
    "zho": "zh",   # Chinese (langcodes already does this, kept explicit)
}

# Languages routinely written in more than one script, where the script
# subtag carries information even though CLDR's likely-subtag data would call
# it "the default" and we would otherwise drop it. Keyed by the *resolved*
# BCP-47 base tag.
_ALWAYS_KEEP_SCRIPT = {"zh", "sr"}

_WARNED_UNKNOWN: set = set()


def split_label(label: str) -> Tuple[str, Optional[str]]:
    """Split a raw fastText LID label into (code, script or None).

    Accepts ``eng_Latn``, ``__label__eng_Latn``, ``pt`` and ``__label__pt``.
    Raises ValueError for anything that is neither shape.
    """
    raw = label[len(LABEL_PREFIX):] if label.startswith(LABEL_PREFIX) else label
    if "_" not in raw:
        # bare lid.176-style code: 2 or 3 letters, nothing else
        if raw.isalpha() and 2 <= len(raw) <= 3:
            return raw, None
        raise ValueError(f"not a fastText LID label: {label!r}")
    code, script = raw.rsplit("_", 1)
    if not code or not script:
        raise ValueError(f"not a fastText LID label: {label!r}")
    return code, script


def parse_label(label: str) -> Tuple[str, str]:
    """Split a ``xxx_Yyyy`` label into (iso3, iso15924_script).

    Raises ValueError for a label with no script subtag; use
    :func:`split_label` when either shape is acceptable.
    """
    code, script = split_label(label)
    if script is None:
        raise ValueError(f"not a language_Script label: {label!r}")
    return code, script


def _drop_redundant_script(tag: str) -> str:
    """Drop a script subtag that ``standardize_tag`` kept for lack of data."""
    base, _, script = tag.partition("-")
    if not script or base in _ALWAYS_KEEP_SCRIPT:
        return tag
    try:
        if langcodes.Language.get(base).maximize().script == script:
            return base
    except Exception:
        pass
    return tag


def to_bcp47(label: str) -> str:
    """``eng_Latn`` -> ``en``, ``zho_Hans`` -> ``zh-Hans``, ``pt`` -> ``pt``."""
    iso3, script = split_label(label)
    code = _MACRO_OVERRIDES.get(iso3, iso3)
    tag = f"{code}-{script}" if script else code
    try:
        if not langcodes.Language.get(tag).is_valid() and tag not in _WARNED_UNKNOWN:
            _WARNED_UNKNOWN.add(tag)
            LOG.warning("label %r is not a valid language tag; "
                        "emitting it as-is", label)
        standardized = langcodes.standardize_tag(tag)
    except Exception:
        return tag
    return _drop_redundant_script(standardized)


def build_label_maps(labels: list[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build the (raw label -> bcp47) and (bcp47 -> raw label) maps.

    Labels are processed in file order, so when two labels collapse to the
    same BCP-47 tag the *first* one in the label list wins the reverse
    mapping - deterministic, and matches "first/primary sense wins" used
    elsewhere in these BCP-47 conversions.
    """
    forward: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    for label in labels:
        raw = label[len(LABEL_PREFIX):] if label.startswith(LABEL_PREFIX) else label
        bcp47 = to_bcp47(label)
        forward[raw] = bcp47
        reverse.setdefault(bcp47, raw)
    return forward, reverse


class LabelMapper:
    """Bidirectional LID-label <-> BCP-47 lookup built from `labels.json`."""

    def __init__(self, labels: list[str]):
        self._forward, self._reverse = build_label_maps(labels)

    @property
    def available_languages(self) -> set:
        return set(self._reverse.keys())

    def to_bcp47(self, label: str) -> str:
        raw = (label[len(LABEL_PREFIX):]
               if label.startswith(LABEL_PREFIX) else label)
        if raw not in self._forward:
            raise KeyError(f"unknown label: {label!r}")
        return self._forward[raw]

    def to_glotlid(self, bcp47_tag: str) -> str:
        if bcp47_tag not in self._reverse:
            raise KeyError(f"no label maps to BCP-47 tag: {bcp47_tag!r}")
        return self._reverse[bcp47_tag]


# ---------------------------------------------------------------------------
# Macrolanguage varieties
#
# GlotLID labels individual varieties, not macrolanguages: casual Arabic input
# comes back as ajp_Arab (South Levantine) or ars_Arab (Najdi/Saudi) rather
# than arb_Arab, and Chinese may come back as yue_Hani (Cantonese). That
# fidelity is the point of the model and is worth keeping - it is free
# text-side dialect identification. But most consumers (a TTS voice picker, a
# translation target, an OVOS session lang) want the macrolanguage tag they
# know how to act on, and would treat "ajp-Arab" as an unsupported language.
#
# So both are available: detect() can collapse, detect_raw() never does.
# ---------------------------------------------------------------------------

VARIETY_TO_MACRO: Dict[str, str] = {
    # Arabic
    "arb": "ar", "ars": "ar", "apc": "ar", "ajp": "ar", "aeb": "ar",
    "ary": "ar", "arz": "ar", "acm": "ar", "afb": "ar", "arq": "ar",
    "shu": "ar", "apd": "ar", "ayl": "ar",
    # Chinese
    "cmn": "zh", "yue": "zh", "wuu": "zh", "hak": "zh", "nan": "zh",
    "gan": "zh", "cjy": "zh", "cdo": "zh", "hsn": "zh", "lzh": "zh",
    # other macrolanguages GlotLID splits
    "pes": "fa", "prs": "fa",           # Iranian / Dari Persian
    "azj": "az", "azb": "az",           # North / South Azerbaijani
    "uzn": "uz", "uzs": "uz",           # Northern / Southern Uzbek
    "khk": "mn",                        # Halh Mongolian
    "zsm": "ms", "lvs": "lv", "ydd": "yi", "swh": "sw",
    "nob": "no", "nno": "no",           # Bokmal / Nynorsk (ISO 639-3 form)
    "nb": "no", "nn": "no",             # ... and their standardized BCP-47 form
    "plt": "mg", "gaz": "om", "npi": "ne", "pbt": "ps", "als": "sq",
    "ekk": "et", "knc": "kr", "kmr": "ku", "ckb": "ku",
}

# Languages routinely written in more than one script, where the script
# subtag still disambiguates something readers care about after a variety has
# been collapsed onto its macrolanguage (yue-Hani -> zh-Hani, not zh).
_MULTISCRIPT_MACROS = {"zh", "sr"}


def collapse_variety(tag: str) -> str:
    """
    Collapse a variety tag onto its macrolanguage, keeping any script subtag.

    ``ajp-Arab`` -> ``ar``, ``yue-Hani`` -> ``zh-Hani``, ``pt`` -> ``pt``.
    Tags that are not known varieties are returned unchanged.
    """
    if not tag:
        return tag
    primary, _, rest = tag.partition("-")
    macro = VARIETY_TO_MACRO.get(primary)
    if not macro:
        return tag
    # a script subtag is only worth keeping if it still disambiguates
    if rest and macro in _MULTISCRIPT_MACROS:
        return f"{macro}-{rest}"
    return macro
