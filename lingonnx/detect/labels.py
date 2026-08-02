"""GlotLID label <-> BCP-47 mapping.

GlotLID labels look like ``eng_Latn`` or ``__label__eng_Latn`` (fastText's
native prefix) - an ISO 639-3 code, an underscore, an ISO 15924 script code.
We convert those to BCP-47 tags:

- ISO 639-3 -> ISO 639-1 when a two-letter code exists (eng -> en).
- The script subtag is kept when it disambiguates something real-world
  speakers actually care about (Chinese Han script, Serbian Cyrillic-vs-Latin)
  and dropped when it is just the unsurprising default for that language
  (eng_Latn -> en, not en-Latn; ast_Latn -> ast, not ast-Latn).
- Languages with no ISO 639-1 code keep their ISO 639-3 code as the primary
  subtag (ast_Latn -> ast).

Determining "the unsurprising default script" is delegated to `langcodes`
(CLDR likely-subtags data via ``Language.maximize()``), which is precise for
the vast majority of the 2102 GlotLID labels. It is a language, not a script,
that decides whether a script is meaningful, though: langcodes' likely-subtag
data says Cyrl is Serbian's *statistically* most common script, and Hans is
Mandarin's - but both languages are routinely written in a second script that
readers need distinguished, so we never drop the script for them regardless
of what the likelihood tables say.
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    import langcodes
    _HAS_LANGCODES = True
except ImportError:  # pragma: no cover - langcodes is a hard dependency
    _HAS_LANGCODES = False

LABEL_PREFIX = "__label__"

# ISO 639-3 codes that have a 2-letter ISO 639-1 equivalent but langcodes
# does not resolve automatically, because they name an *individual* language
# under a macrolanguage rather than the macrolanguage itself (e.g. GlotLID's
# "arb" = Standard Arabic, a specific variety of the "ar" macrolanguage; GlotLID
# never emits a bare "ara" label). Mapping these to the macrolanguage's
# 2-letter code is what every downstream consumer (translators, TTS voice
# pickers, etc.) actually expects.
_MACRO_OVERRIDES: Dict[str, str] = {
    "arb": "ar",   # Standard Arabic -> Arabic macrolanguage
    "cmn": "zh",   # Mandarin -> Chinese macrolanguage
    "zho": "zh",   # Chinese (langcodes already does this, kept explicit)
}

# Languages where GlotLID's script subtag encodes information real users
# care about, even though CLDR likely-subtag data would call it "the
# default" and we'd otherwise drop it. Keyed by the *resolved* BCP-47 base
# tag (post 639-3 -> 639-1 mapping).
_ALWAYS_KEEP_SCRIPT = {"zh", "sr"}


def _base_tag(iso3: str) -> str:
    """ISO 639-3 -> best BCP-47 primary-language subtag."""
    if iso3 in _MACRO_OVERRIDES:
        return _MACRO_OVERRIDES[iso3]
    if _HAS_LANGCODES:
        try:
            lang = langcodes.Language.get(iso3)
            resolved = lang.language or iso3
            # langcodes accepts unknown-but-well-formed codes without error;
            # guard against it silently echoing back nonsense subtags.
            if lang.is_valid():
                return resolved
        except Exception:
            pass
    return iso3


def _default_script(base: str) -> Optional[str]:
    """The script langcodes' CLDR data considers unsurprising for `base`."""
    if not _HAS_LANGCODES:
        return None
    try:
        return langcodes.Language.get(base).maximize().script
    except Exception:
        return None


def parse_label(label: str) -> tuple[str, str]:
    """Split a raw GlotLID label into (iso3, iso15924_script).

    Accepts both ``eng_Latn`` and fastText-prefixed ``__label__eng_Latn``.
    Raises ValueError for anything that doesn't match the ``xxx_Yyyy`` shape.
    """
    raw = label[len(LABEL_PREFIX):] if label.startswith(LABEL_PREFIX) else label
    if "_" not in raw:
        raise ValueError(f"not a GlotLID label: {label!r}")
    iso3, script = raw.rsplit("_", 1)
    if not iso3 or not script:
        raise ValueError(f"not a GlotLID label: {label!r}")
    return iso3, script


def to_bcp47(glotlid_label: str) -> str:
    """``eng_Latn`` -> ``en``, ``zho_Hans`` -> ``zh-Hans``, ``ast_Latn`` -> ``ast``."""
    iso3, script = parse_label(glotlid_label)
    base = _base_tag(iso3)
    default_script = _default_script(base)
    if base in _ALWAYS_KEEP_SCRIPT or script != default_script:
        return f"{base}-{script}"
    return base


def build_label_maps(glotlid_labels: list[str]) -> tuple[Dict[str, str], Dict[str, str]]:
    """Build the (glotlid_label -> bcp47) and (bcp47 -> glotlid_label) maps.

    Labels are processed in file order, so when two GlotLID labels collapse
    to the same BCP-47 tag (shouldn't normally happen once scripts are kept
    where they matter) the *first* one in the label list wins the reverse
    mapping - deterministic, and matches "first/primary sense wins" used
    elsewhere in these BCP-47 conversions.
    """
    forward: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    for label in glotlid_labels:
        raw = label[len(LABEL_PREFIX):] if label.startswith(LABEL_PREFIX) else label
        bcp47 = to_bcp47(label)
        forward[raw] = bcp47
        reverse.setdefault(bcp47, raw)
    return forward, reverse


class LabelMapper:
    """Bidirectional GlotLID-label <-> BCP-47 lookup built from `labels.json`."""

    def __init__(self, glotlid_labels: list[str]):
        self._forward, self._reverse = build_label_maps(glotlid_labels)

    @property
    def available_languages(self) -> set:
        return set(self._reverse.keys())

    def to_bcp47(self, glotlid_label: str) -> str:
        raw = (glotlid_label[len(LABEL_PREFIX):]
               if glotlid_label.startswith(LABEL_PREFIX) else glotlid_label)
        if raw not in self._forward:
            raise KeyError(f"unknown GlotLID label: {glotlid_label!r}")
        return self._forward[raw]

    def to_glotlid(self, bcp47_tag: str) -> str:
        if bcp47_tag not in self._reverse:
            raise KeyError(f"no GlotLID label maps to BCP-47 tag: {bcp47_tag!r}")
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
    "nob": "no", "nno": "no",           # Bokmal / Nynorsk
    "plt": "mg", "gaz": "om", "npi": "ne", "pbt": "ps", "als": "sq",
    "ekk": "et", "knc": "kr", "kmr": "ku", "ckb": "ku",
}


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
    if rest and macro in _ALWAYS_KEEP_SCRIPT:
        return f"{macro}-{rest}"
    return macro
