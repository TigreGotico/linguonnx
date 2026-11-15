"""IndicTrans2 text processing, vendored from AI4Bharat's ``IndicTransToolkit``.

Why vendored
------------
IndicTrans2 does not translate raw text. It expects the text AI4Bharat's
``IndicProcessor`` produces: punctuation normalised, native digits folded to
ASCII, URLs/emails/numerals swapped for ``<ID1>`` placeholders, Moses- or
Indic-tokenised, and - for every Indic script that is not Perso-Arabic, Ol
Chiki, Meetei Mayek or Latin - **transliterated into Devanagari**, because the
model's shared vocabulary is Devanagari. The reverse of all of that has to run
on the output, or a Tamil request comes back in fluent Devanagari with nothing
raised.

The obvious move is to depend on ``IndicTransToolkit``, which is where this code
comes from. linguonnx does not, for one reason: that package declares
``transformers`` as a hard dependency. linguonnx deliberately has no
``transformers`` dependency and never passes ``trust_remote_code``, so that the
only things it deserialises from a downloaded model are JSON and SentencePiece.
Pulling ``transformers`` in through an extra would widen that surface for code
this module never calls.

So the processing logic is copied here rather than reimplemented - a
reimplementation would risk diverging from the training-time preprocessing in
ways that produce fluent, wrong output. The only changes are mechanical:
Cython ``cdef``/``cpdef`` declarations dropped, the ``tqdm`` progress bar
removed, and the placeholder map returned to the caller instead of being parked
in a module-level ``Queue``. Behaviour is unchanged, and
``test/test_translate_preprocess.py`` checks it against the real
``IndicProcessor`` when that package happens to be installed.

Source: https://github.com/VarunGumma/IndicTransToolkit -
``IndicTransToolkit/processor.pyx``, MIT licence, Copyright (c) Varun Gumma.
The upstream MIT notice is reproduced in ``docs/licences.md``.

Runtime dependencies (``pip install linguonnx[indic]``): ``regex``,
``sacremoses``, ``indic-nlp-library-itt``. All pure Python; none of them loads
downloaded code.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

__all__ = ["IndicProcessor", "LANGUAGE_TAGS", "FLORES_TO_ISO"]

#: Every language tag the IndicTrans2 vocabularies carry, from AI4Bharat's
#: ``tokenization_indictrans.py``. A tag outside this set is a caller error,
#: not an unknown word: the model would treat it as ordinary text.
LANGUAGE_TAGS = frozenset({
    "asm_Beng", "awa_Deva", "ben_Beng", "bho_Deva", "brx_Deva", "doi_Deva",
    "eng_Latn", "gom_Deva", "gon_Deva", "guj_Gujr", "hin_Deva", "hne_Deva",
    "kan_Knda", "kas_Arab", "kas_Deva", "kha_Latn", "lus_Latn", "mag_Deva",
    "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva",
    "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva",
    "tam_Taml", "tel_Telu", "urd_Arab", "unr_Deva",
})

#: FLORES tag -> the ISO code ``indic_nlp_library`` normalises/tokenises with.
#: Not a language mapping: several languages deliberately borrow a bigger
#: relative's rules (``brx_Deva`` -> ``hi``), which is what IndicTrans2 was
#: trained with.
FLORES_TO_ISO = {
    "asm_Beng": "as", "awa_Deva": "hi", "ben_Beng": "bn", "bho_Deva": "hi",
    "brx_Deva": "hi", "doi_Deva": "hi", "eng_Latn": "en", "gom_Deva": "kK",
    "gon_Deva": "hi", "guj_Gujr": "gu", "hin_Deva": "hi", "hne_Deva": "hi",
    "kan_Knda": "kn", "kas_Arab": "ur", "kas_Deva": "hi", "kha_Latn": "en",
    "lus_Latn": "en", "mag_Deva": "hi", "mai_Deva": "hi", "mal_Mlym": "ml",
    "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne",
    "ory_Orya": "or", "pan_Guru": "pa", "san_Deva": "hi", "sat_Olck": "or",
    "snd_Arab": "ur", "snd_Deva": "hi", "tam_Taml": "ta", "tel_Telu": "te",
    "urd_Arab": "ur", "unr_Deva": "hi",
}

#: Scripts that are *not* transliterated to Devanagari on the way in. The model
#: keeps Perso-Arabic, Ol Chiki, Meetei Mayek and Latin as they are.
NON_TRANSLITERATED_SCRIPTS = frozenset({"Arab", "Aran", "Olck", "Mtei", "Latn"})

_EXTRA = ("linguonnx[indic]")


def _require(module: str):
    """Import an optional dependency, or say exactly what to install.

    Never falls back to an approximation: a wrong tokenisation produces fluent
    output in the wrong words, which is the failure this whole module exists to
    prevent.
    """
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"IndicTrans2 preprocessing needs {module!r}, which is not "
            f"installed. Install the extra: pip install '{_EXTRA}'") from exc


# Devanagari, Bengali, Gujarati, Kannada, Telugu, Oriya, Gurmukhi, Malayalam,
# Meetei Mayek, Ol Chiki, Perso-Arabic and Tamil digits -> ASCII.
_DIGITS = {
    "\u09e6": "0", "\u0ae6": "0", "\u0ce6": "0", "\u0966": "0", "\u0660": "0",
    "\uabf0": "0", "\u0b66": "0", "\u0a66": "0", "\u1c50": "0", "\u06f0": "0",
    "\u09e7": "1", "\u0ae7": "1", "\u0967": "1", "\u0ce7": "1", "\u06f1": "1",
    "\uabf1": "1", "\u0b67": "1", "\u0a67": "1", "\u1c51": "1", "\u0c67": "1",
    "\u09e8": "2", "\u0ae8": "2", "\u0968": "2", "\u0ce8": "2", "\u06f2": "2",
    "\uabf2": "2", "\u0b68": "2", "\u0a68": "2", "\u1c52": "2", "\u0c68": "2",
    "\u09e9": "3", "\u0ae9": "3", "\u0969": "3", "\u0ce9": "3", "\u06f3": "3",
    "\uabf3": "3", "\u0b69": "3", "\u0a69": "3", "\u1c53": "3", "\u0c69": "3",
    "\u09ea": "4", "\u0aea": "4", "\u096a": "4", "\u0cea": "4", "\u06f4": "4",
    "\uabf4": "4", "\u0b6a": "4", "\u0a6a": "4", "\u1c54": "4", "\u0c6a": "4",
    "\u09eb": "5", "\u0aeb": "5", "\u096b": "5", "\u0ceb": "5", "\u06f5": "5",
    "\uabf5": "5", "\u0b6b": "5", "\u0a6b": "5", "\u1c55": "5", "\u0c6b": "5",
    "\u09ec": "6", "\u0aec": "6", "\u096c": "6", "\u0cec": "6", "\u06f6": "6",
    "\uabf6": "6", "\u0b6c": "6", "\u0a6c": "6", "\u1c56": "6", "\u0c6c": "6",
    "\u09ed": "7", "\u0aed": "7", "\u096d": "7", "\u0ced": "7", "\u06f7": "7",
    "\uabf7": "7", "\u0b6d": "7", "\u0a6d": "7", "\u1c57": "7", "\u0c6d": "7",
    "\u09ee": "8", "\u0aee": "8", "\u096e": "8", "\u0cee": "8", "\u06f8": "8",
    "\uabf8": "8", "\u0b6e": "8", "\u0a6e": "8", "\u1c58": "8", "\u0c6e": "8",
    "\u09ef": "9", "\u0aef": "9", "\u096f": "9", "\u0cef": "9", "\u06f9": "9",
    "\uabf9": "9", "\u0b6f": "9", "\u0a6f": "9", "\u1c59": "9", "\u0c6f": "9",
}

#: Ways the model has been seen to mangle the literal string "ID" inside a
#: placeholder, per language. Kept verbatim from upstream, including the one
#: missing comma on the tenth entry (which upstream's Python concatenates into
#: a single string); reproducing it keeps the maps byte-identical.
_INDIC_FAILURE_CASES = [
    "آی ڈی ",
    "ꯑꯥꯏꯗꯤ",
    "आईडी",
    "आई . डी . ",
    "आई . डी .",
    "आई. डी. ",
    "आई. डी.",
    "आय. डी. ",
    "आय. डी.",
    "आय . डी . ",
    "आय . डी ."
    "आइ . डी . ",
    "आइ . डी .",
    "आइ. डी. ",
    "आइ. डी.",
    "ऐटि",
    "آئی ڈی ",
    "ᱟᱭᱰᱤ ᱾",
    "आयडी",
    "ऐडि",
    "आइडि",
    "ᱟᱭᱰᱤ",
]


class IndicProcessor:
    """AI4Bharat's ``IndicProcessor``, minus the Cython and the queue.

    One instance is reusable and holds the Moses tools and the transliterator,
    which are not cheap to build. It is **not** thread-safe, for the same
    reason ``sacremoses`` is not.

    Unlike upstream, :meth:`preprocess` hands the placeholder map back to the
    caller instead of pushing it onto an instance queue, so a preprocess call
    and its matching postprocess call cannot drift apart.
    """

    def __init__(self, inference: bool = True):
        self.inference = inference

        regex = _require("regex")
        sacremoses = _require("sacremoses")
        _require("indicnlp")
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.transliterate.unicode_transliterate import \
            UnicodeIndicTransliterator

        self._re = regex
        self._normalizer_factory = IndicNormalizerFactory()
        self._normalizers: Dict[str, object] = {}
        self._en_tok = sacremoses.MosesTokenizer(lang="en")
        self._en_normalizer = sacremoses.MosesPunctNormalizer()
        self._en_detok = sacremoses.MosesDetokenizer(lang="en")
        self._xliterator = UnicodeIndicTransliterator()

        self._digits_table = {ord(k): v for k, v in _DIGITS.items()}
        for code in range(ord("0"), ord("9") + 1):
            self._digits_table[code] = chr(code)

        self._MULTISPACE_REGEX = regex.compile(r"[ ]{2,}")
        self._DIGIT_SPACE_PERCENT = regex.compile(r"(\d) %")
        self._DOUBLE_QUOT_PUNC = regex.compile(r"\"([,\.]+)")
        self._DIGIT_NBSP_DIGIT = regex.compile(r"(\d) (\d)")
        self._END_BRACKET_SPACE_PUNC_REGEX = regex.compile(r"\) ([\.!:?;,])")
        self._WHITESPACE = regex.compile(r"\s+")

        self._URL_PATTERN = regex.compile(
            r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))"
            r"(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b")
        self._NUMERAL_PATTERN = regex.compile(
            r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|"
            r"\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)")
        self._EMAIL_PATTERN = regex.compile(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
        self._OTHER_PATTERN = regex.compile(r"[A-Za-z0-9]*[#|@]\w+")

        self._PUNC_REPLACEMENTS = [
            (regex.compile(r"\r"), ""),
            (regex.compile(r"\(\s*"), "("),
            (regex.compile(r"\s*\)"), ")"),
            (regex.compile(r"\s:\s?"), ":"),
            (regex.compile(r"\s;\s?"), ";"),
            (regex.compile(r"[`´‘‚’]"), "'"),
            (regex.compile(r"[„“”«»]"), '"'),
            (regex.compile(r"[–—]"), "-"),
            (regex.compile(r"\.\.\."), "..."),
            (regex.compile(r" %"), "%"),
            (regex.compile(r"nº "), "nº "),
            (regex.compile(r" ºC"), " ºC"),
            (regex.compile(r" [?!;]"), lambda m: m.group(0).strip()),
            (regex.compile(r", "), ", "),
        ]

    # -- internals --------------------------------------------------------

    def _normalizer_for(self, iso_lang: str):
        if iso_lang not in self._normalizers:
            self._normalizers[iso_lang] = \
                self._normalizer_factory.get_normalizer(iso_lang)
        return self._normalizers[iso_lang]

    def _punc_norm(self, text: str) -> str:
        for pattern, replacement in self._PUNC_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        text = self._MULTISPACE_REGEX.sub(" ", text)
        text = self._END_BRACKET_SPACE_PUNC_REGEX.sub(r")\1", text)
        text = self._DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = self._DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = self._DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    def _wrap_with_placeholders(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace URLs, emails, numerals and handles with ``<IDn>`` markers.

        The model is not asked to translate a URL; it is asked to move a short
        opaque token, which it does far more reliably. The returned map lists
        every spelling of the marker the model has been observed to emit,
        including transliterated ones, so postprocessing can put the original
        back.
        """
        serial_no = 1
        entity_map: Dict[str, str] = {}
        patterns = [self._EMAIL_PATTERN, self._URL_PATTERN,
                    self._NUMERAL_PATTERN, self._OTHER_PATTERN]

        for pattern in patterns:
            for match in set(pattern.findall(text)):
                if pattern is self._URL_PATTERN:
                    if len(match.replace(".", "")) < 4:
                        continue
                if pattern is self._NUMERAL_PATTERN:
                    stripped = match.replace(" ", "").replace(".", "").replace(":", "")
                    if len(stripped) < 4:
                        continue

                base = f"<ID{serial_no}>"
                for template in ("<ID{n}>", "< ID{n} >", "[ID{n}]", "[ ID{n} ]",
                                 "[ID {n}]", "<ID{n}]", "< ID{n}]", "<ID{n} ]",
                                 "<id{n}>", "< id{n} >", "[id{n}]", "[ id{n} ]",
                                 "[id {n}]", "<id{n}]", "< id{n}]", "<id{n} ]"):
                    entity_map[template.format(n=serial_no)] = match
                for case in _INDIC_FAILURE_CASES:
                    for template in ("<{c}{n}>", "< {c}{n} >", "< {c} {n} >",
                                     "<{c} {n}]", "< {c} {n} ]", "[{c}{n}]",
                                     "[{c} {n}]", "[ {c}{n} ]", "[ {c} {n} ]",
                                     "{c} {n}", "{c}{n}"):
                        entity_map[template.format(c=case, n=serial_no)] = match

                text = text.replace(match, base)
                serial_no += 1

        text = self._WHITESPACE.sub(" ", text).replace(">/", ">").replace("]/", "]")
        return text, entity_map

    def _tokenize_and_transliterate(self, sentence: str, normalizer,
                                    iso_lang: str, transliterate: bool) -> str:
        from indicnlp.tokenize import indic_tokenize
        normed = normalizer.normalize(sentence.strip())
        joined = " ".join(indic_tokenize.trivial_tokenize(normed, iso_lang))
        if transliterate:
            joined = self._xliterator.transliterate(joined, iso_lang, "hi")
            joined = joined.replace(" ् ", "्")
        return joined

    # -- public -----------------------------------------------------------

    def preprocess(self, text: str, src_lang: str,
                   tgt_lang: Optional[str] = None,
                   is_target: bool = False) -> Tuple[str, Dict[str, str]]:
        """One sentence -> ``"<src_tag> <tgt_tag> tokenised text"`` + its map.

        The two tags are what select the language pair. They are not optional
        and they are not a hint: without them IndicTrans2 produces fluent text
        in whatever language it guesses.
        """
        if src_lang not in LANGUAGE_TAGS:
            raise ValueError(
                f"{src_lang!r} is not an IndicTrans2 language tag; "
                f"expected one of {sorted(LANGUAGE_TAGS)}")
        if not is_target and tgt_lang not in LANGUAGE_TAGS:
            raise ValueError(
                f"{tgt_lang!r} is not an IndicTrans2 language tag; "
                f"expected one of {sorted(LANGUAGE_TAGS)}")

        iso_lang = FLORES_TO_ISO.get(src_lang, "hi")
        script = src_lang.split("_")[1]

        text = self._punc_norm(text)
        text = text.translate(self._digits_table)
        entity_map: Dict[str, str] = {}
        if self.inference:
            text, entity_map = self._wrap_with_placeholders(text)

        if iso_lang == "en":
            normed = self._en_normalizer.normalize(text.strip())
            processed = " ".join(self._en_tok.tokenize(normed, escape=False))
        else:
            processed = self._tokenize_and_transliterate(
                text, self._normalizer_for(iso_lang), iso_lang,
                transliterate=script not in NON_TRANSLITERATED_SCRIPTS)

        processed = processed.strip()
        if is_target:
            return processed, entity_map
        return f"{src_lang} {tgt_lang} {processed}", entity_map

    def postprocess(self, text: str, lang: str,
                    entity_map: Optional[Dict[str, str]] = None) -> str:
        """The exact inverse: script fix-ups, placeholders back, detokenise.

        The transliteration here is the half people forget. The model emits
        Devanagari for every transliterated script, so a ``tam_Taml`` request
        that skips this step comes back as fluent Tamil written in Devanagari -
        wrong, and silent.
        """
        entity_map = entity_map or {}
        lang_code, script_code = lang.split("_", 1)

        if script_code in ("Arab", "Aran"):
            text = (text.replace(" ؟", "؟").replace(" ۔", "۔")
                        .replace(" ،", "،").replace("ٮ۪", "ؠ"))
        if lang_code == "ory":
            text = text.replace("ଯ଼", "ୟ")

        for placeholder, original in entity_map.items():
            text = text.replace(placeholder, original)

        if lang == "eng_Latn":
            return self._en_detok.detokenize(text.split(" "))
        from indicnlp.tokenize import indic_detokenize
        iso_lang = FLORES_TO_ISO.get(lang, "hi")
        xlated = self._xliterator.transliterate(text, "hi", iso_lang)
        return indic_detokenize.trivial_detokenize(xlated, iso_lang)

    # -- batch API, matching upstream's shape -----------------------------

    def preprocess_batch(self, batch: List[str], src_lang: str,
                         tgt_lang: Optional[str] = None,
                         is_target: bool = False
                         ) -> Tuple[List[str], List[Dict[str, str]]]:
        results = [self.preprocess(s, src_lang, tgt_lang, is_target) for s in batch]
        return [r[0] for r in results], [r[1] for r in results]

    def postprocess_batch(self, sents: List[str], lang: str = "hin_Deva",
                          entity_maps: Optional[List[Dict[str, str]]] = None
                          ) -> List[str]:
        maps = entity_maps or [{} for _ in sents]
        return [self.postprocess(s, lang, m) for s, m in zip(sents, maps)]
