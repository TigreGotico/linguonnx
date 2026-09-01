"""Tokenizers for the translation models, with no `transformers` dependency.

Most of this is `sentencepiece` plus a JSON vocabulary, and those
architectures differ in exactly two places - how a SentencePiece piece becomes
an id, and what wraps the sentence - so both live in one class per family.
One export ships no SentencePiece model at all and keeps its whole vocabulary
in a `tokenizers` JSON file; :class:`UnigramTextPrefixTokenizer` reads that
directly rather than taking the dependency.

`SpmSeq2SeqTokenizer`
    M2M100 and NLLB. Encodes ``[src_lang] + pieces + [</s>]``. The target
    language is *not* in the input: it is forced as the decoder's first
    generated token (see :mod:`linguonnx.translate.decode`).

    M2M100 reads ids straight out of ``vocab.json``. NLLB has no ``vocab.json``
    and uses fairseq's convention instead: ``id = sp_id + 1``, with the four
    fairseq specials squatting on ids 0-3. Both put their language tokens in a
    block after the ordinary vocabulary, in the order given by
    ``special_tokens_map.json``.

`MarianTokenizer`
    opus-mt. A separate SentencePiece model per side, a shared ``vocab.json``,
    and ``</s>`` appended with no language token at all - the model *is* the
    pair. Marian also expects Moses punctuation normalisation on the input,
    ported here in :func:`normalize_punctuation`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sentencepiece as spm

__all__ = ["normalize_punctuation", "SpmSeq2SeqTokenizer", "MarianTokenizer",
           "FastUnigramTokenizer", "T5SpmTokenizer", "T5TextPrefixTokenizer",
           "UnigramTextPrefixTokenizer", "IndicTransTokenizer",
           "OpenNmtBpeTokenizer", "load_tokenizer", "artifact_lang_codes",
           "bare_lang_code"]


# --------------------------------------------------------------------------
# Moses punctuation normalisation
# --------------------------------------------------------------------------
# A port of sacremoses' MosesPunctNormalizer default replacement chain, which
# MarianTokenizer runs over every input. On plain ASCII it is the identity;
# it matters for curly quotes, CJK full-width punctuation and French quotes,
# which is exactly the text a translator sees. Kept here so linguonnx does not
# take a dependency on sacremoses (which pulls in click, joblib, regex).

_EXTRA_WHITESPACE = [
    (r"\r", r""),
    (r"\(", r" ("),
    (r"\)", r") "),
    (r" +", r" "),
    (r"\) ([.!:?;,])", r")\g<1>"),
    (r"\( ", r"("),
    (r" \)", r")"),
    (r"(\d) %", r"\g<1>%"),
    (r" :", r":"),
    (r" ;", r";"),
]

_NORMALIZE_UNICODE = [
    ("„", r'"'), ("“", r'"'), ("”", r'"'), ("–", r"-"),
    ("—", r" - "), (r" +", r" "), ("´", r"'"),
    ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
    ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
    ("‘", r"'"), ("‹", r"'"), ("›", r"'"), ("’", r"'"),
    ("ʼ", r"'"), ("‚", r"'"), ("′", r"'"), ("″", r'"'),
    ("«", r'"'), ("»", r'"'), ("‟", r'"'), ("‶", r'"'),
    ("〃", r'"'), ("˝", r'"'),
    # CJK punctuation ("。" -> ".") is deliberately absent: sacremoses keeps it
    # in `replace_unicode_punct`, which MarianTokenizer does not call.
]

_FRENCH_QUOTES = [
    (" « ", r'"'), ("« ", r'"'), ("«", r'"'),
    (" » ", r'"'), (" »", r'"'), ("»", r'"'),
]

_HANDLE_PSEUDO_SPACES = [
    (" %", r"%"), ("nbsp;", r" "), (" :", r":"), (" º C", r" ºC"),
    (" cm", r" cm"), (" \\?", r"?"), (" \\!", r"!"), (" ;", r";"),
    (", ", r", "), (r" +", r" "),
]

_EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]

_SUBSTITUTIONS = [
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        _EXTRA_WHITESPACE + _NORMALIZE_UNICODE + _FRENCH_QUOTES
        + _HANDLE_PSEUDO_SPACES + _EN_QUOTATION_FOLLOWED_BY_COMMA
    )
]


def normalize_punctuation(text: str) -> str:
    """Moses punctuation normalisation, as Marian models were trained with."""
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _load_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


#: SentencePiece's word marker. Both instruction-prefixed readers put one in
#: front of every span they segment, which is what "prepend_scheme: always"
#: means in a `tokenizers` Metaspace pre-tokenizer.
_WORD_MARK = "\u2581"

#: Runs of spaces left by a vocabulary that carries the separator twice.
_SPACE_RUN = re.compile(" {2,}")


def _reject_unusable_instruction(prefix: str, ids: Sequence[int],
                                 unk_id: int) -> None:
    """An instruction the vocabulary cannot spell is not an instruction.

    It does not raise on its own: it reaches the encoder as noise, and the
    model answers in whatever direction it settles on. Both readers check it
    the same way, because the failure is the same in both.
    """
    if not ids:
        raise ValueError(
            f"{prefix!r} tokenises to nothing in this model's vocabulary, so "
            f"the encoder would receive no instruction and the model would "
            f"pick a direction of its own")
    if unk_id in ids:
        raise ValueError(
            f"{prefix!r} contains characters this model's vocabulary cannot "
            f"represent; the instruction would reach the encoder as <unk> and "
            f"the model would pick a direction of its own")


def _load_spm(path) -> "spm.SentencePieceProcessor":
    processor = spm.SentencePieceProcessor()
    processor.Load(str(path))
    return processor


# --------------------------------------------------------------------------
# M2M100 / NLLB
# --------------------------------------------------------------------------

#: M2M100 writes its language tokens wrapped (``__ca__``); NLLB writes them
#: bare (``cat_Latn``). Only the wrapper is stripped, never the code itself.
_M2M100_LANG_TOKEN_RE = re.compile(r"^__([^_].*?)__$")


def artifact_lang_codes(files: Dict[str, Path]) -> List[str]:
    """The language codes *this export actually carries*, in its own id order.

    The registry's ``languages`` list is a routing claim. For a bilingual
    entry it is absent entirely - the pair says what the model is for - and
    the language block would then be built from nothing, so every call raises
    ``KeyError: language code 'ca' is not supported by this model`` on a model
    whose vocabulary does carry ``__ca__``. The claim is not the artifact, so
    the artifact is read instead: ``special_tokens_map.json``'s
    ``additional_special_tokens`` is the same list, in the same order, that
    fixes the ids.

    Order is preserved rather than sorted: it *is* the id order of the
    language-token block.
    """
    path = files.get("special_tokens_map")
    if path is None:
        raise ValueError(
            "no language codes: the registry entry lists none and the export "
            "ships no special_tokens_map.json to read them from. Without them "
            "the source tag and the forced target token cannot be built, and "
            "the model would translate into whatever it guesses.")
    tokens = _load_json(path).get("additional_special_tokens") or []
    codes = []
    for token in tokens:
        if not isinstance(token, str):
            continue
        match = _M2M100_LANG_TOKEN_RE.match(token)
        codes.append(match.group(1) if match else token)
    if not codes:
        raise ValueError(
            f"{path} carries no additional_special_tokens, so this export "
            f"declares no language tokens at all")
    return codes


def bare_lang_code(piece: str) -> Optional[str]:
    """``__fr__`` -> ``fr``. Anything that is not a wrapped token -> ``None``."""
    match = _M2M100_LANG_TOKEN_RE.match(piece)
    return match.group(1) if match else None


class SpmSeq2SeqTokenizer:
    """M2M100 and NLLB.

    ``lang_codes`` are the model's own codes (``pt``, or ``por_Latn``), in the
    order they appear in ``special_tokens_map.json``.

    Where the ids come from
    -----------------------
    When the export ships an ``added_tokens.json`` - every M2M100 checkpoint
    does - **that file is the only source of language-token ids**, and
    ``lang_codes`` is used to *check* it, never to count through it. Both
    spellings address the same token, so ``fr`` and ``__fr__`` are equivalent.

    Order still matters for NLLB, which ships no ``added_tokens.json``: there
    the block really is positional, ``lang_block_start + i``, and
    ``lang_codes`` has to arrive in the export's own id order.

    Why the fallback went away
    --------------------------
    This used to try ``added_tokens`` and, if the number of codes it matched
    did not equal ``len(lang_codes)``, silently rebuild the whole map by
    position instead. That is only correct when ``lang_codes`` is
    byte-for-byte the export's own token order, and for a *registry-declared*
    list it is not: the registry stores normalised BCP-47, and
    ``normalize_tag('tl')`` is ``'fil'``. On ``m2m100-418M`` the two lists
    were the same length and differed by one entry in the middle, so the
    counts matched, the fallback ran, and 64 of 100 languages landed on the
    *next* language's token - ``fr`` on Frisian, ``pt`` on Romanian, ``ru`` on
    Sindhi - on the encoder's source tag and the decoder's forced BOS alike.
    On ``m2m100-418M-smugri``, 8 declared codes against a 104-token block put
    Finnish on ``__ar__`` and returned Arabic script. Nothing raised in either
    case.

    So a code the export has no token for is now a hard error, here, at load
    time. A model that cannot address a language has to say so rather than
    quietly address a different one.
    """

    def __init__(self, spm_path, lang_codes: Sequence[str],
                 vocab_path=None, fairseq_offset: int = 0,
                 added_tokens_path=None,
                 eos_id: int = 2, pad_id: int = 1, unk_id: int = 3,
                 bos_id: int = 0):
        self.sp = _load_spm(spm_path)
        self.fairseq_offset = fairseq_offset
        self.eos_id, self.pad_id, self.unk_id, self.bos_id = eos_id, pad_id, unk_id, bos_id

        self.vocab: Optional[Dict[str, int]] = None
        self.inv_vocab: Optional[Dict[int, str]] = None
        if vocab_path is not None:
            self.vocab = _load_json(vocab_path)
            self.inv_vocab = {i: t for t, i in self.vocab.items()}
            lang_block_start = len(self.vocab)
        else:
            lang_block_start = self.sp.get_piece_size() + fairseq_offset

        if added_tokens_path is not None and Path(added_tokens_path).exists():
            # Authoritative, and used alone. See the class docstring.
            added = _load_json(added_tokens_path)
            # M2M100 spells the key wrapped (`__ca__`) while `lang_codes` is
            # bare (`ca`); both address the same token.
            self.lang_code_to_id = {}
            for piece, token_id in added.items():
                self.lang_code_to_id[piece] = int(token_id)
                bare = bare_lang_code(piece)
                if bare is not None:
                    self.lang_code_to_id.setdefault(bare, int(token_id))
            unknown = [code for code in lang_codes
                       if code not in self.lang_code_to_id]
            if unknown:
                raise ValueError(
                    f"this export's added_tokens.json has no language token "
                    f"for {unknown!r}; it carries {len(added)} tokens "
                    f"({', '.join(sorted(added)[:6])}, ...). A language is "
                    f"advertised that the model cannot address. Either the "
                    f"registry entry's `native_codes` must map it to a token "
                    f"the export really has, or the claim must be dropped - "
                    f"deriving the id from a code's position in the declared "
                    f"list translates into a different language and nothing "
                    f"reports it.")
        else:
            self.lang_code_to_id = {code: lang_block_start + i
                                    for i, code in enumerate(lang_codes)}
        # `__fr__` and `fr` are two names for one id, so this inverse keeps
        # whichever was inserted last (the bare code, for M2M100).
        self.id_to_lang_code = {i: c for c, i in self.lang_code_to_id.items()}
        # Take the ids from the values, not from the inverse: two names per id
        # make the inverse smaller than the map, and a language token missing
        # from `_specials` is one that survives `decode()` as `⁇`.
        self._specials = {self.eos_id, self.pad_id, self.unk_id, self.bos_id}
        self._specials |= set(self.lang_code_to_id.values())

    # -- ids ------------------------------------------------------------

    def piece_to_id(self, piece: str) -> int:
        if self.vocab is not None:
            return self.vocab.get(piece, self.unk_id)
        sp_id = self.sp.PieceToId(piece)
        # PieceToId returns 0 for "not in the model", which after the fairseq
        # offset would collide with <pad>. transformers maps it to <unk>.
        return sp_id + self.fairseq_offset if sp_id else self.unk_id

    def id_to_piece(self, token_id: int) -> str:
        if self.vocab is not None:
            return self.inv_vocab.get(token_id, "<unk>")
        return self.sp.IdToPiece(token_id - self.fairseq_offset)

    def lang_id(self, code: str) -> int:
        if code not in self.lang_code_to_id:
            raise KeyError(
                f"language code {code!r} is not supported by this model; its "
                f"vocabulary carries {len(self.lang_code_to_id)} language "
                f"tokens")
        return self.lang_code_to_id[code]

    # -- encode / decode -------------------------------------------------

    def encode(self, text: str, src_lang: str) -> List[int]:
        pieces = self.sp.encode(text, out_type=str)
        return ([self.lang_id(src_lang)]
                + [self.piece_to_id(p) for p in pieces]
                + [self.eos_id])

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [self.id_to_piece(i) for i in ids if i not in self._specials]
        return self.sp.DecodePieces(pieces)


# --------------------------------------------------------------------------
# Marian / opus-mt
# --------------------------------------------------------------------------

class MarianTokenizer:
    """opus-mt. No language token: the model is the pair.

    Some opus-mt models (the multi-target ``tc-big`` and ``ROMANCE`` ones) do
    carry ``>>xxx<<`` target tokens in their vocabulary. They are *not* added
    automatically, because `transformers` does not add them either and the
    published parity numbers were measured without them. Pass
    ``target_token=">>por<<"`` to opt in.
    """

    def __init__(self, source_spm, target_spm, vocab_path,
                 eos_token: str = "</s>", pad_token: str = "<pad>",
                 unk_token: str = "<unk>"):
        """``eos_token``/``pad_token``/``unk_token`` default to the standard
        opus-mt spellings, but not every Marian export uses them: Softcatalà/
        BSC's fairseq-derived exports (e.g. ``aina-translator-ca-*``) spell
        the pad token ``<blank>`` instead of ``<pad>``. Looking the wrong
        string up in ``vocab.json`` is a ``KeyError``, not a silent wrong
        answer, so a mismatched repo fails loudly here rather than mistranslating
        - see :func:`load_tokenizer`, which reads the real strings out of the
        repo's own ``special_tokens_map.json`` before falling back to these
        defaults.
        """
        self.spm_source = _load_spm(source_spm)
        self.spm_target = _load_spm(target_spm)
        self.vocab: Dict[str, int] = _load_json(vocab_path)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.eos_id = self.vocab[eos_token]
        self.pad_id = self.vocab[pad_token]
        self.unk_id = self.vocab[unk_token]
        self._specials = {self.eos_id, self.pad_id, self.unk_id}
        self.language_tokens = sorted(
            t for t in self.vocab if t.startswith(">>") and t.endswith("<<"))

    def encode(self, text: str, target_token: Optional[str] = None) -> List[int]:
        pieces = self.spm_source.encode(normalize_punctuation(text), out_type=str)
        ids = [self.vocab.get(p, self.unk_id) for p in pieces]
        if target_token:
            if target_token not in self.vocab:
                raise KeyError(
                    f"{target_token!r} is not in this model's vocabulary; "
                    f"it knows {self.language_tokens}")
            ids = [self.vocab[target_token]] + ids
        return ids + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [self.inv_vocab.get(i, self.unk_id) for i in ids
                  if i not in self._specials]
        pieces = [p for p in pieces if isinstance(p, str)
                  and not (p.startswith(">>") and p.endswith("<<"))]
        return self.spm_target.DecodePieces(pieces)


class FastUnigramTokenizer:
    """Softcatalà's ``translate-eus-cat``/``translate-oci-cat``: a Marian-
    shaped Pegasus export whose tokenizer is not a raw SentencePiece
    ``.model`` at all - it is a single Unigram model serialized in HF's
    ``tokenizers`` (Rust) library JSON format (``tokenizer.json``), complete
    with its own precompiled-charsmap normaliser and decoder. `sentencepiece`
    cannot read this file; it needs the ``tokenizers`` package instead.

    Each of the two repos this loads for is a plain dedicated pair with no
    group-model ambiguity (confirmed by inspecting ``vocab.json``: neither
    ships any ``>>xxx<<`` token), so, like :class:`MarianTokenizer`, it never
    needs a target token.
    """

    def __init__(self, tokenizer_json, eos_token: str = "</s>",
                 pad_token: str = "<blank>", unk_token: str = "<unk>"):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "this model's tokenizer is a `tokenizers`-library "
                "tokenizer.json, which needs the 'tokenizers' package. "
                "Install it: pip install 'linguonnx[fast-tokenizers]'") from exc
        self._tok = Tokenizer.from_file(str(tokenizer_json))
        vocab = self._tok.get_vocab()
        self.eos_id = vocab[eos_token]
        self.pad_id = vocab[pad_token]
        self.unk_id = vocab[unk_token]
        self._specials = {self.eos_id, self.pad_id, self.unk_id}

    def encode(self, text: str, target_token: Optional[str] = None) -> List[int]:
        # `target_token` is accepted only so this class has the same call
        # shape as `MarianTokenizer` and can share `MarianPipeline`; neither
        # repo this loads for has a group-model prefix token, so a caller
        # that passes one gets a clear error rather than a silently ignored
        # argument.
        if target_token:
            raise KeyError(
                f"{target_token!r} was requested but this model has no "
                f"target-selection tokens; it is a plain dedicated pair")
        ids = self._tok.encode(text, add_special_tokens=False).ids
        return ids + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        ids = [i for i in ids if i not in self._specials]
        return self._tok.decode(ids, skip_special_tokens=False)


class T5SpmTokenizer:
    """MADLAD (T5). One SentencePiece model; ids are its own piece ids.

    The target language is not a decoder-forced id, unlike M2M100/NLLB: it is
    a `<2xx>` piece **prepended to the input text**, exactly like any other
    piece, because MADLAD was trained on ``"<2pt> hello"`` style examples. The
    piece is already in the SentencePiece model (see
    :func:`scripts.sync_registry._madlad_languages`), so encoding it is just
    string concatenation before the normal `sp.encode`.
    """

    def __init__(self, spm_path, eos_id: int = 2, pad_id: int = 1, unk_id: int = 0):
        self.sp = _load_spm(spm_path)
        self.eos_id, self.pad_id, self.unk_id = eos_id, pad_id, unk_id
        self._specials = {self.eos_id, self.pad_id, self.unk_id}

    def encode(self, text: str, prefix: Optional[str] = None) -> List[int]:
        if prefix:
            # A `<2xx>` that is not a piece of this SentencePiece model is not
            # dropped by `sp.encode`; it is shredded into `<`, `2`, `xx`, `>`,
            # which the model has never seen in that position and quietly
            # ignores. The result is a translation into the wrong language
            # with nothing raised, so the piece is checked before use.
            if self.sp.PieceToId(prefix) == self.unk_id:
                raise ValueError(
                    f"{prefix!r} is not a piece of this model's SentencePiece "
                    f"vocabulary, so it cannot select a target language")
        full = f"{prefix} {text}" if prefix else text
        ids = [max(i, self.unk_id) for i in self.sp.encode(full, out_type=int)]
        if prefix and self.sp.PieceToId(prefix) not in ids:
            raise ValueError(
                f"{prefix!r} did not survive tokenisation of the input; the "
                f"encoder would receive no target-language signal")
        return ids + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [self.sp.IdToPiece(i) for i in ids if i not in self._specials]
        return self.sp.DecodePieces(pieces)


class T5TextPrefixTokenizer:
    """An instruction-prefixed T5/UMT5. One SentencePiece model, and the task
    is chosen by a natural-language sentence prepended to the input.

    This is deliberately *not* a subclass of :class:`T5SpmTokenizer`. MADLAD's
    prefix is a single `<2xx>` piece and its checks assert exactly that - the
    piece must exist in the vocabulary, and it must survive tokenisation
    whole. Here the prefix is a sentence that tokenises to many ordinary
    pieces, so both assertions are false by construction. Inheriting and then
    switching them off is how the switched-off version becomes the default for
    the next architecture that borrows this class.

    The guard that *does* transfer is the reason MADLAD has one at all: a
    prefix the SentencePiece model cannot represent does not raise, it just
    reaches the encoder as noise, and the model answers fluently in whatever
    direction it prefers. So the prefix is required to tokenise to at least
    one piece and to contain no `<unk>`.

    The defaults are T5's own (``pad=0``, ``eos=1``), which are *not*
    MADLAD's (``eos=2``, ``pad=1``); they are passed in from the export's
    `config.json` rather than trusted from here. ``unk_id`` defaults to
    whatever the SentencePiece model itself declares, which is the only
    authority on it - `config.json` does not carry one.

    Added tokens are not optional here
    ----------------------------------
    The SentencePiece model does not hold the whole vocabulary. Thalesian's
    Akkadian exports carry 32000 pieces against a vocabulary of 32518, and
    the 518 that are missing are the cuneiform signs themselves, plus the
    diacritics transliteration is written with - they live in
    ``added_tokens.json``.

    So ``sp.encode`` alone turns every cuneiform sign in the input into
    ``<unk>``. It does not raise: the encoder receives a sentence of unknown
    tokens and the decoder answers it fluently, which is why this is a file
    the loader refuses to run without rather than one it treats as absent.
    Decoding is the louder half - an added-token id is simply out of the
    SentencePiece model's range and ``IdToPiece`` raises ``IndexError`` - but
    it is the quiet encoding half that would corrupt the translation.

    Text is therefore split on the added-token literals, longest first, and
    only the spans between them reach ``sp.encode``.

    ``legacy``
    ----------
    SentencePiece prepends a dummy ``\u2581`` to everything it encodes.
    `transformers` stopped doing that for T5 when ``legacy`` is false, which
    changes the *first* piece of every span - "Translate" tokenises as
    ``Trans``/``late`` rather than ``\u2581Translat``/``e``. It is one piece
    in a sentence and it never raises, so the model just receives a slightly
    different input than it was trained on, on every single call. The flag is
    read from the export's ``tokenizer_config.json``; it is not guessed.
    """

    def __init__(self, spm_path, added_tokens_path, eos_id: int = 1,
                 pad_id: int = 0, unk_id: Optional[int] = None,
                 legacy: bool = True, unk_piece: str = "<unk>"):
        self.sp = _load_spm(spm_path)
        self.legacy = legacy
        # How many pieces the unk spelling costs, so the dummy prefix that
        # attaches to it can be dropped with it. See the class docstring.
        self._unk_piece = unk_piece
        self._unk_prefix_len = len(self.sp.encode(unk_piece, out_type=int))
        self.eos_id, self.pad_id = eos_id, pad_id
        self.unk_id = self.sp.unk_id() if unk_id is None else unk_id
        self._specials = {self.eos_id, self.pad_id, self.unk_id}

        self.added: Dict[str, int] = {
            piece: int(token_id)
            for piece, token_id in _load_json(added_tokens_path).items()}
        self._from_id = {token_id: piece for piece, token_id in self.added.items()}
        # Longest first, so a literal that starts with another one is matched
        # whole rather than split across its own prefix.
        self._added_re = re.compile("|".join(
            re.escape(piece) for piece in
            sorted(self.added, key=len, reverse=True))) if self.added else None

    def encode(self, text: str, prefix: Optional[str] = None) -> List[int]:
        if prefix:
            _reject_unusable_instruction(prefix,
                                         self._encode_with_added(prefix),
                                         self.unk_id)
        # The card's own usage snippet joins the two as ``prompt + text`` with
        # the prompt ending in ": ", so the separator is part of the trained
        # surface form and is applied here rather than left to the caller.
        full = f"{prefix}: {text}" if prefix else text
        return self._encode_with_added(full) + [self.eos_id]

    def _encode_with_added(self, text: str) -> List[int]:
        # No `max(i, unk_id)` clamp on the spans, unlike `T5SpmTokenizer`.
        # That clamp is harmless only because MADLAD's unk is 0; T5 numbers
        # its unk 2, above both pad and eos, so the same line would rewrite
        # those two ids into <unk> rather than floor anything.
        if self._added_re is None:
            return self._sp_encode(text)
        ids: List[int] = []
        position = 0
        for match in self._added_re.finditer(text):
            span = text[position:match.start()]
            if span:
                ids.extend(self._sp_encode(span))
            ids.append(self.added[match.group()])
            position = match.end()
        if text[position:]:
            ids.extend(self._sp_encode(text[position:]))
        return ids

    def _sp_encode(self, text: str) -> List[int]:
        if self.legacy:
            return self.sp.encode(text, out_type=int)
        # Encode the unk spelling in front so SentencePiece's dummy prefix
        # lands on that instead, then drop both.
        return self.sp.encode(self._unk_piece + text,
                              out_type=int)[self._unk_prefix_len:]

    def decode(self, ids: Sequence[int]) -> str:
        """Decoded runs and added tokens, joined with a single space.

        The space is the reference tokenizer's rule and is kept for parity
        rather than for looking right. Nothing in the ids says whether two
        cuneiform signs were written apart, because a space between two added
        tokens is not itself a token - so the separator has to be supplied on
        the way out, and the reference supplies one.

        It is not always what a reader wants: transliteration brackets are
        added tokens too, so ``a-na ⌈ma⌉-ri`` comes back as
        ``a-na ⌈ ma ⌉ -ri``. Spacing it any other way would make this
        library's output differ from every other consumer of the same
        weights, which is the worse of the two.
        """
        out: List[str] = []
        run: List[str] = []
        for token_id in ids:
            if token_id in self._specials:
                continue
            piece = self._from_id.get(token_id)
            if piece is None:
                run.append(self.sp.IdToPiece(token_id))
                continue
            if run:
                out.append(self.sp.DecodePieces(run))
                run = []
            out.append(piece)
        if run:
            out.append(self.sp.DecodePieces(run))
        return " ".join(part for part in out if part)


class UnigramTextPrefixTokenizer:
    """The same instruction-prefixed T5, where the export ships a
    ``tokenizer.json`` instead of a SentencePiece model.

    `cuneiformBase-400m` has no ``spiece.model`` at all: its vocabulary is a
    Unigram model serialised in the `tokenizers` library's JSON format. That
    file is the one thing the export has, so the choice is to read it or to
    drop the model.

    Reading it does not need the `tokenizers` package. A Unigram model is a
    vocabulary of pieces with log probabilities, and segmenting with it is a
    Viterbi pass picking the highest-scoring path over the piece lattice -
    thirty lines, no build step, and it holds the tokenizer to the same
    "no torch, no transformers" rule as the rest of the library.

    What it does **not** implement is the rest of a `tokenizers` pipeline.
    A ``normalizer`` - Unicode NFKC folding compiled into a
    ``precompiled_charsmap`` - would change the text before the lattice ever
    saw it, and quietly ignoring one produces a plausible tokenisation of the
    wrong string. Anything this class cannot honour is a refusal at load
    time, not a silent approximation, so the two Softcatalà exports that do
    carry a normaliser keep going to :class:`FastUnigramTokenizer`.
    """

    def __init__(self, tokenizer_json, eos_id: int = 1, pad_id: int = 0,
                 unk_id: Optional[int] = None):
        spec = _load_json(tokenizer_json)
        model = spec.get("model") or {}
        if model.get("type") != "Unigram":
            raise ValueError(
                f"{tokenizer_json} is a {model.get('type')!r} tokenizer; this "
                f"reader implements Unigram only")
        if spec.get("normalizer") is not None:
            raise ValueError(
                f"{tokenizer_json} declares a normaliser, which this reader "
                f"does not implement. Ignoring it would tokenise a string the "
                f"model was never given; install the 'fast-tokenizers' extra "
                f"for exports that carry one")
        if model.get("byte_fallback"):
            raise ValueError(
                f"{tokenizer_json} uses byte fallback, so an out-of-vocabulary "
                f"character becomes byte pieces rather than <unk>; this reader "
                f"does not implement that")

        self.pieces = [piece for piece, _ in model["vocab"]]
        self._score = {piece: (index, score)
                       for index, (piece, score) in enumerate(model["vocab"])}
        self._longest = max(len(piece) for piece in self.pieces)
        self.eos_id, self.pad_id = eos_id, pad_id
        self.unk_id = model["unk_id"] if unk_id is None else unk_id

        # Added tokens are matched before the lattice and are never scored.
        # In this export the space is one of them, which is why a span handed
        # to the lattice never contains one and gets exactly one word marker.
        self.added = {entry["content"]: int(entry["id"])
                      for entry in spec.get("added_tokens", ())}
        self._from_id = {token_id: piece for piece, token_id in self.added.items()}
        self._added_re = re.compile("|".join(
            re.escape(piece) for piece in
            sorted(self.added, key=len, reverse=True))) if self.added else None
        self._specials = {self.eos_id, self.pad_id, self.unk_id}

    def encode(self, text: str, prefix: Optional[str] = None) -> List[int]:
        if prefix:
            # Through the added-token path, not the bare lattice: the space
            # is an added token here, and a lattice given one on its own
            # would report the whole instruction as unrepresentable.
            _reject_unusable_instruction(prefix,
                                         self._encode_with_added(prefix),
                                         self.unk_id)
        full = f"{prefix}: {text}" if prefix else text
        return self._encode_with_added(full) + [self.eos_id]

    def _encode_with_added(self, text: str) -> List[int]:
        if self._added_re is None:
            return self._words(text)
        ids: List[int] = []
        position = 0
        for match in self._added_re.finditer(text):
            if match.start() > position:
                ids.extend(self._words(text[position:match.start()]))
            ids.append(self.added[match.group()])
            position = match.end()
        if text[position:]:
            ids.extend(self._words(text[position:]))
        return ids

    def _words(self, span: str) -> List[int]:
        """One marked word per run of non-space, the rest dropped.

        The pre-tokenizer splits on whitespace and throws it away. That the
        plain space survives at all is not this step's doing - it is an added
        token, matched before any of this - so a tab or a newline leaves no
        token behind, while a space leaves its own.
        """
        ids: List[int] = []
        for word in span.split():
            ids.extend(self._segment(_WORD_MARK + word))
        return ids

    def _segment(self, span: str) -> List[int]:
        """Viterbi over the piece lattice: the best-scoring path wins."""
        length = len(span)
        best: List[Optional[Tuple[float, int, int]]] = [None] * (length + 1)
        best[0] = (0.0, -1, -1)
        for end in range(1, length + 1):
            for start in range(max(0, end - self._longest), end):
                if best[start] is None:
                    continue
                hit = self._score.get(span[start:end])
                if hit is None:
                    continue
                index, score = hit
                candidate = (best[start][0] + score, start, index)
                if best[end] is None or candidate[0] > best[end][0]:
                    best[end] = candidate
            if best[end] is None and best[end - 1] is not None:
                # A character no piece covers costs nothing and becomes <unk>,
                # which is what the reference implementation does rather than
                # failing the whole segmentation.
                best[end] = (best[end - 1][0], end - 1, self.unk_id)
        ids: List[int] = []
        position = length
        while position > 0:
            _, start, index = best[position]
            ids.append(index)
            position = start
        return ids[::-1]

    def decode(self, ids: Sequence[int]) -> str:
        """Pieces joined, word marks turned back into spaces, runs collapsed.

        A space between two words is carried twice in this vocabulary - once
        as the added token for `" "`, and again as the next word's mark - so
        decoding both gives `"the  king"`. The reference implementation
        returns exactly that, and it is the one place here where matching it
        is the worse choice: re-encoding its output yields an extra space
        token, so text that goes out and comes back drifts a token per space
        per cycle. Collapsing the run keeps the round trip stable and cannot
        change meaning, since a run of spaces was never more than a
        separator. Cuneiform is unaffected - signs are single-spaced already.
        """
        out: List[str] = []
        for token_id in ids:
            if token_id in self._specials:
                continue
            piece = self._from_id.get(token_id)
            out.append(piece if piece is not None else self.pieces[token_id])
        return _SPACE_RUN.sub(" ", "".join(out).replace(_WORD_MARK, " ")).strip(" ")


class IndicTransTokenizer:
    """IndicTrans2. Two SentencePiece models and two plain JSON dictionaries.

    Source and target are separate vocabularies, so encoding and decoding do
    not share a table. Encoding is ``[src_tag, tgt_tag] + pieces + [</s>]``,
    with the two tags taken from the *already preprocessed* string - they are
    prefixed by :class:`~linguonnx.translate._indic_processor.IndicProcessor`,
    not here, because the text they describe has to be normalised the same way
    the tags claim.

    This is a port of AI4Bharat's ``tokenization_indictrans.py``, which
    linguonnx cannot use directly: it is remote code loaded through
    ``trust_remote_code`` and subclasses ``PreTrainedTokenizer``. The logic it
    contains is four lines of SentencePiece plus two dict lookups, reproduced
    below.
    """

    #: The models ship a frozen sinusoidal position table of this size, so an
    #: input longer than this cannot be embedded at all.
    MAX_POSITIONS = 256

    def __init__(self, src_spm, tgt_spm, src_vocab, tgt_vocab,
                 unk_token: str = "<unk>", pad_token: str = "<pad>",
                 eos_token: str = "</s>", bos_token: str = "<s>"):
        self.spm_source = _load_spm(src_spm)
        self.spm_target = _load_spm(tgt_spm)
        self.src_encoder: Dict[str, int] = _load_json(src_vocab)
        self.tgt_encoder: Dict[str, int] = _load_json(tgt_vocab)
        self.tgt_decoder = {i: t for t, i in self.tgt_encoder.items()}
        self.unk_id = self.src_encoder[unk_token]
        self.pad_id = self.src_encoder[pad_token]
        self.eos_id = self.src_encoder[eos_token]
        self.bos_id = self.src_encoder[bos_token]
        self._specials = {self.src_encoder[t] for t in
                          (unk_token, pad_token, eos_token, bos_token)}

    def encode(self, tagged_text: str) -> List[int]:
        """``"hin_Deva eng_Latn नमस्ते"`` -> ids, tags included.

        The tags are split off the front and looked up as whole vocabulary
        entries; they must not go through SentencePiece, which would shred
        ``hin_Deva`` into pieces the model has never seen in that position.
        """
        parts = tagged_text.split(" ", 2)
        if len(parts) < 3:
            raise ValueError(
                "IndicTrans2 input must start with a source and a target tag, "
                f"as '<src_tag> <tgt_tag> <text>'; got {tagged_text!r}")
        src_tag, tgt_tag, text = parts
        pieces = [src_tag, tgt_tag] + self.spm_source.EncodeAsPieces(text)
        return [self.src_encoder.get(p, self.unk_id) for p in pieces] + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        """Ids -> text, still tokenised and still in Devanagari.

        Turning that back into the target script is
        :meth:`IndicProcessor.postprocess`'s job, not this method's.
        """
        pieces = [self.tgt_decoder.get(i, "<unk>") for i in ids
                  if i not in self._specials]
        return "".join(pieces).replace("▁", " ").strip()


class OpenNmtBpeTokenizer:
    """Proxecto Nós ``nos-coda_iacobus``. Moses tokenisation + subword-nmt BPE.

    OpenNMT-py keeps **separate source and target vocabularies**. The export
    concatenates them as ``[target | source]``, so an encoder input id is a
    source index plus ``source_offset`` while a decoder output id indexes the
    target half directly. Getting that offset wrong does not raise; it silently
    feeds the model a different sentence.

    ``<unk>`` is kept in the output on purpose. Upstream ``onmt_translate``
    hides it with ``-replace_unk``, which copies the aligned source word using
    the decoder's cross-attention weights. Those weights are not outputs of the
    exported graph, so the substitution cannot be reproduced - and inventing a
    replacement would be a guess presented as a translation. See
    ``docs/translate.md``.
    """

    def __init__(self, vocab_path, bpe_code_path, src_lang: str, tgt_lang: str):
        meta = _load_json(vocab_path)
        self.source_vocab: List[str] = meta["source_vocab"]
        self.target_vocab: List[str] = meta["target_vocab"]
        self.source_offset: int = int(meta["source_offset"])
        self.source_index = {token: i for i, token in enumerate(self.source_vocab)}
        self.unk_id = int(meta.get("unk", 0))
        self.pad_id = int(meta.get("pad", 1))
        self.bos_id = int(meta.get("bos", 2))
        self.eos_id = int(meta.get("eos", 3))
        # `<unk>` is deliberately absent: it is a real, informative output.
        self._specials = {self.pad_id, self.bos_id, self.eos_id}

        sacremoses = _require_opennmt("sacremoses")
        apply_bpe = _require_opennmt("subword_nmt.apply_bpe")
        with open(bpe_code_path, encoding="utf-8") as handle:
            # The merge table alone is not the segmentation the model was
            # trained on. `subword-nmt` is a two-argument tool: the merges say
            # *how* to join, the `--vocabulary` says *how far*. Given a
            # vocabulary, `apply_bpe` re-splits any segment the vocabulary does
            # not contain until every piece is one the model has an embedding
            # for; given none, it applies every merge that fits and happily
            # produces a segment the vocabulary never saw.
            #
            # OpenNMT-py's Nós pipeline builds the vocabulary *from* the
            # vocabulary-constrained output, so the two only agree when the
            # constraint is applied on the way in too. Without it, the encoder
            # is fed `<unk>` for a word the model knows perfectly well:
            # `duerme` merged to `duer@@ me`, and `duer@@` is not in
            # `nos-mt-es-arg`'s source vocabulary, while `du@@ er@@ me` is.
            # Nothing raises - the id is a valid `<unk>` - and the damage is
            # amplified on the way out, where one unknown source word costs
            # several unknown target words.
            self.bpe = apply_bpe.BPE(handle, vocab=set(self.source_vocab))
        self.moses_tokenizer = sacremoses.MosesTokenizer(lang=src_lang)
        self.moses_detokenizer = sacremoses.MosesDetokenizer(lang=tgt_lang)

    def encode(self, text: str) -> List[int]:
        """Moses-tokenise, BPE-segment, look up, offset. No EOS is appended.

        OpenNMT-py does not put ``</s>`` on the source side, and the export
        preserves that, so neither does this.
        """
        tokens = self.bpe.process_line(
            " ".join(self.moses_tokenizer.tokenize(text, escape=False))).split()
        return [self.source_index.get(t, self.unk_id) + self.source_offset
                for t in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        """Ids -> text: drop BPE continuation markers, then Moses-detokenise.

        The merge marker is stripped with ``@\\s*`` rather than
        ``replace("@@ ", "")``. That is the upstream ``sed 's/@\\s*//g'`` rule,
        and the two are *not* equivalent: a word-final ``@@`` before
        punctuation loses its marker under the naive form and glues two words
        together.

        An id with no entry in ``target_vocab`` decodes to ``<unk>``, never to
        nothing. Most exports size the embedding table to the target vocabulary
        exactly, but ``nos-coda_iacobus-es-pt`` pads it to 32768 against a
        27968-entry vocabulary, leaving 4800 rows that carry logits and name no
        token. Dropping those - as this did - deletes a word from the middle of
        a sentence and leaves fluent, complete-looking text behind, which is
        the one failure mode a reader cannot see. ``<unk>`` is already this
        class's word for "the model emitted something it cannot spell", and it
        is visible.
        """
        pieces = [self.target_vocab[i] if 0 <= i < len(self.target_vocab)
                  else "<unk>"
                  for i in ids if i not in self._specials]
        merged = re.sub(r"@\s*", "", " ".join(pieces))
        return self.moses_detokenizer.detokenize(merged.split())


def _require_opennmt(module: str):
    """Import an OpenNMT preprocessing dependency, or name the extra."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"OpenNMT-BPE preprocessing needs {module!r}, which is not "
            f"installed. Install the extra: "
            f"pip install 'linguonnx[opennmt]'") from exc


def _marian_special_tokens(special_tokens_map_path: Optional[Path]) -> Dict[str, str]:
    """Read ``eos_token``/``pad_token``/``unk_token`` off a Marian repo's own
    ``special_tokens_map.json``, if it shipped one.

    Standard opus-mt exports spell them as plain strings
    (``{"pad_token": "<pad>"}``); HF's ``tokenizers``-library exports (the
    ``aina-translator-*`` family) wrap each in an object
    (``{"pad_token": {"content": "<blank>", ...}}``). Both shapes are read;
    an absent file, or a value that is neither shape, falls back to
    :class:`MarianTokenizer`'s opus-mt defaults so nothing changes for the
    repos already relying on them.
    """
    if special_tokens_map_path is None:
        return {}
    try:
        raw = _load_json(special_tokens_map_path)
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, str] = {}
    for key in ("eos_token", "pad_token", "unk_token"):
        value = raw.get(key)
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            out[key] = value["content"]
    return out


def load_tokenizer(arch: str, files: Dict[str, Path], lang_codes: Sequence[str],
                   pair: Optional[Sequence[str]] = None):
    """Build the right tokenizer for ``arch`` from the downloaded ``files``."""
    if arch == "marian":
        specials = _marian_special_tokens(files.get("special_tokens_map"))
        return MarianTokenizer(files["source_spm"], files["target_spm"], files["vocab"],
                               **specials)
    if arch == "m2m100":
        codes = lang_codes or artifact_lang_codes(files)
        return SpmSeq2SeqTokenizer(files["spm"], codes, vocab_path=files["vocab"],
                                   added_tokens_path=files.get("added_tokens"))
    if arch == "nllb":
        codes = lang_codes or artifact_lang_codes(files)
        return SpmSeq2SeqTokenizer(files["spm"], codes, fairseq_offset=1)
    if arch == "madlad":
        return T5SpmTokenizer(files["spm"])
    if arch == "t5-prefix":
        config = _load_json(files["config"])
        if "spm" not in files:
            # No SentencePiece model in the export at all - the vocabulary is
            # only in `tokenizer.json`. See `UnigramTextPrefixTokenizer`.
            return UnigramTextPrefixTokenizer(
                files["tokenizer_json"],
                eos_id=config["eos_token_id"], pad_id=config["pad_token_id"])
        tokenizer_config = _load_json(files["tokenizer_config"])
        return T5TextPrefixTokenizer(
            files["spm"], files["added_tokens"],
            eos_id=config["eos_token_id"], pad_id=config["pad_token_id"],
            legacy=tokenizer_config.get("legacy", True),
            unk_piece=tokenizer_config.get("unk_token") or "<unk>")
    if arch == "indictrans2":
        return IndicTransTokenizer(files["spm_src"], files["spm_tgt"],
                                   files["dict_src"], files["dict_tgt"])
    if arch == "opennmt-bpe":
        if not pair:
            raise ValueError("opennmt-bpe models are bilingual; `pair` is required")
        return OpenNmtBpeTokenizer(files["vocab"], files["bpe_code"],
                                   src_lang=pair[0], tgt_lang=pair[1])
    if arch == "pegasus-fast":
        specials = _marian_special_tokens(files.get("special_tokens_map"))
        return FastUnigramTokenizer(files["tokenizer_json"], **specials)
    raise ValueError(f"unknown architecture {arch!r}")
