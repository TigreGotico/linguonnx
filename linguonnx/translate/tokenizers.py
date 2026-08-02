"""Tokenizers for the translation models, with no `transformers` dependency.

Everything here is `sentencepiece` plus a JSON vocabulary. The three
architectures differ in exactly two places - how a SentencePiece piece becomes
an id, and what wraps the sentence - so both live in one class per family:

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
from typing import Dict, List, Optional, Sequence

import sentencepiece as spm

__all__ = ["normalize_punctuation", "SpmSeq2SeqTokenizer", "MarianTokenizer",
           "load_tokenizer"]


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


def _load_spm(path) -> "spm.SentencePieceProcessor":
    processor = spm.SentencePieceProcessor()
    processor.Load(str(path))
    return processor


# --------------------------------------------------------------------------
# M2M100 / NLLB
# --------------------------------------------------------------------------

class SpmSeq2SeqTokenizer:
    """M2M100 and NLLB.

    ``lang_codes`` are the model's own codes (``pt``, or ``por_Latn``), in the
    order they appear in ``special_tokens_map.json``. That order *is* the id
    order of the language-token block, so the ids are derived from it rather
    than hardcoded.
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
            # Authoritative when present (M2M100 ships one).
            added = _load_json(added_tokens_path)
            self.lang_code_to_id = {code: int(added[code])
                                    for code in lang_codes if code in added}
            if len(self.lang_code_to_id) != len(lang_codes):
                self.lang_code_to_id = {code: lang_block_start + i
                                        for i, code in enumerate(lang_codes)}
        else:
            self.lang_code_to_id = {code: lang_block_start + i
                                    for i, code in enumerate(lang_codes)}
        self.id_to_lang_code = {i: c for c, i in self.lang_code_to_id.items()}
        self._specials = {self.eos_id, self.pad_id, self.unk_id, self.bos_id}
        self._specials |= set(self.id_to_lang_code)

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
            raise KeyError(f"language code {code!r} is not supported by this model")
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


def load_tokenizer(arch: str, files: Dict[str, Path], lang_codes: Sequence[str]):
    """Build the right tokenizer for ``arch`` from the downloaded ``files``."""
    if arch == "marian":
        return MarianTokenizer(files["source_spm"], files["target_spm"], files["vocab"])
    if arch == "m2m100":
        return SpmSeq2SeqTokenizer(files["spm"], lang_codes, vocab_path=files["vocab"],
                                   added_tokens_path=files.get("added_tokens"))
    if arch == "nllb":
        return SpmSeq2SeqTokenizer(files["spm"], lang_codes, fairseq_offset=1)
    raise ValueError(f"unknown architecture {arch!r}")
