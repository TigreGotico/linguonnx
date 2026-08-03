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
           "FastUnigramTokenizer", "T5SpmTokenizer", "IndicTransTokenizer",
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
            self.bpe = apply_bpe.BPE(handle)
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
        """
        pieces = [self.target_vocab[i] for i in ids
                  if i not in self._specials and 0 <= i < len(self.target_vocab)]
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
