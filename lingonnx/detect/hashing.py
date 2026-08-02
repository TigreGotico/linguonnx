"""fastText feature hashing, vendored from TigreGotico/glotlid-onnx's
``glotlid_hash.py`` (https://huggingface.co/TigreGotico/glotlid-onnx).

This is the reference implementation the ONNX graph was designed against —
kept logically identical on purpose, including the sign-extension of bytes
>= 0x80 before the FNV-1a XOR. That sign-extension is not a bug: fastText's
C++ ``Dictionary::hash`` treats each byte as a signed ``int8_t`` before
XOR-ing it into the hash, so non-ASCII n-grams must be hashed the same way
here or every non-Latin-script bucket lookup breaks.

Do not "clean up" the sign-extension or the two near-identical add_subwords /
subwords_of_known code paths — they mirror fastText's Dictionary::addSubwords
and Dictionary::getSubwords respectively, which really are separate methods
with separate cases (in-vocab word vs. unseen token, EOS handling, etc.).
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

import numpy as np

EOS = "</s>"
BOW = "<"
EOW = ">"
# fastText treats these as token separators (Dictionary::readWord)
SEPARATORS = " \t\n\v\f\r"


def fnv1a(s: str) -> int:
    """fastText's `Dictionary::hash` - FNV-1a over the UTF-8 bytes.

    fastText casts each byte to `int8_t` before the XOR, so bytes >= 0x80 are
    sign-extended to 32 bits. Missing this makes every non-ASCII n-gram hash
    to the wrong bucket.
    """
    h = 2166136261
    for b in s.encode("utf-8"):
        if b >= 0x80:
            b |= 0xFFFFFF00
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def tokenize(text: str) -> List[str]:
    """Whitespace tokenization, then fastText's implicit end-of-line EOS."""
    for sep in SEPARATORS[1:]:
        text = text.replace(sep, " ")
    return [t for t in text.split(" ") if t] + [EOS]


def compute_subwords(word: str, minn: int, maxn: int, bucket: int,
                      nwords: int) -> List[int]:
    """fastText's `Dictionary::computeSubwords` on the BOW/EOW-wrapped word.

    Character n-grams are counted in *codepoints*, but fastText walks the raw
    byte string and skips UTF-8 continuation bytes, so the slicing below over
    a list of characters is equivalent.
    """
    chars = list(word)
    out: List[int] = []
    n = len(chars)
    for i in range(n):
        for j in range(i + minn, min(n, i + maxn) + 1):
            ngram = "".join(chars[i:j])
            out.append(nwords + fnv1a(ngram) % bucket)
    return out


class GlotLIDFeaturizer:
    """Maps raw text to the fastText feature ids the ONNX graph expects.

    Shared by every model in the registry - GlotLID, OpenLID, OpenLID-v2 and
    lid.176 all use the same fastText hashing, differing only in the
    ``nwords``/``minn``/``maxn``/``bucket`` values read from their config.
    """

    def __init__(self, words: Sequence[str], nwords: int, minn: int = 2,
                 maxn: int = 5, bucket: int = 1_000_000):
        self.words: List[str] = list(words)
        self.nwords = nwords
        self.minn = minn
        self.maxn = maxn
        self.bucket = bucket
        self.word2id: Dict[str, int] = {w: i for i, w in enumerate(self.words)}
        # Subwords of in-vocabulary words are precomputed by fastText at load
        # time (Dictionary::initNgrams); we compute them lazily and cache.
        self._cache: Dict[int, List[int]] = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def from_files(cls, vocab_path: str, meta_path: str) -> "GlotLIDFeaturizer":
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(vocab_path, encoding="utf-8") as fh:
            words = fh.read().split("\n")
        if words and words[-1] == "":
            words.pop()
        return cls(words, meta["nwords"], meta["minn"], meta["maxn"],
                    meta["bucket"])

    # -- fastText internals ----------------------------------------------
    def subwords_of_known(self, wid: int) -> List[int]:
        """`Dictionary::getSubwords(wid)` - the word id itself plus n-grams."""
        cached = self._cache.get(wid)
        if cached is None:
            word = self.words[wid]
            if word == EOS:
                cached = [wid]
            else:
                cached = [wid] + compute_subwords(
                    BOW + word + EOW, self.minn, self.maxn, self.bucket,
                    self.nwords)
            self._cache[wid] = cached
        return cached

    def add_subwords(self, line: List[int], token: str) -> None:
        """`Dictionary::addSubwords`."""
        wid = self.word2id.get(token, -1)
        if wid < 0:
            if token != EOS:
                line.extend(compute_subwords(BOW + token + EOW, self.minn,
                                              self.maxn, self.bucket,
                                              self.nwords))
        elif self.maxn <= 0:
            line.append(wid)
        else:
            line.extend(self.subwords_of_known(wid))

    def __call__(self, text: str) -> np.ndarray:
        """`Dictionary::getLine` for a supervised model - returns feature ids."""
        line: List[int] = []
        for token in tokenize(text):
            self.add_subwords(line, token)
        return np.asarray(line, dtype=np.int64)


# The hashing is fastText's, not GlotLID's; the original name is kept as the
# public one for backward compatibility.
FastTextFeaturizer = GlotLIDFeaturizer
