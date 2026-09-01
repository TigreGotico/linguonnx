"""Input bounds, and the errors raised when they are crossed.

Every entry point in linguonnx can be reached by untrusted text - the detect
and translate helpers are what a server puts behind an HTTP handler - and none
of the work they do is cheap in the size of that text. Feature hashing is a
pure-Python loop that holds the GIL, encoder self-attention is quadratic, and
beam search re-gathers the whole cross-attention cache once per generated
token. A single large request therefore slows down every other request in the
process, not only its own.

So the bounds live here, in one module, with an environment variable each: an
operator can tune them without patching the library. The defaults are chosen
to sit above anything a legitimate caller sends and well below the point where
one request can hurt the process.

Crossing a bound raises. Truncating instead would give a plausible answer
computed from part of the input, which is the failure mode that is hardest to
notice in production.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "InputTooLongError",
    "EmptyInputError",
    "DecodeError",
    "CorruptModelFileError",
    "MAX_DETECT_CHARS",
    "MAX_LINE_TOKENS",
    "MAX_ENCODER_TOKENS",
    "MAX_NUM_BEAMS",
    "MAX_LENGTH_PENALTY",
    "MAX_MODEL_MB",
    "has_visible_content",
    "check_length",
]


class InputTooLongError(ValueError):
    """Input crossed a configured size bound."""


class EmptyInputError(ValueError):
    """Input has no visible content, so there is nothing to work on."""


class DecodeError(RuntimeError):
    """Generation produced no usable sequence."""


class CorruptModelFileError(ValueError):
    """A downloaded side file does not match the model it belongs to."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _env_optional_int(name: str) -> "int | None":
    """An integer bound that is simply absent when the variable is unset.

    Distinct from :func:`_env_int`, which always has a default: some bounds
    have no sensible number to fall back to, and "unset" has to stay
    distinguishable from "set to something large".
    """
    if os.environ.get(name) is None:
        return None
    return _env_int(name, 0)


#: Characters accepted by :class:`~linguonnx.detect.LanguageDetector`.
#: Language identification accuracy saturates after a few hundred characters,
#: so 10,000 is already generous; it is here only to keep the pure-Python
#: hashing loop short.
MAX_DETECT_CHARS = _env_int("LINGUONNX_MAX_DETECT_CHARS", 10_000)

#: Tokens read from one line of text before the featurizer stops.
#:
#: The value is fastText's ``Dictionary::MAX_LINE_SIZE``, but be careful about
#: what that buys: in fastText 0.9.2 the cap sits in the *unsupervised*
#: ``getLine`` overload only, and the supervised overload these LID models use
#: reads to the end of the line. So this is linguonnx's own bound, set at the
#: reference's own idea of a long line, and not a parity fix. It is tunable
#: for an operator who wants byte-for-byte fastText behaviour on huge lines.
MAX_LINE_TOKENS = _env_int("LINGUONNX_MAX_LINE_TOKENS", 1024)

#: Tokens accepted by the translation encoder. 1024 is the largest positional
#: table among the registered architectures (M2M100); Marian stops at 512, so
#: this bound never rejects text a registered model could have handled.
MAX_ENCODER_TOKENS = _env_int("LINGUONNX_MAX_ENCODER_TOKENS", 1024)

#: Beams accepted by beam search. Every beam is a full row of the cross
#: attention cache, re-gathered each step, so cost grows linearly with this
#: number while translation quality stops improving well below 32.
MAX_NUM_BEAMS = _env_int("LINGUONNX_MAX_NUM_BEAMS", 32)

#: Largest single model, in MB, that routing may put on a route. ``None``, the
#: default, is no bound at all.
#:
#: Unlike every other limit here this one is a *routing* bound, not an input
#: bound. It does not truncate or refuse work: the router answers the same
#: question with a smaller model set, and a pair that needed one 4.9 GB model
#: is served by a chain of small ones instead. Set it on a host that cannot
#: afford the download - a Pi, a metered link - and coverage stays, latency
#: grows. See ``docs/routing.md``.
MAX_MODEL_MB = _env_optional_int("LINGUONNX_MAX_MODEL_MB")


def operator_budget_is_set() -> bool:
    """Whether a *human* set a size budget, as opposed to the library.

    ``max_model_mb=UNSET`` is not "the default": it reads
    ``LINGUONNX_MAX_MODEL_MB``, and then the cold-download budget
    (``LINGUONNX_MAX_DOWNLOAD_MB``, itself defaulting to
    ``DEFAULT_MAX_DOWNLOAD_MB``). Only the last of those three is the
    library's own opinion, and only the library's own opinion may be waived
    by a caller who names a model by hand - a budget set on a metered link or
    a small-disk device belongs to the operator and outranks anything the API
    infers.

    Read from the environment on every call, not from :data:`MAX_MODEL_MB`,
    which froze at import time.
    """
    return any(os.environ.get(name) is not None for name in
               ("LINGUONNX_MAX_MODEL_MB", "LINGUONNX_MAX_DOWNLOAD_MB")) \
        or MAX_MODEL_MB is not None

#: Upper bound on ``length_penalty``. Beyond this the length term dominates
#: the model score outright and beam ranking stops depending on the model.
MAX_LENGTH_PENALTY = 10.0


# Format characters that occupy no width and that ``str.strip()`` keeps:
# zero-width space/non-joiner/joiner, word joiner, BOM, and the bidi marks.
# Text pasted out of HTML or a PDF carries these routinely, and a string made
# only of them looks empty to a reader while reaching full tokenisation.
_INVISIBLE = re.compile(r"[​-‏ -‮⁠-⁤﻿]")


def has_visible_content(text: str) -> bool:
    """True if ``text`` holds anything a reader would see.

    ``str.strip()`` alone is not enough: it uses Python's whitespace
    definition, which does not include the zero-width and bidi format
    characters.
    """
    return bool(_INVISIBLE.sub("", text).strip())


def check_length(size: int, limit: int, unit: str, knob: str) -> None:
    """Raise :class:`InputTooLongError` if ``size`` is over ``limit``."""
    if size > limit:
        raise InputTooLongError(
            f"input is {size} {unit}, over the limit of {limit}; "
            f"split the input or raise {knob}")
