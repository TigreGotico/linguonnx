"""Encoder-decoder generation on raw ONNX graphs, with numpy only.

`optimum` and `transformers.generate()` would do this, but both import torch,
and the point of linguonnx is `onnxruntime` + `numpy` + `sentencepiece`. So the
loop is written out here.

The three graphs
----------------

``encoder_model.onnx``
    ``input_ids, attention_mask -> last_hidden_state``. Runs once.

``decoder_model.onnx`` (step 0, no cache)
    ``input_ids, encoder_hidden_states, encoder_attention_mask -> logits`` plus
    a full set of ``present.*`` key/value tensors, self-attention *and*
    cross-attention.

``decoder_with_past_model.onnx`` (every later step)
    Takes one token, plus every cached key/value as ``past_key_values.*``, and
    returns ``logits`` plus new ``present.*`` for the self-attention entries
    only. The cross-attention entries do not change after step 0 - the encoder
    output they were computed from is fixed - so they are threaded straight
    through from the step-0 outputs.

Cache wiring is done **by name**, not by position: every
``past_key_values.<i>.<attn>.<kv>`` input is matched to the graph output of the
same name with ``past_key_values`` swapped for ``present``. Nothing assumes a
layer count, an attention-head layout, or that the two decoder graphs order
their tensors the same way. If a graph ever turns up with a naming scheme this
does not cover, :class:`Seq2SeqDecoder` raises at load time instead of
producing silent garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from linguonnx.limits import (MAX_ENCODER_TOKENS, MAX_LENGTH_PENALTY,
                              MAX_NUM_BEAMS, DecodeError, check_length)

__all__ = ["Seq2SeqDecoder", "GenerationConfig"]


@dataclass
class GenerationConfig:
    """Decoding knobs. ``num_beams=1`` means greedy.

    The values are validated on construction. They are caller-settable all the
    way from ``Translator(...)``, several of them cost memory linearly or
    worse, and two of them have a range where the decoder still returns
    something - just not a translation. An error at the boundary is the only
    place the caller can act on it.
    """

    max_new_tokens: int = 128
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    #: Stop as soon as ``num_beams`` hypotheses have finished. `transformers`
    #: defaults this off and so does linguonnx, because the cheaper rule ends
    #: the search while a better hypothesis is still growing - it costs about
    #: one sentence in ten against a `transformers` reference. Set it to
    #: ``True`` when speed matters more than matching.
    early_stopping: bool = False
    #: Token ids that must never be generated as content, sourced from the
    #: model's own ``generation_config.json`` (``bad_words_ids``, single-token
    #: entries only). Some Marian checkpoints rank their own
    #: ``decoder_start_token_id``/``pad_token_id`` above every real word at
    #: generation step 1 - a known degenerate mode the upstream config already
    #: names by banning that id - but nothing before this field enforced the
    #: ban. Unenforced, the decoder emits that id for the whole budget, the
    #: tokenizer strips it as a special token on the way out, and the caller
    #: gets "" back for a perfectly good input. See ``linguonnx#42``.
    banned_token_ids: FrozenSet[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be a positive int, got "
                f"{self.max_new_tokens!r}")
        if not isinstance(self.num_beams, int) or self.num_beams < 1:
            raise ValueError(
                f"num_beams must be a positive int, got {self.num_beams!r}")
        if self.num_beams > MAX_NUM_BEAMS:
            # Beams past the number of finite candidates are padded with PAD
            # and contribute nothing, but they are still full rows of the
            # cross-attention cache, re-gathered on every step.
            raise ValueError(
                f"num_beams={self.num_beams} is over the limit of "
                f"{MAX_NUM_BEAMS}; beam search stops paying for itself long "
                f"before this, and every beam costs a full copy of the "
                f"attention cache (raise LINGUONNX_MAX_NUM_BEAMS to allow it)")
        if not np.isfinite(self.length_penalty):
            raise ValueError(
                f"length_penalty must be finite, got {self.length_penalty!r}")
        if not 0.0 <= self.length_penalty <= MAX_LENGTH_PENALTY:
            # Beam scores are log probabilities, so they are negative, and the
            # score is divided by length**length_penalty. A negative exponent
            # therefore *multiplies* the penalty term by the length and makes
            # longer sequences rank higher - including a beam padded out with
            # PAD tokens, which is the worst hypothesis in the set.
            raise ValueError(
                f"length_penalty must be between 0.0 and "
                f"{MAX_LENGTH_PENALTY}, got {self.length_penalty}; a negative "
                f"value inverts beam ranking and can hand back the worst "
                f"hypothesis")
        if not isinstance(self.no_repeat_ngram_size, int) \
                or self.no_repeat_ngram_size < 0:
            raise ValueError(
                f"no_repeat_ngram_size must be a non-negative int, got "
                f"{self.no_repeat_ngram_size!r}")
        if self.no_repeat_ngram_size == 1:
            # Size 1 has no useful reading. Here the n-gram prefix is empty, so
            # every token ever emitted is banned from recurring and the output
            # collapses; in transformers the lookup key is the whole sequence,
            # which never matches, so the setting silently does nothing. Two
            # opposite wrong answers, neither of them what a caller wants.
            raise ValueError(
                "no_repeat_ngram_size=1 would ban every token from appearing "
                "twice, which no real translation survives; use 0 to disable "
                "the guard or 2 or more to block repeated phrases")
        # Accept any iterable (a model passes a plain list); freeze it once
        # here so every later lookup is a fast set membership test instead of
        # a per-step scan.
        self.banned_token_ids = frozenset(int(t) for t in self.banned_token_ids)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _banned_ngram_tokens(sequence: Sequence[int], ngram_size: int) -> List[int]:
    """Tokens that would close a repeat of an n-gram already in ``sequence``."""
    if ngram_size <= 0 or len(sequence) < ngram_size:
        return []
    prefix = tuple(sequence[-(ngram_size - 1):]) if ngram_size > 1 else ()
    banned = []
    for i in range(len(sequence) - ngram_size + 1):
        window = tuple(sequence[i:i + ngram_size])
        if window[:-1] == prefix:
            banned.append(window[-1])
    return banned


class Seq2SeqDecoder:
    """Greedy and beam-search generation over three ONNX Runtime sessions."""

    def __init__(self, encoder_session, decoder_session, decoder_past_session,
                 eos_id: int, pad_id: int, decoder_start_id: int,
                 max_input_tokens: int = MAX_ENCODER_TOKENS):
        self.max_input_tokens = max_input_tokens
        self.encoder = encoder_session
        self.decoder = decoder_session
        self.decoder_past = decoder_past_session
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.decoder_start_id = decoder_start_id

        self._enc_inputs = [i.name for i in self.encoder.get_inputs()]
        self._enc_output = self.encoder.get_outputs()[0].name
        self._dec_inputs = [i.name for i in self.decoder.get_inputs()]
        self._dec_outputs = [o.name for o in self.decoder.get_outputs()]
        self._past_inputs = [i.name for i in self.decoder_past.get_inputs()]
        self._past_outputs = [o.name for o in self.decoder_past.get_outputs()]

        self._logits_name = self._pick(self._dec_outputs, "logits")
        self._hidden_input = self._pick(self._dec_inputs, "encoder_hidden_states")
        self._enc_mask_input = self._pick(self._dec_inputs, "encoder_attention_mask")
        self._dec_ids_input = self._pick(self._dec_inputs, "input_ids")
        self._past_ids_input = self._pick(self._past_inputs, "input_ids")
        self._past_mask_input = self._pick(self._past_inputs, "encoder_attention_mask")

        # past_key_values.<i>.<attn>.<kv>  <->  present.<i>.<attn>.<kv>
        self._cache_names = [n for n in self._past_inputs
                             if n.startswith("past_key_values.")]
        self._present_for: Dict[str, str] = {}
        produced = set(self._dec_outputs) | set(self._past_outputs)
        for name in self._cache_names:
            present = name.replace("past_key_values.", "present.", 1)
            if present not in produced:
                raise ValueError(
                    f"decoder cache input {name!r} has no matching graph output "
                    f"{present!r}; this decoder uses a key/value naming scheme "
                    f"linguonnx does not know how to wire")
            self._present_for[name] = present
        # Only these are refreshed each step; the rest are cross-attention and
        # stay at their step-0 value.
        self._refreshed = {name for name, present in self._present_for.items()
                           if present in self._past_outputs}

    @staticmethod
    def _pick(names: Sequence[str], wanted: str) -> str:
        if wanted in names:
            return wanted
        matches = [n for n in names if n.endswith(wanted)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"cannot find input/output {wanted!r} among {list(names)}")

    # -- graph calls ------------------------------------------------------

    def _encode(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Self-attention is quadratic in the source length, and beam search
        # then re-gathers the cross-attention cache - which is also linear in
        # that length - once per generated token. Long inputs are therefore
        # paid for twice over, and past the model's positional table the
        # result is not a translation anyway.
        check_length(input_ids.shape[-1], self.max_input_tokens, "tokens",
                     "LINGUONNX_MAX_ENCODER_TOKENS")
        mask = np.ones_like(input_ids, dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": mask}
        feeds = {name: feeds[name] for name in self._enc_inputs}
        hidden = self.encoder.run([self._enc_output], feeds)[0]
        return hidden, mask

    def _first_step(self, hidden: np.ndarray, mask: np.ndarray,
                    token: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feeds = {
            self._dec_ids_input: np.array([[token]], dtype=np.int64),
            self._hidden_input: hidden,
            self._enc_mask_input: mask,
        }
        outputs = self.decoder.run(self._dec_outputs, feeds)
        named = dict(zip(self._dec_outputs, outputs))
        cache = {name: named[self._present_for[name]] for name in self._cache_names}
        return named[self._logits_name][:, -1, :], cache

    def _step(self, tokens: np.ndarray, mask: np.ndarray,
              cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feeds = dict(cache)
        feeds[self._past_ids_input] = tokens
        feeds[self._past_mask_input] = mask
        outputs = self.decoder_past.run(self._past_outputs, feeds)
        named = dict(zip(self._past_outputs, outputs))
        new_cache = dict(cache)
        for name in self._refreshed:
            new_cache[name] = named[self._present_for[name]]
        return named[self._logits_name][:, -1, :], new_cache

    @staticmethod
    def _expand(array: np.ndarray, n: int) -> np.ndarray:
        return np.repeat(array, n, axis=0)

    def _reorder(self, cache: Dict[str, np.ndarray],
                 index: np.ndarray) -> Dict[str, np.ndarray]:
        """Beam search's one genuinely fiddly step: keep the cache aligned.

        After the top-k selection each surviving beam may descend from *any*
        previous beam, so every cached tensor's batch dimension is gathered by
        the parent-beam index. Cross-attention entries are reordered too - they
        are per-beam rows even though their content is constant per input.
        """
        return {name: value[index] for name, value in cache.items()}

    # -- generation -------------------------------------------------------

    def generate(self, input_ids: Sequence[int],
                 forced_bos_token_id: Optional[int] = None,
                 config: Optional[GenerationConfig] = None) -> List[int]:
        config = config or GenerationConfig()
        ids = np.asarray([list(input_ids)], dtype=np.int64)
        if config.num_beams <= 1:
            return self._greedy(ids, forced_bos_token_id, config)
        return self._beam(ids, forced_bos_token_id, config)

    def _greedy(self, input_ids: np.ndarray, forced_bos: Optional[int],
                config: GenerationConfig) -> List[int]:
        hidden, mask = self._encode(input_ids)
        logits, cache = self._first_step(hidden, mask, self.decoder_start_id)

        generated: List[int] = []
        for step in range(config.max_new_tokens):
            if step == 0 and forced_bos is not None:
                token = forced_bos
            else:
                scores = logits[0].astype(np.float64)
                for banned in config.banned_token_ids:
                    scores[banned] = -np.inf
                for banned in _banned_ngram_tokens(generated, config.no_repeat_ngram_size):
                    scores[banned] = -np.inf
                token = int(np.argmax(scores))
            if token == self.eos_id:
                break
            generated.append(token)
            if step == config.max_new_tokens - 1:
                break
            logits, cache = self._step(
                np.array([[token]], dtype=np.int64), mask, cache)
        if not generated:
            # The first sampled token was EOS. Empty input never gets this
            # far, so an empty generation is a failure, and the caller has to
            # be able to tell it from the empty string it asks for.
            raise DecodeError(
                "greedy decoding ended before emitting a single token; "
                "no output can be produced for this input")
        return generated

    def _hypothesis_score(self, sequence: Sequence[int], total_logprob: float,
                          length_penalty: float) -> float:
        """Length-normalised score of a finished hypothesis.

        The divisor counts the **decoder start token as well**, which is not a
        detail: `transformers` normalises by ``decoder_input_ids.shape[-1]``,
        and that tensor begins with ``decoder_start_token_id``. Dividing by the
        generated length alone is off by one, and the error does not cancel
        between hypotheses of different lengths - it re-ranks them. Matching
        this is worth roughly a third of the sentences on a beam-4 comparison
        against `transformers`.
        """
        return total_logprob / (len(sequence) + 1) ** length_penalty

    def _beam(self, input_ids: np.ndarray, forced_bos: Optional[int],
              config: GenerationConfig) -> List[int]:
        """Beam search, written to agree with `transformers` token for token.

        The published parity numbers for every model in the registry were
        measured against `transformers.generate()`, so the selection rules here
        follow ``BeamSearchScorer`` rather than a textbook beam search. Three of
        them are load-bearing and none of them fails loudly when broken - they
        just return a slightly different, plausible sentence:

        * the length normaliser counts the decoder start token
          (:meth:`_hypothesis_score`);
        * an EOS candidate ranked at or below ``num_beams`` is **discarded**,
          not finished, because a hypothesis that bad would never win;
        * search stops on the "cannot be beaten" test, not on "``num_beams``
          hypotheses exist". ``early_stopping=True`` restores the cheaper,
          slightly worse rule.
        """
        beams = config.num_beams
        length_penalty = config.length_penalty
        hidden, mask = self._encode(input_ids)
        logits, cache = self._first_step(hidden, mask, self.decoder_start_id)

        # Expand the single step-0 result into `beams` identical rows. Only the
        # first beam starts alive: with identical rows, an all-zero score vector
        # would make the top-k pick the same token `beams` times.
        cache = {name: self._expand(value, beams) for name, value in cache.items()}
        mask = self._expand(mask, beams)
        logits = self._expand(logits, beams)

        scores = np.full(beams, -np.inf, dtype=np.float64)
        scores[0] = 0.0
        sequences: List[List[int]] = [[] for _ in range(beams)]
        # Best `beams` finished hypotheses, worst first is not maintained; the
        # list is trimmed instead, which is cheap at these sizes.
        finished: List[Tuple[float, List[int]]] = []

        def remember(sequence: List[int], total: float) -> None:
            finished.append((self._hypothesis_score(sequence, total, length_penalty),
                             list(sequence)))
            if len(finished) > beams:
                finished.sort(key=lambda item: item[0], reverse=True)
                del finished[beams:]

        for step in range(config.max_new_tokens):
            logprobs = _log_softmax(logits.astype(np.float64))
            if step == 0 and forced_bos is not None:
                forced = np.full_like(logprobs, -np.inf)
                forced[:, forced_bos] = 0.0
                logprobs = forced
            for banned in config.banned_token_ids:
                logprobs[:, banned] = -np.inf
            if config.no_repeat_ngram_size:
                for beam, sequence in enumerate(sequences):
                    for banned in _banned_ngram_tokens(sequence, config.no_repeat_ngram_size):
                        logprobs[beam, banned] = -np.inf

            total = scores[:, None] + logprobs
            flat = total.ravel()
            # 2*beams candidates so that `beams` survivors remain even if every
            # other candidate ends the sentence on this step.
            take = min(2 * beams, flat.size)
            top = np.argpartition(-flat, take - 1)[:take]
            top = top[np.argsort(-flat[top])]

            next_scores, next_tokens, next_parents, next_seqs = [], [], [], []
            vocab = logprobs.shape[1]
            for rank, index in enumerate(top):
                parent, token = int(index // vocab), int(index % vocab)
                score = float(flat[index])
                if not np.isfinite(score):
                    continue
                sequence = sequences[parent]
                if token == self.eos_id:
                    # Ranked below the beam width: `transformers` drops it.
                    if rank < beams and sequence:
                        remember(sequence, score)
                else:
                    next_scores.append(score)
                    next_tokens.append(token)
                    next_parents.append(parent)
                    next_seqs.append(sequence + [token])
                if len(next_tokens) == beams:
                    break

            if not next_tokens:
                break

            done = False
            if len(finished) >= beams:
                if config.early_stopping:
                    done = True
                else:
                    # Nothing still running can beat the worst kept hypothesis,
                    # even if it ended on the very next token. `step + 1` is the
                    # decoder length before this step's token is appended, which
                    # is the length `transformers` normalises this bound by.
                    best_attainable = (float(flat[top[0]])
                                       / (step + 1) ** length_penalty)
                    done = min(item[0] for item in finished) >= best_attainable
            if done:
                break

            while len(next_tokens) < beams:  # pad a collapsed beam set
                next_scores.append(-np.inf)
                next_tokens.append(self.pad_id)
                next_parents.append(next_parents[0])
                next_seqs.append(list(next_seqs[0]))

            scores = np.asarray(next_scores, dtype=np.float64)
            sequences = next_seqs
            cache = self._reorder(cache, np.asarray(next_parents, dtype=np.int64))
            if step == config.max_new_tokens - 1:
                break
            logits, cache = self._step(
                np.asarray(next_tokens, dtype=np.int64)[:, None], mask, cache)

        # Out of budget with too few finished hypotheses: the live beams count
        # too, exactly as `transformers` folds them in at ``finalize()``.
        if len(finished) < beams:
            for i in range(beams):
                if sequences[i] and np.isfinite(scores[i]):
                    remember(sequences[i], float(scores[i]))
        if not finished:
            # Every beam was padded to -inf and none reached EOS. Returning []
            # here would decode to "", which is exactly what the caller layer
            # returns for empty input - so a real decode failure would arrive
            # indistinguishable from "you gave me nothing".
            raise DecodeError(
                f"beam search finished no hypothesis and every beam collapsed "
                f"after {config.max_new_tokens} steps; no output can be "
                f"produced for this input")
        return max(finished, key=lambda item: item[0])[1]
