"""Decode-loop mechanics against a tiny real ONNX seq2seq.

The graphs built here are a few kilobytes and carry no weights worth the name,
but they are genuine ONNX: three sessions, the same
``past_key_values.*``/``present.*`` naming the exported translation models use,
and a decoder whose logits depend on the **whole** generated history rather
than only the last token. That last property is what makes the beam tests
meaningful - if the KV cache is reordered wrongly after a top-k step, a beam
inherits another beam's history and the output changes.

The toy model is:

    state_t = (decoder_start + sum of tokens generated so far) mod N_STATES
    logits_t = TABLE[state_t]

which the tests mirror in numpy to get a reference answer.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from linguonnx.limits import (MAX_ENCODER_TOKENS, DecodeError,
                              InputTooLongError)
from linguonnx.translate.decode import (GenerationConfig, Seq2SeqDecoder,
                                        _banned_ngram_tokens)

VOCAB = 12
N_STATES = 7
EOS = 2
PAD = 1
START = 3

rng = np.random.default_rng(3)
TABLE = rng.normal(size=(N_STATES, VOCAB)).astype(np.float32)


# --------------------------------------------------------------------------
# the toy graphs
# --------------------------------------------------------------------------

def _consts():
    return [
        numpy_helper.from_array(TABLE, "TABLE"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "AX2"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "AX1"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "AX12"),
        numpy_helper.from_array(np.array(N_STATES, dtype=np.int64), "NSTATES"),
    ]


def _state_to_logits(source: str, prefix: str):
    """nodes turning a (B,1,P) float cache into (B,1,VOCAB) logits."""
    return [
        helper.make_node("ReduceSum", [source, "AX12"], [prefix + "sum"], keepdims=0),
        helper.make_node("Cast", [prefix + "sum"], [prefix + "i"], to=TensorProto.INT64),
        helper.make_node("Mod", [prefix + "i", "NSTATES"], [prefix + "idx"]),
        helper.make_node("Gather", ["TABLE", prefix + "idx"], [prefix + "g"], axis=0),
        helper.make_node("Unsqueeze", [prefix + "g", "AX1"], ["logits"]),
    ]


def _make_encoder(path):
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["f", "AX2"], ["last_hidden_state"]),
    ]
    graph = helper.make_graph(
        nodes, "encoder",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT,
                                       ["b", "s", 1])],
        initializer=_consts())
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def _make_decoder(path):
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["f", "AX2"], ["present.0.decoder.key"]),
        helper.make_node("Identity", ["present.0.decoder.key"], ["present.0.decoder.value"]),
        helper.make_node("ReduceSum", ["encoder_hidden_states", "AX12"],
                         ["present.0.encoder.key"], keepdims=1),
        helper.make_node("Identity", ["present.0.encoder.key"], ["present.0.encoder.value"]),
    ] + _state_to_logits("present.0.decoder.key", "d")
    graph = helper.make_graph(
        nodes, "decoder",
        [helper.make_tensor_value_info("encoder_attention_mask", TensorProto.INT64, ["b", "s"]),
         helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", 1]),
         helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, ["b", "s", 1])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 1, VOCAB]),
         helper.make_tensor_value_info("present.0.decoder.key", TensorProto.FLOAT, ["b", 1, "p"]),
         helper.make_tensor_value_info("present.0.decoder.value", TensorProto.FLOAT, ["b", 1, "p"]),
         helper.make_tensor_value_info("present.0.encoder.key", TensorProto.FLOAT, ["b", 1, 1]),
         helper.make_tensor_value_info("present.0.encoder.value", TensorProto.FLOAT, ["b", 1, 1])],
        initializer=_consts())
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def _make_decoder_with_past(path):
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["f", "AX2"], ["cur"]),
        helper.make_node("Concat", ["past_key_values.0.decoder.key", "cur"],
                         ["present.0.decoder.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.decoder.value", "cur"],
                         ["present.0.decoder.value"], axis=2),
    ] + _state_to_logits("present.0.decoder.key", "d")
    inputs = [
        helper.make_tensor_value_info("encoder_attention_mask", TensorProto.INT64, ["b", "s"]),
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", 1]),
        helper.make_tensor_value_info("past_key_values.0.decoder.key", TensorProto.FLOAT, ["b", 1, "p"]),
        helper.make_tensor_value_info("past_key_values.0.decoder.value", TensorProto.FLOAT, ["b", 1, "p"]),
        helper.make_tensor_value_info("past_key_values.0.encoder.key", TensorProto.FLOAT, ["b", 1, 1]),
        helper.make_tensor_value_info("past_key_values.0.encoder.value", TensorProto.FLOAT, ["b", 1, 1]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 1, VOCAB]),
        helper.make_tensor_value_info("present.0.decoder.key", TensorProto.FLOAT, ["b", 1, "p"]),
        helper.make_tensor_value_info("present.0.decoder.value", TensorProto.FLOAT, ["b", 1, "p"]),
    ]
    graph = helper.make_graph(nodes, "decoder_past", inputs, outputs,
                              initializer=_consts())
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    directory = tmp_path_factory.mktemp("toy")
    paths = [directory / name for name in
             ("enc.onnx", "dec.onnx", "dec_past.onnx")]
    _make_encoder(paths[0])
    _make_decoder(paths[1])
    _make_decoder_with_past(paths[2])
    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in paths]
    return Seq2SeqDecoder(*sessions, eos_id=EOS, pad_id=PAD, decoder_start_id=START)


# --- the same model, in numpy ---------------------------------------------

def _logits_for(history):
    return TABLE[(START + sum(history)) % N_STATES].astype(np.float64)


def _reference_greedy(forced_bos, max_new_tokens):
    generated = []
    for step in range(max_new_tokens):
        token = forced_bos if (step == 0 and forced_bos is not None) \
            else int(np.argmax(_logits_for(generated)))
        if token == EOS:
            break
        generated.append(token)
    return generated


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _reference_beam(forced_bos, beams, max_new_tokens, length_penalty=1.0):
    """A plain, obviously-correct beam search over the same transition rule.

    Deliberately the same algorithm the implementation claims to run, written
    the slow and readable way: expand every live beam over the whole
    vocabulary, keep the top ``2 * beams`` candidates (so ``beams`` survivors
    remain even if every other candidate ends the sentence), then split them
    into finished and live.
    """
    live = [(0.0, [])]
    finished = []
    for step in range(max_new_tokens):
        candidates = []
        for score, history in live:
            logprobs = _log_softmax(_logits_for(history))
            if step == 0 and forced_bos is not None:
                candidates.append((score, history + [forced_bos]))
                continue
            for token in range(VOCAB):
                candidates.append((score + logprobs[token], history + [token]))
        candidates.sort(key=lambda item: -item[0])
        candidates = candidates[:2 * beams]
        live = []
        for score, history in candidates:
            if history[-1] == EOS:
                if len(history) > 1:
                    finished.append((score / (len(history) - 1) ** length_penalty,
                                     history[:-1]))
                continue
            if len(live) < beams:
                live.append((score, history))
        if not live or len(finished) >= beams:
            break
    if finished:
        return max(finished, key=lambda item: item[0])[1]
    return max(live)[1]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

INPUT = [5, 6, 7, EOS]


def test_greedy_matches_the_reference_model(toy):
    out = toy.generate(INPUT, config=GenerationConfig(num_beams=1, max_new_tokens=20))
    assert out == _reference_greedy(None, 20)
    assert out, "the toy model should produce something before EOS"


def test_greedy_respects_max_new_tokens(toy):
    for budget in (1, 3, 5):
        out = toy.generate(INPUT, config=GenerationConfig(num_beams=1,
                                                          max_new_tokens=budget))
        assert len(out) <= budget


def test_greedy_stops_at_eos_and_never_emits_it(toy):
    out = toy.generate(INPUT, config=GenerationConfig(num_beams=1, max_new_tokens=64))
    assert EOS not in out
    assert len(out) < 64, "the toy model must reach EOS on its own"


def test_forced_bos_is_the_first_generated_token_greedy(toy):
    out = toy.generate(INPUT, forced_bos_token_id=9,
                       config=GenerationConfig(num_beams=1, max_new_tokens=10))
    assert out[0] == 9
    assert out == _reference_greedy(9, 10)


def test_beam_matches_the_reference_beam_search(toy):
    """The cache-reordering test: logits depend on the whole history, so a
    mis-ordered KV cache gives a different sequence."""
    for beams in (2, 3, 4):
        config = GenerationConfig(num_beams=beams, max_new_tokens=16)
        assert toy.generate(INPUT, config=config) == \
            _reference_beam(None, beams, 16)


def test_beam_with_forced_bos_matches_the_reference(toy):
    for beams in (2, 4):
        config = GenerationConfig(num_beams=beams, max_new_tokens=12)
        out = toy.generate(INPUT, forced_bos_token_id=8, config=config)
        assert out[0] == 8
        assert out == _reference_beam(8, beams, 12)


def test_beam_respects_max_new_tokens(toy):
    for budget in (1, 2, 5):
        out = toy.generate(INPUT, forced_bos_token_id=8,
                           config=GenerationConfig(num_beams=4,
                                                    max_new_tokens=budget))
        assert len(out) <= budget


def test_beam_never_emits_eos(toy):
    out = toy.generate(INPUT, config=GenerationConfig(num_beams=4, max_new_tokens=32))
    assert EOS not in out


def test_length_penalty_is_applied(toy):
    """A penalty below 1 favours short hypotheses; above 1, long ones."""
    short = toy.generate(INPUT, config=GenerationConfig(
        num_beams=4, max_new_tokens=24, length_penalty=0.2))
    long = toy.generate(INPUT, config=GenerationConfig(
        num_beams=4, max_new_tokens=24, length_penalty=3.0))
    assert len(short) <= len(long)


def test_cache_wiring_is_by_name_not_position(toy):
    """Every past_key_values.* input is paired with a present.* output, and only
    the self-attention half is refreshed each step."""
    assert set(toy._cache_names) == {
        "past_key_values.0.decoder.key", "past_key_values.0.decoder.value",
        "past_key_values.0.encoder.key", "past_key_values.0.encoder.value"}
    assert toy._refreshed == {"past_key_values.0.decoder.key",
                              "past_key_values.0.decoder.value"}


def test_unwireable_cache_raises_at_load_time(tmp_path):
    """A graph whose cache names do not pair up must fail loudly, not silently."""
    enc, dec = tmp_path / "e.onnx", tmp_path / "d.onnx"
    _make_encoder(enc)
    _make_decoder(dec)
    broken = tmp_path / "p.onnx"
    _make_decoder_with_past(broken)
    model = onnx.load(broken)
    for value in model.graph.input:
        if value.name == "past_key_values.0.encoder.key":
            value.name = "past_key_values.99.nonsense.key"
    for node in model.graph.node:
        node.input[:] = ["past_key_values.99.nonsense.key" if n ==
                         "past_key_values.0.encoder.key" else n for n in node.input]
    onnx.save(model, broken)
    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in (enc, dec, broken)]
    with pytest.raises(ValueError, match="naming scheme"):
        Seq2SeqDecoder(*sessions, eos_id=EOS, pad_id=PAD, decoder_start_id=START)


def test_no_repeat_ngram_guard_breaks_a_loop(toy):
    """Forced to start at token 1, the toy model falls into an endless "7 7 7";
    the guard must break out of it."""
    plain = toy.generate(INPUT, forced_bos_token_id=1,
                         config=GenerationConfig(num_beams=1, max_new_tokens=24))
    assert plain[-6:] == [7] * 6, "expected the unguarded run to loop"

    guarded = toy.generate(INPUT, forced_bos_token_id=1,
                           config=GenerationConfig(num_beams=1, max_new_tokens=24,
                                                    no_repeat_ngram_size=2))
    bigrams = [tuple(guarded[i:i + 2]) for i in range(len(guarded) - 1)]
    assert len(bigrams) == len(set(bigrams)), guarded
    assert len(guarded) < len(plain), "the guard should let the run reach EOS"


class TestBannedTokenIds:
    """linguonnx#42: mt-hitz-gl-eu-int8 (and the fp32 export it was quantised
    from) returned "" for every input. The decoder's own
    ``decoder_start_token_id``/``pad_token_id`` outscored every real word at
    generation step 1 - a known Marian degenerate mode upstream already names
    by shipping ``bad_words_ids: [[<that id>]]`` in ``generation_config.json``
    - but nothing enforced the ban, so the id was emitted for the whole
      budget and the tokenizer silently stripped it as a special token,
    leaving an empty string behind an HTTP 200.

    State 4 of the toy model (``forced_bos=8`` lands there) has PAD ranked
    first, reproducing exactly that shape of failure without needing the real
    HiTZ graphs.
    """

    def test_unguarded_greedy_reproduces_the_bug(self, toy):
        # Regression pin: without a ban PAD is chosen as real content right
        # after the forced token, and the toy model then loops forever since
        # PAD != EOS. This is what a caller used to get back as "".
        out = toy.generate(INPUT, forced_bos_token_id=8,
                           config=GenerationConfig(num_beams=1, max_new_tokens=12))
        assert out[1] == PAD, "expected the unguarded run to hit the bug"

    def test_banned_token_ids_keeps_pad_out_of_greedy_output(self, toy):
        out = toy.generate(INPUT, forced_bos_token_id=8,
                           config=GenerationConfig(
                               num_beams=1, max_new_tokens=12,
                               banned_token_ids=frozenset({PAD})))
        assert PAD not in out
        assert out, "banning PAD must not make the decoder produce nothing"

    def test_banned_token_ids_keeps_pad_out_of_beam_output(self, toy):
        for beams in (2, 4):
            out = toy.generate(INPUT, forced_bos_token_id=8,
                               config=GenerationConfig(
                                   num_beams=beams, max_new_tokens=12,
                                   banned_token_ids=frozenset({PAD})))
            assert PAD not in out

    def test_banned_token_ids_defaults_to_empty(self):
        assert GenerationConfig().banned_token_ids == frozenset()

    def test_banned_token_ids_accepts_any_iterable(self):
        assert GenerationConfig(banned_token_ids=[3, 3, 5]).banned_token_ids == \
            frozenset({3, 5})


def test_banned_ngram_tokens():
    assert _banned_ngram_tokens([1, 2, 3, 1], 2) == [2]
    assert _banned_ngram_tokens([1, 2, 3], 0) == []
    assert _banned_ngram_tokens([1], 3) == []
    assert _banned_ngram_tokens([1, 2, 1, 2, 1], 3) == [2]


# --------------------------------------------------------------------------
# input bounds and config validation
# --------------------------------------------------------------------------

class TestGenerationConfigValidation:
    """Every knob here reaches the library from `Translator(...)`, so this is
    the boundary where a bad value has to stop."""

    @pytest.mark.parametrize("beams", [0, -1, 1.5, "4"])
    def test_num_beams_must_be_a_positive_int(self, beams):
        with pytest.raises(ValueError, match="num_beams"):
            GenerationConfig(num_beams=beams)

    def test_num_beams_has_a_ceiling(self):
        from linguonnx.limits import MAX_NUM_BEAMS

        with pytest.raises(ValueError, match="LINGUONNX_MAX_NUM_BEAMS"):
            GenerationConfig(num_beams=MAX_NUM_BEAMS + 1)
        GenerationConfig(num_beams=MAX_NUM_BEAMS)  # the ceiling itself is fine

    @pytest.mark.parametrize("tokens", [0, -5, 2.0])
    def test_max_new_tokens_must_be_a_positive_int(self, tokens):
        with pytest.raises(ValueError, match="max_new_tokens"):
            GenerationConfig(max_new_tokens=tokens)

    @pytest.mark.parametrize("penalty", [-0.1, -2.0, 1e9, float("nan"),
                                         float("inf")])
    def test_length_penalty_is_range_checked(self, penalty):
        with pytest.raises(ValueError, match="length_penalty"):
            GenerationConfig(length_penalty=penalty)

    @pytest.mark.parametrize("penalty", [0.0, 1.0, 2.0])
    def test_sane_length_penalties_are_accepted(self, penalty):
        assert GenerationConfig(length_penalty=penalty).length_penalty == penalty

    def test_no_repeat_ngram_size_one_is_rejected(self):
        with pytest.raises(ValueError, match="every token"):
            GenerationConfig(no_repeat_ngram_size=1)

    def test_no_repeat_ngram_size_zero_and_two_are_accepted(self):
        assert GenerationConfig(no_repeat_ngram_size=0).no_repeat_ngram_size == 0
        assert GenerationConfig(no_repeat_ngram_size=2).no_repeat_ngram_size == 2

    def test_no_repeat_ngram_size_must_not_be_negative(self):
        with pytest.raises(ValueError, match="no_repeat_ngram_size"):
            GenerationConfig(no_repeat_ngram_size=-2)


class TestEncoderInputBound:
    def test_over_long_input_raises_and_names_the_knob(self, toy):
        toy.max_input_tokens = 8
        try:
            with pytest.raises(InputTooLongError,
                               match="LINGUONNX_MAX_ENCODER_TOKENS"):
                toy.generate([5] * 9, config=GenerationConfig(num_beams=1))
        finally:
            toy.max_input_tokens = MAX_ENCODER_TOKENS

    def test_input_at_the_bound_is_accepted(self, toy):
        toy.max_input_tokens = 4
        try:
            assert toy.generate(INPUT, config=GenerationConfig(num_beams=1))
        finally:
            toy.max_input_tokens = MAX_ENCODER_TOKENS

    def test_the_default_bound_comes_from_the_limits_module(self, toy):
        assert toy.max_input_tokens == MAX_ENCODER_TOKENS


class TestDecodeFailureIsDistinguishable:
    """`translate()` returns "" for empty input, so decode failure must not
    also come back as an empty sequence."""

    def test_beam_search_raises_when_no_beam_survives(self, toy, monkeypatch):
        # -inf logits everywhere: no candidate is finite, so the beam set
        # collapses on the first step and nothing ever reaches EOS.
        def dead_first_step(hidden, mask, token):
            logits, cache = Seq2SeqDecoder._first_step(toy, hidden, mask, token)
            return np.full_like(logits, -np.inf), cache

        monkeypatch.setattr(toy, "_first_step", dead_first_step)
        with pytest.raises(DecodeError, match="no output can be produced"):
            toy.generate(INPUT, config=GenerationConfig(num_beams=4))

    def test_greedy_raises_when_the_first_token_is_eos(self, toy, monkeypatch):
        def eos_first_step(hidden, mask, token):
            logits, cache = Seq2SeqDecoder._first_step(toy, hidden, mask, token)
            forced = np.full_like(logits, -1e9)
            forced[:, EOS] = 0.0
            return forced, cache

        monkeypatch.setattr(toy, "_first_step", eos_first_step)
        with pytest.raises(DecodeError, match="single token"):
            toy.generate(INPUT, config=GenerationConfig(num_beams=1))

    def test_a_successful_decode_still_returns_tokens(self, toy):
        assert toy.generate(INPUT, config=GenerationConfig(num_beams=4))
