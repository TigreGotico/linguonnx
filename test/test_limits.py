"""Input bounds and the errors raised when they are crossed.

Everything here is a pure function or a hand-built object: no model files, no
network. The point of each test is that an over-size or content-free input
*raises* and says which knob to turn, instead of quietly producing an answer
computed from part of the input.
"""

import json

import numpy as np
import pytest

from linguonnx.detect.hashing import GlotLIDFeaturizer
from linguonnx.detect.hs import HSCombiner
from linguonnx.limits import (CorruptModelFileError, InputTooLongError,
                              check_length, has_visible_content)

WORDS = ["the", "and", "hello", "</s>"]


def _featurizer(**kwargs):
    return GlotLIDFeaturizer(words=WORDS, nwords=len(WORDS), minn=2, maxn=4,
                             bucket=1000, **kwargs)


class TestVisibleContent:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "\t\n",
        "​",            # zero-width space
        "‍‍",      # zero-width joiner, as pasted from HTML
        "﻿",            # BOM
        "​ ‎\n",   # mixed with real whitespace and a bidi mark
    ])
    def test_invisible_text_has_no_content(self, text):
        assert has_visible_content(text) is False

    @pytest.mark.parametrize("text", ["a", " ola ", "​ola​", "。"])
    def test_visible_text_is_kept(self, text):
        assert has_visible_content(text) is True


class TestCheckLength:
    def test_at_the_limit_is_allowed(self):
        check_length(10, 10, "tokens", "KNOB")

    def test_over_the_limit_names_the_knob_and_both_numbers(self):
        with pytest.raises(InputTooLongError) as err:
            check_length(11, 10, "tokens", "KNOB")
        message = str(err.value)
        assert "11 tokens" in message
        assert "10" in message
        assert "KNOB" in message


class TestFeaturizerBounds:
    def test_character_limit_raises_rather_than_truncating(self):
        featurize = _featurizer(max_chars=50)
        with pytest.raises(InputTooLongError, match="LINGUONNX_MAX_DETECT_CHARS"):
            featurize("a" * 51)

    def test_input_at_the_character_limit_still_works(self):
        featurize = _featurizer(max_chars=50)
        assert featurize("a" * 50).size > 0

    def test_token_limit_raises_and_names_its_knob(self):
        featurize = _featurizer(max_chars=10_000, max_tokens=8)
        with pytest.raises(InputTooLongError, match="LINGUONNX_MAX_LINE_TOKENS"):
            featurize(" ".join(["ola"] * 9))

    def test_token_limit_counts_the_callers_tokens_only(self):
        # tokenize() appends fastText's implicit EOS; it must not eat one slot
        # of the caller's budget.
        featurize = _featurizer(max_chars=10_000, max_tokens=8)
        assert featurize(" ".join(["ola"] * 8)).size > 0

    def test_the_defaults_are_the_configured_module_values(self):
        from linguonnx import limits

        featurize = _featurizer()
        assert featurize.max_chars == limits.MAX_DETECT_CHARS
        assert featurize.max_tokens == limits.MAX_LINE_TOKENS

    def test_a_huge_input_is_rejected_before_any_hashing_happens(self):
        featurize = _featurizer()
        with pytest.raises(InputTooLongError):
            featurize("a" * 200_000)


class TestHSTreeValidation:
    """A truncated or stale hs_tree.json is non-zero, so the cache keeps it."""

    def test_a_tree_for_another_label_count_is_named_as_such(self, tmp_path):
        tree = tmp_path / "hs_tree.json"
        tree.write_text(json.dumps({
            "paths": [[1], [0, 1], [0, 1]],
            "codes": [[False], [True, True], [False, True]],
        }))
        with pytest.raises(CorruptModelFileError) as err:
            HSCombiner.from_file(tree, num_labels=176)
        assert "3 labels" in str(err.value)
        assert "176" in str(err.value)

    def test_a_matching_tree_loads(self, tmp_path):
        tree = tmp_path / "hs_tree.json"
        tree.write_text(json.dumps({
            "paths": [[1], [0, 1], [0, 1]],
            "codes": [[False], [True, True], [False, True]],
        }))
        assert HSCombiner.from_file(tree, num_labels=3).paths[0] == [1]

    def test_truncated_json_is_not_an_opaque_decode_error(self, tmp_path):
        tree = tmp_path / "hs_tree.json"
        tree.write_text('{"paths": [[1], [0, 1]], "cod')
        with pytest.raises(CorruptModelFileError, match="not valid JSON"):
            HSCombiner.from_file(tree)

    def test_a_missing_key_is_named(self, tmp_path):
        tree = tmp_path / "hs_tree.json"
        tree.write_text(json.dumps({"paths": [[0]]}))
        with pytest.raises(CorruptModelFileError, match="'codes'"):
            HSCombiner.from_file(tree)

    def test_paths_and_codes_of_different_lengths_are_rejected(self):
        with pytest.raises(CorruptModelFileError, match="2 paths but 1 codes"):
            HSCombiner(paths=[[0], [0]], codes=[[True]])

    def test_a_path_longer_than_its_code_is_rejected(self):
        with pytest.raises(CorruptModelFileError, match="label 1"):
            HSCombiner(paths=[[1], [0, 1]], codes=[[True], [True]])

    def test_a_node_index_outside_the_tree_is_rejected(self):
        # 3 labels means 2 internal nodes, so node 7 cannot exist. Without
        # this check it becomes an IndexError inside the path walk.
        with pytest.raises(CorruptModelFileError, match="node 7"):
            HSCombiner(paths=[[1], [0, 7], [0, 1]],
                       codes=[[True], [True, True], [False, True]])

    def test_a_short_model_output_is_reported_with_both_shapes(self):
        combine = HSCombiner(paths=[[1], [0, 1], [0, 1]],
                             codes=[[False], [True, True], [False, True]])
        with pytest.raises(CorruptModelFileError) as err:
            combine(np.array([0.5], dtype=np.float32))
        assert "needs 2 nodes" in str(err.value)
        assert "outputs 1" in str(err.value)
