"""Decoder-only translators: Projecte AINA's Plume models.

Offline. A tiny real ONNX graph with the export's input and output names
checks the cache and attention-mask wiring of ``CausalLMDecoder``; the prompt
is checked against the format string on the upstream card; the registry is
checked for the entries the sync script writes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from linguonnx.model_manager import list_models
from linguonnx.translate.decode import CausalLMDecoder, GenerationConfig
from linguonnx.translate.models import TranslationModel, capability_from_entry
from linguonnx.translate.preprocess import pipeline_for

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sync_registry  # noqa: E402

TRANSLATE = list_models(kind="translate")

VOCAB = 10
N_STATES = 7
EOS = 1
PAD = 3
HEAD_DIM = 2
TABLE = np.random.default_rng(11).normal(size=(N_STATES, VOCAB)).astype(np.float32)


# --------------------------------------------------------------------------
# A toy decoder-only graph
# --------------------------------------------------------------------------

def _make_causal_lm(path):
    """state = (sum of every cached value + attention-mask width) mod N_STATES.

    The cache holds each token's id, ``HEAD_DIM`` times, so a dropped or
    stale cache and a mask that does not grow both change the state, and so
    the next token.
    """
    consts = [
        numpy_helper.from_array(TABLE, "TABLE"),
        numpy_helper.from_array(np.array([1, 3], dtype=np.int64), "AX13"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "AX1"),
        numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "AX123"),
        numpy_helper.from_array(np.array(N_STATES, dtype=np.int64), "NSTATES"),
    ]
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["f", "AX13"], ["u"]),
        helper.make_node("Concat", ["u", "u"], ["new"], axis=3),
        helper.make_node("Concat", ["past_key_values.0.key", "new"],
                         ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.value", "new"],
                         ["present.0.value"], axis=2),
        helper.make_node("ReduceSum", ["present.0.key", "AX123"], ["ksum"],
                         keepdims=0),
        helper.make_node("Cast", ["attention_mask"], ["mf"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["mf", "AX1"], ["msum"], keepdims=0),
        helper.make_node("Add", ["ksum", "msum"], ["total"]),
        helper.make_node("Cast", ["total"], ["ti"], to=TensorProto.INT64),
        helper.make_node("Mod", ["ti", "NSTATES"], ["state"]),
        helper.make_node("Gather", ["TABLE", "state"], ["g"], axis=0),
        helper.make_node("Unsqueeze", ["g", "AX1"], ["logits"]),
    ]
    cache = ["b", 1, "p", HEAD_DIM]
    present = ["b", 1, "p+s", HEAD_DIM]
    graph = helper.make_graph(
        nodes, "causal_lm",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "p+s"]),
         helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, cache),
         helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, cache)],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 1, VOCAB]),
         helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, present),
         helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, present)],
        initializer=consts)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    path = tmp_path_factory.mktemp("causal") / "model.onnx"
    _make_causal_lm(path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _reference_greedy(prompt, max_new_tokens):
    """The same model in numpy: the state reads the whole history."""
    history, generated = list(prompt), []
    for _ in range(max_new_tokens):
        state = (HEAD_DIM * sum(history) + len(history)) % N_STATES
        token = int(np.argmax(TABLE[state]))
        if token == EOS:
            break
        generated.append(token)
        history.append(token)
    return generated


@pytest.mark.parametrize("prompt", [[0, 5, 2], [0, 3, 8], [0, 9, 8, 6, 5, 2]])
def test_greedy_decoding_matches_the_numpy_model(session, prompt):
    decoder = CausalLMDecoder(session, eos_id=EOS, pad_id=PAD)
    expected = _reference_greedy(prompt, 12)
    assert expected, "the toy prompt must generate at least one token"
    config = GenerationConfig(num_beams=1, max_new_tokens=12)
    assert decoder.generate(prompt, config=config) == expected


def test_beam_search_runs_on_the_same_graph(session):
    decoder = CausalLMDecoder(session, eos_id=EOS, pad_id=PAD)
    output = decoder.generate([0, 5, 2],
                              config=GenerationConfig(num_beams=3, max_new_tokens=12))
    assert output and EOS not in output


def test_a_graph_without_a_cache_is_refused(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Cast", ["input_ids"], ["logits"], to=TensorProto.FLOAT)],
        "no_cache",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", "s"])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "model.onnx")
    session = ort.InferenceSession(str(tmp_path / "model.onnx"),
                                   providers=["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="past_key_values"):
        CausalLMDecoder(session, eos_id=EOS, pad_id=PAD)


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

#: Quoted from https://huggingface.co/projecte-aina/Plume32k, "Run the model".
CARD_PROMPT = '<s> [{}] {} \n[{}]'


class _RecordingTokenizer:
    def __init__(self):
        self.prompts = []

    def encode(self, prompt):
        self.prompts.append(prompt)
        return [0]


@pytest.mark.parametrize("model_id", ["Plume32k", "Plume32k-int8",
                                      "Plume128k", "Plume128k-int8"])
@pytest.mark.parametrize("src,tgt,src_code,tgt_code", [
    ("es", "ca", "spa_Latn", "cat_Latn"),
    ("ca", "eu", "cat_Latn", "eus_Latn"),
])
def test_the_prompt_is_the_cards_prompt(model_id, src, tgt, src_code, tgt_code):
    model = TranslationModel(model_id, entry=TRANSLATE[model_id])
    model._tokenizer = _RecordingTokenizer()
    sentence = "Ayer se fue, tomó sus cosas y se puso a navegar."
    pipeline_for("causal-lm").encode(model, sentence, src, tgt)
    assert model._tokenizer.prompts == [
        CARD_PROMPT.format(src_code, sentence, tgt_code)]


def test_a_newline_in_the_text_is_refused_by_the_pipeline():
    model = TranslationModel("Plume32k-int8", entry=TRANSLATE["Plume32k-int8"])
    model._tokenizer = _RecordingTokenizer()
    with pytest.raises(ValueError, match="single line"):
        pipeline_for("causal-lm").encode(model, "one\ntwo", "en", "ca")


def test_each_line_is_its_own_prompt(monkeypatch):
    model = TranslationModel("Plume32k-int8", entry=TRANSLATE["Plume32k-int8"])
    monkeypatch.setattr("linguonnx.translate.models.entry_runnability",
                        lambda entry: (True, None))
    seen = []
    pipeline = pipeline_for("causal-lm")
    monkeypatch.setattr(pipeline, "encode",
                        lambda model, text, src, tgt, target_token=None:
                        seen.append(text) or [len(text)])
    monkeypatch.setattr(pipeline, "decode",
                        lambda model, ids, src, tgt: f"<{ids[0]}>")

    class _Decoder:
        def generate(self, ids, forced_bos_token_id=None, config=None):
            return list(ids)

    model._decoder = _Decoder()
    model._banned_token_ids = frozenset()
    assert model.translate("one\n\nthree", "en", "ca") == "<3>\n\n<5>"
    assert seen == ["one", "three"]


# --------------------------------------------------------------------------
# The tokenizer
# --------------------------------------------------------------------------

def _tiny_tokenizer_json(path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, decoders

    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, "<pad>": 3, "[cat_Latn]": 4,
             "hola": 5, ".": 6, "món": 7}
    tok = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    tok.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>", "[cat_Latn]"])
    tok.save(str(path))


def test_the_tokenizer_encodes_the_prompts_own_specials_and_adds_none(tmp_path):
    from linguonnx.translate.tokenizers import FastPromptTokenizer

    _tiny_tokenizer_json(tmp_path / "tokenizer.json")
    tok = FastPromptTokenizer(tmp_path / "tokenizer.json")
    assert tok.encode("<s> hola [cat_Latn]") == [0, 5, 4]


def test_the_tokenizer_drops_specials_and_cleans_up_spaces(tmp_path):
    from linguonnx.translate.tokenizers import FastPromptTokenizer

    _tiny_tokenizer_json(tmp_path / "tokenizer.json")
    ids = [5, 7, 6, 1]
    assert FastPromptTokenizer(tmp_path / "tokenizer.json",
                               clean_up_tokenization_spaces=True
                               ).decode(ids) == "hola món."
    assert FastPromptTokenizer(tmp_path / "tokenizer.json").decode(ids) == "hola món ."


# --------------------------------------------------------------------------
# Classification and the registry
# --------------------------------------------------------------------------

PLUME_FILES = {"model.onnx": 1, "model.onnx_data": 1, "config.json": 1,
               "tokenizer.json": 1, "tokenizer_config.json": 1,
               "generation_config.json": 1}


def test_a_causal_lm_export_is_a_translation_candidate():
    assert sync_registry.classify(
        PLUME_FILES, {"architectures": ["GemmaForCausalLM"],
                      "model_type": "gemma"}) == "translate"


def test_a_sequence_classifier_export_is_still_a_lid_candidate():
    files = {"model.onnx": 1, "config.json": 1, "tokenizer.json": 1}
    assert sync_registry.classify(
        files, {"architectures": ["XLMRobertaForSequenceClassification"]}) == "lid"


def test_an_unlisted_causal_lm_is_skipped_with_the_reason():
    detail = {"config": {"architectures": ["LlamaForCausalLM"],
                         "model_type": "llama"}}
    with pytest.raises(sync_registry.SkipRepo, match="CAUSAL_LM_MODELS"):
        sync_registry._arch(detail, PLUME_FILES, "some-chat-model")


SUPERVISED = {(src, tgt) for other in ("de", "en", "es", "eu", "fr", "gl", "it", "pt")
              for src, tgt in (("ca", other), (other, "ca"))}


@pytest.mark.parametrize("model_id,graph,external", [
    ("Plume32k", "model.onnx", ["model.onnx_data"]),
    ("Plume32k-int8", "int8/model_quantized.onnx", []),
    ("Plume128k", "model.onnx", ["model.onnx_data"]),
    ("Plume128k-int8", "int8/model_quantized.onnx",
     ["int8/model_quantized.onnx_data"]),
])
def test_plume_is_registered_for_its_sixteen_supervised_directions(
        model_id, graph, external):
    entry = TRANSLATE[model_id]
    assert entry["arch"] == "causal-lm"
    assert entry["graphs"] == {"model": graph}
    assert entry["extra_files"] == external
    assert entry["prompt_template"] == "<s> [{src}] {text} \n[{tgt}]"
    assert entry["native_codes"]["eu"] == "eus_Latn"
    assert capability_from_entry(entry).directions == frozenset(SUPERVISED)


def test_plume_is_no_longer_skipped():
    import linguonnx
    path = Path(linguonnx.__file__).parent / "model_index" / "skipped.json"
    skipped = json.loads(path.read_text(encoding="utf-8"))
    assert "TigreGotico/Plume32k-onnx" not in skipped
    assert "TigreGotico/Plume128k-onnx" not in skipped


# --------------------------------------------------------------------------
# Real weights
# --------------------------------------------------------------------------

#: Greedy output of the PyTorch checkpoint `projecte-aina/Plume32k` on FLORES-200
#: devtest sentences, one per row: the card's prompt, `generate(do_sample=False,
#: num_beams=1, max_new_tokens=128)`, the prompt and the stop token cut off.
REFERENCE = ROOT / "test" / "data" / "plume32k_torch_greedy.json"


@pytest.mark.network
@pytest.mark.parametrize("row", json.loads(REFERENCE.read_text(encoding="utf-8")),
                         ids=lambda row: f"{row['src']}-{row['tgt']}-{row['index']}")
def test_plume32k_fp32_matches_the_pytorch_reference_token_for_token(row):
    """Downloads the 8.5 GB fp32 export and needs about 9 GB of memory."""
    from linguonnx.translate import load_translator

    tx = load_translator(model="Plume32k", precision="fp32", num_beams=1)
    model = tx.model("Plume32k")
    prompt = model.pipeline.encode(model, row["source"], row["src"], row["tgt"])
    assert prompt == row["prompt_ids"]
    config = GenerationConfig(num_beams=1, max_new_tokens=128)
    assert model.decoder.generate(prompt, config=config) == row["ids"]
    assert tx.translate(row["source"], row["src"], row["tgt"],
                        model="Plume32k") == row["text"]
