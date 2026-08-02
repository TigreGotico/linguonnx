"""Real translation, real weights, real routing.

Downloads the int8 graphs on first run (about 1.2 GB for M2M100-418M plus
about 350 MB per opus-mt pair). Skip with `-m "not network"` when offline.

The reference comparisons import `transformers` and `optimum`, which pull in
torch. That is fine **here**: the whole point is to prove the hand-written
numpy decoder reproduces what the torch stack produces. Nothing in
`linguonnx/` imports either of them, and :func:`test_the_library_never_imports_torch`
holds that line.
"""

import sys

import pytest

from linguonnx import load_translator

pytestmark = pytest.mark.network

optimum = pytest.importorskip("optimum.onnxruntime", reason="reference outputs need optimum")
transformers = pytest.importorskip("transformers")

SENTENCE = "The library opens at nine in the morning."


@pytest.fixture(scope="module")
def tx():
    return load_translator()


def _reference(repo, text, src=None, tgt=None, subfolder="int8", num_beams=4):
    """What optimum + transformers produce from the same ONNX graphs."""
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    kwargs = {"subfolder": subfolder} if subfolder else {}
    tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
    model = ORTModelForSeq2SeqLM.from_pretrained(repo, use_cache=True, **kwargs)
    forced = None
    if src is not None:
        tokenizer.src_lang = src
        forced = tokenizer.convert_tokens_to_ids(_lang_token(repo, tgt))
    inputs = tokenizer(text, return_tensors="pt")
    ids = model.generate(**inputs, forced_bos_token_id=forced,
                         num_beams=num_beams, max_new_tokens=64)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def _lang_token(repo, code):
    return code if "nllb" in repo else f"__{code}__"


# --- Marian: the model is the pair ----------------------------------------

@pytest.mark.parametrize("tgt,model_id,repo", [
    ("pt", "opus-mt-en-pt-int8", "TigreGotico/opus-mt-en-pt-onnx"),
    ("gl", "opus-mt-en-gl-int8", "TigreGotico/opus-mt-en-gl-onnx"),
    ("ca", "opus-mt-en-ca-int8", "TigreGotico/opus-mt-en-ca-onnx"),
])
def test_marian_matches_the_optimum_reference(tx, tgt, model_id, repo):
    ours = tx.translate(SENTENCE, src="en", tgt=tgt, model=model_id)
    assert ours.strip(), "empty translation"
    assert ours == _reference(repo, SENTENCE)


def test_marian_is_what_routing_picks_for_en_pt(tx):
    out, route = tx.translate(SENTENCE, src="en", tgt="pt", return_route=True)
    assert route.model_ids == ("opus-mt-en-pt-int8",)
    assert route.hops[0].dedicated
    assert "biblioteca" in out.lower()


# --- M2M100: forced_bos_token_id selects the target ------------------------

@pytest.mark.parametrize("tgt", ["pt", "gl", "ca"])
def test_m2m100_matches_the_optimum_reference(tx, tgt):
    ours = tx.translate(SENTENCE, src="en", tgt=tgt, model="m2m100-418M-int8")
    assert ours.strip()
    assert ours == _reference("TigreGotico/m2m100-418M-onnx", SENTENCE,
                              src="en", tgt=tgt)


def test_m2m100_target_selection_actually_changes_the_language(tx):
    """Getting forced_bos wrong does not raise - it silently returns the wrong
    language. So assert the three targets differ from each other."""
    outputs = {tgt: tx.translate(SENTENCE, src="en", tgt=tgt,
                                 model="m2m100-418M-int8")
               for tgt in ("pt", "gl", "ca", "de")}
    assert len(set(outputs.values())) == 4, outputs
    assert "matí" in outputs["ca"] or "biblioteca" in outputs["ca"].lower()
    assert "Bibliothek" in outputs["de"]


def test_m2m100_has_no_basque(tx):
    model = tx.model("m2m100-418M-int8")
    assert "eu" not in model.languages
    with pytest.raises(KeyError):
        model.native_code("eu")


# --- the routed, multi-hop case -------------------------------------------

def test_portuguese_to_basque_routes_away_from_m2m100(tx):
    out, route = tx.translate("Bom dia, o tempo esta muito bom hoje.",
                              src="pt", tgt="eu", return_route=True)
    assert route.n_hops == 2
    assert route.pivots == ("en",)
    assert route.model_ids == ("opus-mt-pt-en-int8", "opus-mt-en-eu-int8")
    assert out.strip()
    # Basque is agglutinative and looks like nothing else here; "eguraldi"
    # (weather) and "egun on" (good morning) are the giveaways.
    assert "egun on" in out.lower() or "eguraldi" in out.lower(), out


def test_each_hop_of_the_route_matches_its_own_reference(tx):
    """A chain is only correct if each link is; check the links separately."""
    text = "Bom dia, o tempo esta muito bom hoje."
    route = tx.route("pt", "eu")
    intermediate = tx.translate(text, model=route.hops[0].model_id)
    assert intermediate == _reference("TigreGotico/opus-mt-pt-en-onnx", text)
    final = tx.translate(intermediate, model=route.hops[1].model_id)
    assert final == _reference("TigreGotico/opus-mt-en-eu-onnx", intermediate)
    assert tx.translate(text, src="pt", tgt="eu") == final


# --- generation settings ---------------------------------------------------

def test_greedy_and_beam_both_produce_sane_output(tx):
    greedy = tx.translate(SENTENCE, src="en", tgt="pt", num_beams=1)
    beam = tx.translate(SENTENCE, src="en", tgt="pt", num_beams=4)
    assert greedy.strip() and beam.strip()
    assert "biblioteca" in greedy.lower() and "biblioteca" in beam.lower()


def test_max_new_tokens_truncates(tx):
    short = tx.translate(SENTENCE, src="en", tgt="pt", max_new_tokens=3)
    full = tx.translate(SENTENCE, src="en", tgt="pt")
    assert len(short) < len(full)


def test_empty_input_is_empty_output(tx):
    assert tx.translate("   ", src="en", tgt="pt") == ""


# --- the hard constraint ---------------------------------------------------

def test_the_library_never_imports_torch():
    """linguonnx must run on onnxruntime + numpy + sentencepiece alone."""
    import subprocess
    code = ("import sys, linguonnx;"
            "tx = linguonnx.load_translator();"
            "tx.route('pt', 'eu');"
            "assert 'torch' not in sys.modules, sorted(sys.modules)[:0] or 'torch imported';"
            "assert 'transformers' not in sys.modules;"
            "print('clean')")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
