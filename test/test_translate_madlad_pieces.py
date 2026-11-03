"""Every MADLAD target tag sends a piece of MADLAD's own vocabulary.

The registry speaks BCP-47 and MADLAD spells its pieces its own way: `fr-CA`
is `<2fr_CA>`, `sr-Latn` is `<2sh>`, `fa-AF` is `<2prs>`. A tag sent as
`<2fr-CA>` is not a piece, and `T5SpmTokenizer` refuses it, so every such
language raised at translation time while routing offered it.
"""
import urllib.request

import pytest

from linguonnx.detect.labels import to_bcp47
from linguonnx.model_manager import list_models
from linguonnx.translate.models import load_model
from linguonnx.translate.tokenizers import T5SpmTokenizer

TRANSLATE = list_models(kind="translate")
MADLAD = sorted(k for k, e in TRANSLATE.items() if e["arch"] == "madlad")


def _targets(model_id):
    entry = TRANSLATE[model_id]
    return entry.get("languages") or entry.get("tgt_languages")


@pytest.mark.parametrize("model_id", MADLAD)
def test_no_target_piece_is_spelled_as_a_bcp47_tag(model_id):
    model = load_model(model_id)
    assert not [tag for tag in _targets(model_id) if "-" in model.native_code(tag)]


@pytest.mark.parametrize("model_id", ["madlad400-3b-mt", "madlad400-3b-mt-int8"])
def test_every_base_target_tag_maps_back_from_its_piece(model_id):
    """The fine-tunes map `oc` to `oc_aranes` on purpose; the base model's
    pieces are its own languages, each spelled the MADLAD way."""
    model = load_model(model_id)
    for tag in _targets(model_id):
        native = model.native_code(tag)
        assert native == tag or to_bcp47(native) == tag, (tag, native)


@pytest.mark.parametrize("tag,piece", [("fr-CA", "fr_CA"), ("sr-Latn", "sh"),
                                       ("fa-AF", "prs"), ("zh-Hant", "zh_Hant"),
                                       ("crh", "crh"), ("pt", "pt")])
def test_the_base_model_sends_its_own_spelling(tag, piece):
    assert load_model("madlad400-3b-mt-int8").native_code(tag) == piece


@pytest.mark.network
def test_every_base_target_piece_is_in_the_vocabulary(tmp_path):
    url = "https://huggingface.co/TigreGotico/madlad400-3b-mt-onnx/resolve/main/spiece.model"
    path = tmp_path / "spiece.model"
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    tokenizer = T5SpmTokenizer(path)
    model = load_model("madlad400-3b-mt-int8")
    refused = []
    for tag in _targets("madlad400-3b-mt-int8"):
        try:
            tokenizer.encode("Good day.", prefix=f"<2{model.native_code(tag)}>")
        except ValueError:
            refused.append(tag)
    assert refused == []
