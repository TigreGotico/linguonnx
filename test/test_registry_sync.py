"""The generated registries have to be valid, and routable, without a network.

``translate.json`` and ``lid.json`` are produced by ``scripts/sync_registry.py``
from the HuggingFace API. Everything here reads the *committed* files, so it
runs offline and guards the artefact rather than the generator - except
:func:`test_committed_registry_matches_the_hub`, which is the drift check and
is marked ``network``.

The tests that matter most are the coverage ones. A registry entry is a
*claim*: "route src -> tgt through me". A wrong claim does not raise, it
returns fluent text in a language nobody asked for, so the claims are checked
against what each architecture can actually serve.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from linguonnx import model_manager
from linguonnx.translate import load_translator
from linguonnx.translate.graph import (LICENSE_TIERS, NoRouteError,
                                       normalize_tag)

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_registry.py"


def _load_sync_registry():
    """Import ``scripts/sync_registry.py`` as a module, for unit-testing its
    pure functions (``_arch``, ``_marian_group_target_tokens``, ...) without
    a network call. It is a script, not a package, so it is not importable
    the normal way.
    """
    spec = importlib.util.spec_from_file_location("sync_registry", SYNC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync_registry = _load_sync_registry()

TRANSLATE = model_manager.list_models(kind="translate")
LID = model_manager.list_models(kind="lid")

#: M2M100's 100 languages, per its own tokenizer, do **not** include Basque.
#: Everything Iberian routes through opus-mt for `eu`, and an entry that
#: claimed otherwise would make the router pick M2M100 and quietly emit
#: Spanish. This is the single most load-bearing fact in the registry.
M2M100_HAS_NO_BASQUE = "eu"


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", sorted(TRANSLATE))
def test_translate_entry_is_complete(model_id):
    entry = TRANSLATE[model_id]
    for key in ("model_id", "hf_repo", "arch", "graphs", "side_files",
                "extra_files", "license", "license_tier", "size_mb",
                "precision"):
        assert key in entry, f"{model_id} is missing {key!r}"
    assert entry["model_id"] == model_id
    assert entry["hf_repo"].startswith("TigreGotico/")
    assert entry["arch"] in (
        "marian", "m2m100", "nllb", "madlad", "indictrans2", "opennmt-bpe",
        "pegasus-fast")
    assert entry["precision"] in ("fp32", "int8")
    assert entry["size_mb"] > 0


@pytest.mark.parametrize("model_id", sorted(TRANSLATE))
def test_translate_entry_declares_a_licence(model_id):
    entry = TRANSLATE[model_id]
    assert entry["license"].strip(), f"{model_id} has an empty licence"
    assert entry["license_tier"] in LICENSE_TIERS, \
        f"{model_id} has licence tier {entry['license_tier']!r}"


@pytest.mark.parametrize("model_id", sorted(TRANSLATE))
def test_translate_entry_references_plausible_files(model_id):
    entry = TRANSLATE[model_id]
    assert set(entry["graphs"]) == {"encoder", "decoder", "decoder_with_past"}
    for name in entry["graphs"].values():
        assert name.endswith(".onnx"), f"{model_id}: {name} is not a graph"
    for name in entry["extra_files"]:
        assert name.endswith(".onnx_data"), \
            f"{model_id}: {name} is not external weight data"
    # int8 entries must point at int8 graphs; fp32 entries must not.
    is_int8 = entry["precision"] == "int8"
    for name in entry["graphs"].values():
        assert name.startswith("int8/") == is_int8, \
            f"{model_id}: {name} does not match precision {entry['precision']}"


@pytest.mark.parametrize("model_id", sorted(TRANSLATE))
def test_translate_entry_carries_side_files_its_tokenizer_needs(model_id):
    entry = TRANSLATE[model_id]
    side = entry["side_files"]
    required = {
        "marian": {"source_spm", "target_spm", "vocab", "config"},
        "m2m100": {"spm", "config", "special_tokens_map", "vocab"},
        "nllb": {"spm", "config", "special_tokens_map"},
        "madlad": {"spm", "config"},
        "indictrans2": {"spm_src", "spm_tgt", "dict_src", "dict_tgt", "config"},
        "opennmt-bpe": {"vocab", "config", "bpe_code"},
        "pegasus-fast": {"tokenizer_json", "config"},
    }[entry["arch"]]
    assert required <= set(side), \
        f"{model_id} ({entry['arch']}) is missing {required - set(side)}"


@pytest.mark.parametrize("model_id", sorted(LID))
def test_lid_entry_is_complete(model_id):
    entry = LID[model_id]
    for key in ("model_id", "hf_repo", "onnx_file", "side_files", "license",
                "engine", "loss", "num_labels", "size_mb", "precision"):
        assert key in entry, f"{model_id} is missing {key!r}"
    assert entry["license"].strip()
    assert entry["onnx_file"].endswith(".onnx")
    assert entry["num_labels"] > 0
    assert entry["size_mb"] > 0
    assert {"vocab", "labels", "config"} <= set(entry["side_files"])


def test_every_entry_declares_exactly_one_kind_of_coverage():
    """A model is a pair, an any-to-any set, or a directional set - never
    zero of those and never more than one."""
    for model_id, entry in TRANSLATE.items():
        has_pair = "pair" in entry
        has_set = bool(entry.get("languages"))
        has_directional = bool(entry.get("src_languages") or entry.get("tgt_languages"))
        kinds = sum((has_pair, has_set, has_directional))
        assert kinds == 1, (
            f"{model_id} declares {kinds} kinds of coverage "
            f"(pair={has_pair}, languages={has_set}, directional={has_directional})")


def test_directional_entries_declare_both_sides():
    """A one-directional covering-set model needs both a source and a target set."""
    for model_id, entry in TRANSLATE.items():
        if "src_languages" in entry or "tgt_languages" in entry:
            assert entry.get("src_languages"), f"{model_id} has no src_languages"
            assert entry.get("tgt_languages"), f"{model_id} has no tgt_languages"


# ---------------------------------------------------------------------------
# Coverage claims vs what the architecture can serve
# ---------------------------------------------------------------------------

def test_no_m2m100_entry_claims_basque():
    """M2M100 has no Basque. Any entry claiming `eu` would silently mis-route.

    The claim is checked after BCP-47 normalisation, because the registry
    stores the model's own codes and the router works in normalised tags -
    a bug that only appears on one side of that conversion is still a bug.
    """
    for model_id, entry in TRANSLATE.items():
        if entry["arch"] != "m2m100":
            continue
        claimed = {normalize_tag(code) for code in entry.get("languages", ())}
        if "pair" in entry:
            claimed |= {normalize_tag(code) for code in entry["pair"]}
        assert M2M100_HAS_NO_BASQUE not in claimed, (
            f"{model_id} is m2m100 and claims Basque; M2M100 cannot translate "
            f"it and the router would pick this model over opus-mt-*-eu")


def test_basque_is_served_only_by_models_that_actually_have_it():
    """Whatever claims `eu` must be a model that actually ships it, never M2M100."""
    for model_id, entry in TRANSLATE.items():
        endpoints = {normalize_tag(c) for c in entry.get("pair", ())}
        endpoints |= {normalize_tag(c) for c in entry.get("languages", ())}
        if "eu" in endpoints:
            assert entry["arch"] in ("marian", "nllb", "madlad", "pegasus-fast"), \
                f"{model_id} claims Basque with arch {entry['arch']}"


def test_multi_target_group_models_record_their_target_token():
    """opus-mt group models pick the wrong language without a prefix token.

    ``opus-mt-en-pl-onnx`` is an export of ``opus-mt-en-sla``, which serves
    several Slavic languages from one decoder. Without ``>>pol<<`` on the
    input it answers in whichever it likes. If the registry ever lists such an
    entry without the token, translation is silently wrong.
    """
    for model_id, entry in TRANSLATE.items():
        base = entry.get("base_model", "")
        if not base or entry["arch"] != "marian" or not entry.get("pair"):
            # A group model has no pair to compare against: it covers the
            # whole family and carries a `target_token_template` instead. See
            # `test_group_models_carry_a_target_token_template` below.
            continue
        stem = base.split("/")[-1]
        for prefix in ("opus-mt-", "tc-big-"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        parts = stem.split("-")
        if len(parts) != 2:
            continue
        _, base_tgt = parts
        if base_tgt != entry["pair"][1]:
            token = entry.get("target_token", "")
            assert token.startswith(">>") and token.endswith("<<"), (
                f"{model_id} exports {base} but targets {entry['pair'][1]}; "
                f"it needs a target_token and has {token!r}")


def test_group_models_carry_a_target_token_template():
    """A Marian entry with no pair must be able to name its target.

    ``opus-mt-itc-itc`` serves any Italic language from one decoder and picks
    which one from a ``>>ita<<`` prefix token. An entry with neither a pair
    nor a way to build that token cannot select a target at all, and answers
    fluently in the wrong language rather than raising.

    ``opus-mt-tc-big-cat_oci_spa-en`` is the one genuine exception: many
    sources, one fixed target, no ``>>xxx<<`` token in its vocabulary at all
    (verified by hand - see ``_KNOWN_UNRESOLVED_GROUP_MODELS`` in
    ``scripts/sync_registry.py``), so there is nothing to disambiguate and no
    template is needed. It is recognised the same way
    ``entry_runnability`` recognises it: no ``pair``, and exactly one
    selectable target.
    """
    for model_id, entry in TRANSLATE.items():
        if entry["arch"] != "marian" or entry.get("pair"):
            continue
        targets = entry.get("tgt_languages") if entry.get("tgt_languages") is not None \
            else entry.get("languages")
        if isinstance(targets, list) and len(targets) == 1:
            continue
        template = entry.get("target_token_template", "")
        assert "{code}" in template, (
            f"{model_id} is a multi-target Marian model with no "
            f"target_token_template")
        # A directional group model (`opus-mt-tc-big-en-zle`: English -> the
        # six East-Slavic codes its vocabulary carries a token for) states its
        # targets in `tgt_languages`; an any-to-any one states them in
        # `languages`. Either way, every target has to build a token.
        for tag in targets:
            native = (entry.get("native_codes") or {}).get(tag, tag)
            assert template.format(code=native)


def test_target_tokens_are_only_set_where_they_are_needed():
    for model_id, entry in TRANSLATE.items():
        if "target_token" not in entry:
            continue
        assert entry["arch"] == "marian", \
            f"{model_id} sets target_token but is {entry['arch']}"


# ---------------------------------------------------------------------------
# The 19 models excluded for an architecture/tokenizer mismatch, now resolved
# ---------------------------------------------------------------------------

#: The 14 `aina-translator-ca-*` pairs: tagged `model_type: m2m_100` but
#: actually ship a dual Marian tokenizer (`source.spm` + `target.spm` +
#: `vocab.json`). See `_arch()` in scripts/sync_registry.py.
_AINA_MARIAN_PAIRS = (
    "aina-translator-ca-de", "aina-translator-ca-en", "aina-translator-ca-es",
    "aina-translator-ca-fr", "aina-translator-ca-it", "aina-translator-ca-pt",
    "aina-translator-de-ca", "aina-translator-en-ca", "aina-translator-es-ca",
    "aina-translator-eu-ca", "aina-translator-fr-ca", "aina-translator-gl-ca",
    "aina-translator-it-ca", "aina-translator-pt-ca",
)

#: Softcatalà's two exports: tagged `model_type: pegasus` but ship a
#: `tokenizers`-library `tokenizer.json` instead of an OpenNMT BPE vocab.
_SOFTCATALA_PEGASUS_FAST = ("translate-eus-cat", "translate-oci-cat")

#: The 3 `opus-mt-tc-big-*` group models: underscore-joined macro-language
#: names (`cat_oci_spa`) or a family/collection code (`itc`) that the repo
#: name's own `xx-yy` shape hides real per-checkpoint coverage behind.
_OPUS_MT_TC_BIG_GROUP_MODELS = (
    "opus-mt-tc-big-cat_oci_spa-en",
    "opus-mt-tc-big-en-cat_oci_spa",
    "opus-mt-tc-big-itc-itc",
)


@pytest.mark.parametrize("model_id", _AINA_MARIAN_PAIRS)
def test_aina_pairs_are_registered_as_marian(model_id):
    assert model_id in TRANSLATE, f"{model_id} is still unregistered"
    assert TRANSLATE[model_id]["arch"] == "marian"
    assert "source_spm" in TRANSLATE[model_id]["side_files"]
    assert "target_spm" in TRANSLATE[model_id]["side_files"]


@pytest.mark.parametrize("model_id", _SOFTCATALA_PEGASUS_FAST)
def test_softcatala_exports_are_registered_as_pegasus_fast(model_id):
    assert model_id in TRANSLATE, f"{model_id} is still unregistered"
    assert TRANSLATE[model_id]["arch"] == "pegasus-fast"
    assert "tokenizer_json" in TRANSLATE[model_id]["side_files"]


@pytest.mark.parametrize("model_id", _OPUS_MT_TC_BIG_GROUP_MODELS)
def test_opus_mt_tc_big_group_models_are_registered(model_id):
    assert model_id in TRANSLATE, f"{model_id} is still unregistered"
    entry = TRANSLATE[model_id]
    assert entry["arch"] == "marian"
    assert not entry.get("pair"), (
        f"{model_id} is a group model; it must not be registered as a "
        f"single dedicated pair")
    # Every one of these needs *some* way to name its target: either a
    # multi-target token template, or (the one no-token exception) exactly
    # one fixed target.
    template = entry.get("target_token_template", "")
    single_fixed_target = (isinstance(entry.get("tgt_languages"), list)
                           and len(entry["tgt_languages"]) == 1)
    assert "{code}" in template or single_fixed_target, (
        f"{model_id} can select neither a token-based target nor a single "
        f"fixed one")


def test_opus_mt_tc_big_itc_itc_excludes_the_unverified_kea_token():
    """`kea` (Kabuverdianu) has a real `>>kea<<` vocab token but a real
    translation into it came back as fluent Spanish, not Kabuverdianu -
    confirmed with GlotLID. Registering it would repeat the MADLAD failure
    mode this project already fixed once.
    """
    entry = TRANSLATE.get("opus-mt-tc-big-itc-itc")
    if entry is None:
        pytest.skip("opus-mt-tc-big-itc-itc is not registered")
    assert "kea" not in entry["languages"]


def test_opus_mt_tc_big_cat_oci_spa_en_is_directional_not_multi_target():
    """No `>>xxx<<` token in this checkpoint's vocabulary at all: many
    sources, one fixed target, nothing to disambiguate."""
    entry = TRANSLATE.get("opus-mt-tc-big-cat_oci_spa-en")
    if entry is None:
        pytest.skip("opus-mt-tc-big-cat_oci_spa-en is not registered")
    assert entry.get("src_languages") and entry.get("tgt_languages") == ["en"]
    assert "target_token_template" not in entry


#: Two more many-source/one-fixed-target group models, found by a live
#: registry audit after the three above were resolved: same shape as
#: `opus-mt-tc-big-cat_oci_spa-en`, but reached through `_marian_pair`'s
#: `GroupModel` path (their family code - `gmq` Scandinavian, `zle`
#: East-Slavic - IS a recognised ISO 639-5 collection, unlike `cat_oci_spa`),
#: not the underscore-joined-name special case. Both resolve through the same
#: `_marian_group_entry` with no per-repo code.
_DIRECTIONAL_GROUP_MODELS_NO_TOKEN = ("opus-mt-gmq-en", "opus-mt-tc-big-zle-en")


@pytest.mark.parametrize("model_id", _DIRECTIONAL_GROUP_MODELS_NO_TOKEN)
def test_directional_group_models_with_no_token_are_registered(model_id):
    """A group model whose vocabulary has zero `>>xxx<<` tokens is not
    broken - it is many-to-one, with nothing to disambiguate. Generalises
    the `cat_oci_spa-en` case rather than special-casing it alone."""
    assert model_id in TRANSLATE, f"{model_id} is still unregistered"
    entry = TRANSLATE[model_id]
    assert entry["arch"] == "marian"
    assert not entry.get("pair")
    assert entry.get("tgt_languages") == ["en"]
    assert entry.get("src_languages")
    assert "target_token_template" not in entry


def test_m2m100_en_hau_finetune_is_registered_as_a_bilingual_pair():
    """Same shape as the other Masakhane m2m100_418M_*_rel_news_ft
    fine-tunes already in BILINGUAL_FINETUNES: card names exactly en/ha,
    and the export's own generation_config.json bakes in forced_bos_token_id
    - so no per-call target token is needed, same as its siblings."""
    entry = TRANSLATE.get("m2m100_418M_en_hau_rel_news_ft")
    if entry is None:
        pytest.skip("m2m100_418M_en_hau_rel_news_ft is not registered")
    assert entry["arch"] == "m2m100"
    assert entry["pair"] == ["en", "ha"]


# ---------------------------------------------------------------------------
# Unit tests for the arch-detection logic itself
# ---------------------------------------------------------------------------

def test_arch_prefers_dual_spm_over_model_type():
    """A dual source.spm/target.spm/vocab.json is Marian, whatever
    `config.model_type` claims - the exact `aina-translator-ca-*` mismatch."""
    detail = {"config": {"model_type": "m2m_100"}}
    files = {"source.spm": 1, "target.spm": 1, "vocab.json": 1}
    assert sync_registry._arch(detail, files) == "marian"


def test_arch_still_reads_m2m100_without_dual_spm():
    """A real M2M100 export (single merged sentencepiece.bpe.model, no
    source.spm/target.spm) must not be reclassified as Marian."""
    detail = {"config": {"model_type": "m2m_100"}}
    files = {"sentencepiece.bpe.model": 1, "vocab.json": 1}
    assert sync_registry._arch(detail, files) == "m2m100"


def test_arch_still_reads_nllb_without_vocab_json():
    detail = {"config": {"model_type": "m2m_100"}}
    files = {"sentencepiece.bpe.model": 1}
    assert sync_registry._arch(detail, files) == "nllb"


def test_arch_reads_pegasus_fast_from_tokenizer_json():
    """`model_type: pegasus` with a tokenizers-library tokenizer.json (no
    OpenNMT BPE vocab file) is Softcatalà's shape, not ProxectoNos'."""
    detail = {"config": {"model_type": "pegasus"}}
    files = {"tokenizer.json": 1}
    assert sync_registry._arch(detail, files) == "pegasus-fast"


def test_arch_still_reads_opennmt_bpe_variants():
    detail = {"config": {"model_type": "pegasus"}}
    assert sync_registry._arch(detail, {"onmt_vocab.json": 1}) == "opennmt-bpe"
    assert sync_registry._arch(detail, {"nos_vocab.json": 1}) == "opennmt-bpe"


def test_arch_refuses_unrecognised_pegasus_tokenizer_shape():
    detail = {"config": {"model_type": "pegasus"}}
    with pytest.raises(sync_registry.SkipRepo):
        sync_registry._arch(detail, {})


def test_marian_special_tokens_reads_both_json_shapes(tmp_path):
    """Standard opus-mt spells special tokens as plain strings; the
    `aina-translator-ca-*` family (a `tokenizers`-library export) wraps each
    in an object. Both have to be read the same way."""
    from linguonnx.translate.tokenizers import _marian_special_tokens

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"eos_token": "</s>", "pad_token": "<pad>",
                                "unk_token": "<unk>"}))
    assert sync_registry  # keep the module import used (silence linters)
    assert _marian_special_tokens(flat) == {
        "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({
        "eos_token": {"content": "</s>"},
        "pad_token": {"content": "<blank>"},
        "unk_token": {"content": "<unk>"},
    }))
    assert _marian_special_tokens(wrapped) == {
        "eos_token": "</s>", "pad_token": "<blank>", "unk_token": "<unk>"}

    assert _marian_special_tokens(None) == {}


# ---------------------------------------------------------------------------
# Routing smoke tests - no weights are downloaded, routing is pure JSON
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def translator():
    """The default graph: permissive int8. Lazy - nothing is downloaded."""
    return load_translator()


@pytest.mark.parametrize("src,tgt", [
    ("en", "pt"), ("pt", "eu"), ("gl", "ca"),
    # `en -> hi` alone dodges a whole class of bug: Hindi has a dedicated
    # opus-mt model, so it resolves even when every Indic-only model is
    # unroutable. The rest of the Indic set has no such cover.
    ("en", "hi"), ("en", "kn"), ("en", "ta"), ("en", "ur"), ("en", "gu"),
    ("en", "mr"), ("en", "ml"), ("en", "te"), ("en", "pa"),
    ("hi", "en"), ("ta", "en"), ("hi", "ta"),
])
def test_pair_resolves(translator, src, tgt):
    route = translator.route(src, tgt)
    assert route.hops, f"{src}->{tgt} produced an empty route"
    assert route.hops[0].src == src
    assert route.hops[-1].tgt == tgt


def test_pt_to_eu_never_routes_through_a_model_without_basque(translator):
    """The Basque leg must land on a model that has Basque, at every hop.

    A route may legitimately pivot (pt -> es -> eu); what it may not do is put
    `eu` on a hop served by M2M100, which does not know the language.
    """
    for route in translator.routes("pt", "eu"):
        for hop in route.hops:
            if "eu" in (hop.src, hop.tgt):
                entry = TRANSLATE[hop.model_id]
                assert entry["arch"] != "m2m100", (
                    f"route {route} puts Basque on {hop.model_id}, which is "
                    f"M2M100 and has no Basque")


def test_default_graph_is_permissively_licensed(translator):
    """Non-commercial models are available but never silently in the default."""
    for model_id, entry in translator.models.items():
        assert entry["license_tier"] != "non-commercial", (
            f"{model_id} is {entry['license']} and is in the default graph")


def test_noncommercial_models_exist_but_are_opt_in():
    tiers = {e["license_tier"] for e in TRANSLATE.values()}
    assert "non-commercial" in tiers, "NLLB should still be in the registry"
    opt_in = load_translator(include_noncommercial=True)
    assert len(opt_in.models) > len(load_translator().models)


def test_pt_to_mwl_resolves_in_one_hop(translator):
    """Mirandese: a wrong coverage claim would silently drop it.

    Two models carry `mwl`: MADLAD, and `opus-mt-itc-itc` - which was
    unreachable while the generator minted it as `pair: ["itc", "itc"]`,
    because no caller can ask for the Italic family. Now that it registers its
    real 16-language token set, it wins on cost (165 MB int8 against MADLAD's
    4945 MB), which is the point of fixing it.
    """
    route = translator.route("pt", "mwl")
    assert route.n_hops == 1
    assert route.hops[-1].model_id.startswith(
        ("madlad400-3b-mt", "opus-mt-itc-itc"))


def test_pt_to_kea_resolves_via_nllb():
    """Kabuverdianu: NLLB-only, and only reachable opted in (CC-BY-NC-4.0)."""
    translator = load_translator(include_noncommercial=True)
    route = translator.route("pt", "kea")
    assert route.hops[-1].model_id.startswith("nllb-600M")


def test_en_to_an_resolves_aragonese(translator):
    route = translator.route("en", "an")
    assert route.hops[-1].tgt == "an"


def test_hi_to_ta_resolves_direct(translator):
    """Hindi -> Tamil must not detour through English.

    It used to assert the hop was ``indictrans2-indic-indic``. IndicTrans2 has
    no inference pipeline here, so the registry marks it unrunnable and the
    router now serves the pair from a model that runs. What must hold either
    way is that one model does the pair, rather than a pivot through English
    compounding the error twice.
    """
    route = translator.route("hi", "ta")
    assert route.n_hops == 1, route
    assert TRANSLATE[route.hops[0].model_id].get("runnable") is not False


# ---------------------------------------------------------------------------
# Runnability: what is routed has to be what can be executed
# ---------------------------------------------------------------------------

#: Every architecture `linguonnx.translate.preprocess` registers a pipeline
#: for. Kept as a literal set rather than read from the registry, so that
#: adding an entry for an architecture nobody wrote a pipeline for fails here.
RUNNABLE_ARCHS = {"marian", "m2m100", "nllb", "madlad", "indictrans2",
                  "opennmt-bpe", "pegasus-fast"}


@pytest.mark.parametrize("model_id", sorted(TRANSLATE))
def test_unrunnable_entries_say_why(model_id):
    entry = TRANSLATE[model_id]
    if entry.get("runnable") is False:
        assert entry.get("unrunnable_reason", "").strip(), \
            f"{model_id} is unrunnable with no reason recorded"
    else:
        assert "unrunnable_reason" not in entry


@pytest.mark.parametrize("model_id", sorted(TRANSLATE))
def test_the_registry_flags_every_architecture_without_a_pipeline(model_id):
    """The flag and the pipeline registry have to agree.

    An entry whose architecture has no registered pipeline would be routable
    and would fail at the last possible moment, which is what the flag exists
    to prevent. The reverse is just as wrong: an architecture that *does* have
    a pipeline and is still flagged is a model the router refuses for no
    reason.
    """
    from linguonnx.translate.preprocess import pipeline_for

    entry = TRANSLATE[model_id]
    implemented = entry["arch"] in RUNNABLE_ARCHS
    if implemented:
        assert pipeline_for(entry["arch"]) is not None
    assert (entry.get("runnable") is not False) == implemented, (
        f"{model_id} is {entry['arch']} and its runnable flag disagrees with "
        f"what TranslationModel.translate implements")


@pytest.mark.parametrize("src,tgt", [
    ("en", "kn"), ("en", "as"), ("en", "mai"), ("en", "gu"), ("en", "mr"),
    ("en", "ta"), ("en", "te"), ("en", "ml"), ("en", "pa"), ("en", "ur"),
    ("en", "sd"), ("en", "ks"), ("en", "doi"), ("en", "brx"), ("en", "mni"),
    ("en", "sa"), ("en", "sat"), ("gl", "en"), ("en", "gl"), ("pt", "gl"),
])
def test_can_translate_never_promises_a_pair_translate_cannot_do(translator, src, tgt):
    """`can_translate` is asked *before* committing, so it must not lie.

    Every one of these pairs is claimed by a model with no inference pipeline.
    A `True` here followed by `NotImplementedError` two calls later is worse
    than a `False`, because the caller had asked.
    """
    if not translator.can_translate(src, tgt):
        # Refusing is always an honest answer; promising is what needs proof.
        with pytest.raises(NoRouteError):
            translator.route(src, tgt)
        return
    for hop in translator.route(src, tgt).hops:
        assert TRANSLATE[hop.model_id].get("runnable") is not False, (
            f"can_translate({src!r}, {tgt!r}) is True but the route runs "
            f"{hop.model_id}, which cannot be executed")


def test_available_languages_are_all_reachable_through_runnable_models(translator):
    """A language listed but served only by an unrunnable model is a lie too."""
    runnable = {normalize_tag(code)
                for model_id, entry in translator.models.items()
                if entry.get("runnable") is not False
                for code in (list(entry.get("languages", ()))
                             + list(entry.get("pair", ()))
                             + list(entry.get("src_languages") or ())
                             + list(entry.get("tgt_languages") or ()))}
    assert translator.available_languages <= runnable


def test_no_ranked_route_runs_a_model_that_cannot_be_executed(translator):
    for src, tgt in [("en", "ta"), ("hi", "ta"), ("en", "gl"), ("gl", "pt"),
                     ("en", "hi"), ("hi", "en")]:
        for route in translator.routes(src, tgt, limit=50):
            for hop in route.hops:
                assert TRANSLATE[hop.model_id].get("runnable") is not False, \
                    f"{route} runs {hop.model_id}, which cannot be executed"


def test_indictrans2_en_indic_never_serves_indic_to_english(translator):
    """en-indic is eng_Latn -> {indic}, one-way; the reverse must use indic-en."""
    model_id = next(m for m in TRANSLATE if m.startswith("indictrans2-en-indic"))
    entry = TRANSLATE[model_id]
    from linguonnx.translate.models import capability_from_entry

    cap = capability_from_entry(entry)
    assert cap.covers("en", "hi")
    assert not cap.covers("hi", "en")
    for route in translator.routes("hi", "en", limit=50):
        for hop in route.hops:
            assert not hop.model_id.startswith("indictrans2-en-indic"), (
                f"{route} runs {model_id} backwards")


def test_nos_coda_model_never_serves_its_reverse_direction(translator):
    """nos-coda_iacobus-en-gl is English -> Galician only, never gl -> en."""
    model_id = next(m for m in TRANSLATE if m.startswith("nos-coda_iacobus-en-gl"))
    entry = TRANSLATE[model_id]
    assert tuple(entry["pair"]) == ("en", "gl")
    for route in translator.routes("gl", "en", limit=50):
        for hop in route.hops:
            assert not hop.model_id.startswith("nos-coda_iacobus-en-gl"), (
                f"{route} runs {model_id} backwards")


def test_general_default_is_m2m100_418m_int8():
    """The default general-coverage model stays MIT-licensed and small."""
    entry = TRANSLATE["m2m100-418M-int8"]
    assert entry["license_tier"] == "permissive"
    assert entry["precision"] == "int8"
    assert entry["arch"] == "m2m100"


def test_glotlid_is_the_lid_default():
    from linguonnx.detect import DEFAULT_MODEL_ID

    assert DEFAULT_MODEL_ID.startswith("glotlid")
    assert LID[DEFAULT_MODEL_ID]["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_sync_script_is_executable_and_parses():
    result = subprocess.run([sys.executable, str(SYNC_SCRIPT), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.network
def test_committed_registry_matches_the_hub():
    """``--check`` is the CI guard: exit 0 only when the JSON is current
    AND every translate/lid repo on the Hub resolves into it.

    Both directions matter. A stale committed JSON (Hub moved, JSON did not)
    is a content diff and always failed here. A repo that Hub-side extraction
    silently skips on *every* run never shows up as a diff at all - generated
    and committed agree, both are missing it - so `--check` used to report
    "up to date" while 43 published models were absent from the registry.
    `sync_registry.py --check` now also fails when it skips a translate/lid
    repo that is not in its documented allowlist, which is what actually
    catches that direction.
    """
    result = subprocess.run([sys.executable, str(SYNC_SCRIPT), "--check"],
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, (
        "registry has drifted from the Hub, or a Hub model is being silently "
        "skipped outside the documented allowlist; re-run "
        f"scripts/sync_registry.py\n{result.stdout}\n{result.stderr}")


def test_registry_json_is_stable_and_sorted():
    """Sorted keys, or every regeneration makes a diff nobody can review."""
    for kind in ("translate", "lid"):
        path = REPO_ROOT / "linguonnx" / "model_index" / f"{kind}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert list(raw) == sorted(raw), f"{kind}.json entries are not sorted"
        for model_id, entry in raw.items():
            assert list(entry) == sorted(entry), \
                f"{kind}.json: keys of {model_id} are not sorted"


# --------------------------------------------------------------------------
# A Marian export whose vocab.json cannot spell its own source language
# --------------------------------------------------------------------------

class TestTheEnKoClaimIsWithdrawn:
    """`Helsinki-NLP/opus-mt-tc-big-en-ko` cannot read English.

    Its `vocab.json` is the *target* (Korean) vocabulary: only 21% of the
    pieces `source.spm` produces have an id, so `doctor`, `patient`,
    `recover` and `weeks` all arrive at the encoder as `<unk>` and the
    decoder answers fluent Korean nonsense
    (`'process 잘 모기 댓글 upon9-2.'`). `transformers` builds the identical
    input ids from the identical files, so this is an upstream export defect
    that no inference change here can repair - the claim goes instead.
    """

    def test_the_registry_no_longer_claims_the_pair(self):
        registry = json.loads((REPO_ROOT / "linguonnx" / "model_index"
                               / "translate.json").read_text(encoding="utf-8"))
        assert not [m for m in registry if m.startswith("opus-mt-en-ko")]

    def test_the_reason_is_recorded_rather_than_forgotten(self):
        skipped = json.loads((REPO_ROOT / "linguonnx" / "model_index"
                              / "skipped.json").read_text(encoding="utf-8"))
        reason = skipped.get("TigreGotico/opus-mt-en-ko-onnx")
        assert reason and "source.spm" in reason

    def test_english_to_korean_still_routes(self):
        """Withdrawing a broken claim must not withdraw the language.

        The replacement was measured, not assumed: `m2m100-418M-int8` scores
        **chrF 30.4** on `en->ko`, FLORES-200 devtest, n=100, beam4 - the same
        corpus, metric and decoding mode every `quality` number in the
        registry uses. It is in the band of the measured entries there
        (`opus-mt-az-en` 25.9, `m2m100_418M_en_hau_rel_news_ft` 37.9). The
        number is not written into `m2m100-418M-int8`'s `quality` field
        because that field is a whole-model claim and this is one pair out of
        that model's ~9900.
        """
        translator = load_translator()
        assert translator.route("en", "ko").hops


def _tiny_spm_blob(tmp_path, words, name):
    """A real SentencePiece model over ``words``, as serialized bytes."""
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / f"{name}.txt"
    corpus.write_text("\n".join(" ".join(words) for _ in range(80)),
                      encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(tmp_path / name),
        vocab_size=32, hard_vocab_limit=False,
        character_coverage=1.0, pad_id=1, eos_id=2, unk_id=0, bos_id=-1)
    return (tmp_path / f"{name}.model").read_bytes()


def _pieces_of(blob):
    spm = pytest.importorskip("sentencepiece")
    processor = spm.SentencePieceProcessor()
    processor.LoadFromSerializedProto(blob)
    return [processor.IdToPiece(i) for i in range(processor.get_piece_size())]


class TestTheSourceCoverageFloorIsEnforced:
    """The check itself, on synthetic exports, with no network.

    Asserting the constant is 0.9 asserts nothing: replacing the call site
    with `pass` left the whole suite green. These feed the function a matched
    export and a mismatched one and assert what it *does*.
    """

    FILES = {"vocab.json": 1, "source.spm": 1}

    @pytest.fixture
    def sync(self):
        """One module instance per test: `SkipRepo` from two separate
        `exec_module` calls are two different classes, and `pytest.raises`
        would never match."""
        return _load_sync_registry()

    def _run(self, sync, monkeypatch, vocab_pieces, blob):
        monkeypatch.setattr(sync, "raw_file",
                            lambda repo_id, path: json.dumps(
                                {piece: i for i, piece in enumerate(vocab_pieces)}))
        sync._marian_source_coverage("Some/repo", self.FILES,
                                     fetch=lambda repo_id: blob)

    def test_a_matched_export_is_accepted(self, sync, monkeypatch, tmp_path):
        blob = _tiny_spm_blob(tmp_path, ["hello", "world", "doctor", "patient"],
                              "matched")
        self._run(sync, monkeypatch, _pieces_of(blob), blob)

    def test_an_export_whose_vocab_cannot_spell_its_source_is_refused(
            self, sync, monkeypatch, tmp_path):
        """`opus-mt-tc-big-en-ko` in miniature: the vocab is the other side."""
        blob = _tiny_spm_blob(tmp_path, ["hello", "world", "doctor", "patient"],
                              "source")
        other = _tiny_spm_blob(tmp_path, ["안녕", "세계", "의사", "환자"], "target")
        pieces = _pieces_of(blob)
        vocab = _pieces_of(other)
        coverage = len(set(pieces) & set(vocab)) / len(set(pieces))
        assert coverage < 0.5, f"the synthetic mismatch is not one: {coverage}"
        with pytest.raises(sync.SkipRepo) as err:
            self._run(sync, monkeypatch, vocab, blob)
        assert "source.spm" in str(err.value)

    def test_the_floor_is_the_line_that_is_actually_applied(
            self, sync, monkeypatch, tmp_path):
        """Just under the floor is refused; just over it is accepted.

        The floor is read from the module, so moving it moves both sides of
        this test together and neither assertion can rot into a tautology.
        """
        blob = _tiny_spm_blob(tmp_path, ["hello", "world", "doctor", "patient"],
                              "floor")
        pieces = sorted(set(_pieces_of(blob)))
        floor = sync._MARIAN_SOURCE_COVERAGE_FLOOR
        over = pieces[:len(pieces) - int(len(pieces) * (1 - floor) / 2)]
        assert len(over) / len(pieces) > floor
        self._run(sync, monkeypatch, over, blob)
        under = pieces[:int(len(pieces) * (floor - 0.05))]
        assert len(under) / len(pieces) < floor
        with pytest.raises(sync.SkipRepo):
            self._run(sync, monkeypatch, under, blob)

    def test_a_network_failure_is_a_skip_not_a_crashed_sync(
            self, sync, monkeypatch):
        """Every other failure mode here takes the SkipRepo path; so does this."""
        monkeypatch.setattr(sync, "raw_file",
                            lambda repo_id, path: json.dumps({"a": 0}))

        def _boom(repo_id):
            raise OSError("connection reset by peer")

        with pytest.raises(sync.SkipRepo) as err:
            sync._marian_source_coverage("Some/repo", self.FILES, fetch=_boom)
        assert "connection reset by peer" in str(err.value)

    def test_an_export_missing_a_side_file_is_left_to_the_other_checks(self, sync):
        """No `source.spm` in the listing is not this check's business."""
        sync._marian_source_coverage("Some/repo", {"vocab.json": 1},
                                     fetch=lambda repo_id: b"")


class TestTheEnKoCheckIsWiredIn:

    def test_the_generator_runs_the_check_for_every_marian_repo(self):
        """The mutation that stayed green: the call site itself."""
        source = SYNC_SCRIPT.read_text(encoding="utf-8")
        body = source[source.index("def translate_entries("):]
        assert "_marian_source_coverage(repo_id, files)" in body
