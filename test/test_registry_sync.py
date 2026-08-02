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

import json
import subprocess
import sys
from pathlib import Path

import pytest

from linguonnx import model_manager
from linguonnx.translate import load_translator
from linguonnx.translate.graph import LICENSE_TIERS, normalize_tag

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_registry.py"

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
        "marian", "m2m100", "nllb", "madlad", "indictrans2", "opennmt-bpe")
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
            assert entry["arch"] in ("marian", "nllb", "madlad"), \
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
        if not base or entry["arch"] != "marian":
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


def test_target_tokens_are_only_set_where_they_are_needed():
    for model_id, entry in TRANSLATE.items():
        if "target_token" not in entry:
            continue
        assert entry["arch"] == "marian", \
            f"{model_id} sets target_token but is {entry['arch']}"


# ---------------------------------------------------------------------------
# Routing smoke tests - no weights are downloaded, routing is pure JSON
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def translator():
    """The default graph: permissive int8. Lazy - nothing is downloaded."""
    return load_translator()


@pytest.mark.parametrize("src,tgt", [("en", "pt"), ("pt", "eu"),
                                     ("gl", "ca"), ("en", "hi")])
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


def test_pt_to_mwl_resolves_via_madlad(translator):
    """Mirandese: only MADLAD carries it; a wrong claim would silently drop it."""
    route = translator.route("pt", "mwl")
    assert route.hops[-1].model_id.startswith("madlad400-3b-mt")


def test_pt_to_kea_resolves_via_nllb():
    """Kabuverdianu: NLLB-only, and only reachable opted in (CC-BY-NC-4.0)."""
    translator = load_translator(include_noncommercial=True)
    route = translator.route("pt", "kea")
    assert route.hops[-1].model_id.startswith("nllb-600M")


def test_en_to_an_resolves_aragonese(translator):
    route = translator.route("en", "an")
    assert route.hops[-1].tgt == "an"


def test_hi_to_ta_resolves_direct_via_indictrans2_indic_indic(translator):
    """Hindi -> Tamil must not detour through English.

    indictrans2-indic-indic is the only model with both languages on the
    Indic side; routing it through en-indic + indic-en would silently add a
    pivot hop this direct model makes unnecessary.
    """
    route = translator.route("hi", "ta")
    assert route.n_hops == 1, route
    assert route.hops[0].model_id.startswith("indictrans2-indic-indic")


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
    """``--check`` is the CI guard: exit 0 only when the JSON is current."""
    result = subprocess.run([sys.executable, str(SYNC_SCRIPT), "--check"],
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, (
        "registry has drifted from the Hub; re-run "
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
