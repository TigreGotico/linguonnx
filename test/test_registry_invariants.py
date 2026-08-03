"""Invariants on what ``scripts/sync_registry.py`` is allowed to emit.

``test_registry_sync.py`` guards the *artefact*: it reads the committed JSON
and checks that the claims in it are servable. This file guards the
*generator*, from the other side. The registry is produced by a script, so
every wrong entry the deep data audit found was not a bad hand-edit - it was a
missing assertion on the script's output:

``opus-mt-itc-itc``
    Registered as ``pair: ["itc", "itc"]``. ``itc`` is an ISO 639-5
    *collection* - the Italic family - so no caller can ask for it and 645 MB
    of a working Romance model was unreachable. Worse, the mandatory
    ``>>ita<<``-style target token was never emitted, because the generator
    only demanded one when the base model's target differed from the repo's,
    and here both sides said ``itc``; and the multi-target safety net in
    ``entry_runnability`` never fired, because it only refuses a Marian entry
    with *no* pair. Guarded by :func:`test_no_collection_or_special_codes` and
    :func:`test_a_pair_is_two_different_languages`.

``m2m100-418M-smugri``
    A TartuNLP Finno-Ugric fine-tune, published claiming 104 languages -
    including ``th -> sw`` - because the fine-tune keeps M2M100's whole token
    inventory and the generator read the covering set off the tokenizer. Its
    own card names eight languages. Guarded by
    :func:`test_languages_stay_inside_the_card_s_claim`.

``madlad400-3b-mt``
    19.7 GB in fp32, over the 8 GB download budget - so ``can_translate``
    answered True for the 270 languages only MADLAD serves and ``translate``
    then raised ``DownloadTooLargeError``. Guarded by
    :func:`test_routable_models_stay_inside_the_download_budget`.

``m2m100_418M_{bam,bbj,fon}_fr_rel_news_ft`` (and their reverse ``fr_*``
siblings, plus ``mos_fr``)
    linguonnx#56 fixed the *shape* of a bilingual entry with no ``languages``
    list - read the codes from the export's own ``special_tokens_map.json``
    rather than build an empty language-token block - and added a registry
    test for it: :func:`test_translate_native_codes.test_language_token_models_can_build_a_language_block`.
    That test only checks that ``side_files`` names a
    ``special_tokens_map.json`` to fetch. It does not check what that file
    actually says. Masakhane's Bambara/Ghomala/Fon/Mossi fine-tunes keep
    ``facebook/m2m100_418M``'s original 100-language token set unchanged and
    reuse an existing one - ``__sw__`` - as the model's own stand-in for the
    language it does not have a token for (the export's README says so:
    ``tokenizer.src_lang = "sw"`` to mean Bambara). Nothing in the registry
    records that reuse, so ``native_code("bam")`` answers ``"bam"``, which is
    not in the file at all, and ``lang_id`` raises for every request. The
    PR's own test passed because the file it checked for *exists* - the 100
    codes in it are just never the three the entry claims to serve. Guarded
    by :func:`test_masakhane_reused_tokens_are_declared_in_native_codes`,
    against the export's actual (vendored, not re-downloaded)
    ``special_tokens_map.json``.

Everything here is offline. The one fact that is not derivable from the
committed JSON - what each repo's Hub card declares - is cached in
``test/data/hub_card_languages.json``, refreshed by the same sync that writes
the registry.
"""

import json
from pathlib import Path

import langcodes
import pytest

from linguonnx import model_manager
from linguonnx.detect.labels import tag_scope
from linguonnx.model_manager import DEFAULT_MAX_DOWNLOAD_MB
from linguonnx.translate import _select_entries
from linguonnx.translate.graph import TranslationGraph, normalize_tag
from linguonnx.translate.models import capability_from_entry
from linguonnx.translate.tokenizers import artifact_lang_codes

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD_LANGUAGES = json.loads(
    (Path(__file__).parent / "data" / "hub_card_languages.json")
    .read_text(encoding="utf-8"))

TRANSLATE = model_manager.list_models(kind="translate")

#: Keys whose values are language codes. ``pair`` is ordered (src, tgt);
#: the rest are sets.
CODE_KEYS = ("pair", "languages", "src_languages", "tgt_languages")


def _codes(entry: dict):
    """Every language code one entry states, with the key it came from."""
    for key in CODE_KEYS:
        for code in entry.get(key) or ():
            yield key, code


def _entries():
    return sorted(TRANSLATE.items())


# ---------------------------------------------------------------------------
# 1. A language family is not a language
# ---------------------------------------------------------------------------

def test_no_collection_or_special_codes():
    """No entry claims coverage of an ISO 639-5 collection or an IANA special code.

    ``itc``, ``sla``, ``roa``, ``gem``, ``bnt``, ``dra``, ``cpf``, ``bat`` are
    *families*; ``mul``, ``mis``, ``und``, ``zxx`` are placeholders. A caller
    asks for a language, so a node named after a family is unreachable - and
    an unreachable node hides the fact that the model behind it needs a target
    token to say anything at all.

    The repo-name regex in ``_marian_pair`` accepts all of them
    (``[a-z]{2,3}``), and the org is actively converting ``opus-mt-mul-en``,
    ``sla-en``, ``dra-en``, ``bnt-en``, ``bat-en``, ``cpf-en`` and
    ``gem-gem``, so this is a fence, not a monument.
    """
    offenders = [
        f"{model_id}.{key} = {code!r} ({tag_scope(code)})"
        for model_id, entry in _entries()
        for key, code in _codes(entry)
        if tag_scope(code) is not None
    ]
    assert not offenders, (
        "these entries name a language family or placeholder instead of a "
        "language:\n  " + "\n  ".join(offenders))


# ---------------------------------------------------------------------------
# 2. A pair is two different languages
# ---------------------------------------------------------------------------

def test_a_pair_is_two_different_languages():
    """``pair[0] != pair[1]``.

    A self-pair is never a real claim. It is what a group model degenerates
    into when the generator reads a family code off both sides of the repo
    name, and it routes: ``Capability.covers`` compared the tuple without a
    same-language guard, so ``itc -> itc`` was a valid one-hop route that
    would emit whichever Romance language the decoder felt like.
    """
    offenders = [f"{model_id}: {entry['pair']}"
                 for model_id, entry in _entries()
                 if entry.get("pair") and entry["pair"][0] == entry["pair"][1]]
    assert not offenders, (
        "a bilingual entry whose two sides are the same language is a group "
        "model in disguise:\n  " + "\n  ".join(offenders))


def test_covers_refuses_a_same_language_hop():
    """The graph refuses ``src == tgt`` for a bilingual capability too.

    The registry assertion above is the data fence; this is the code fence
    behind it. ``covers`` guarded the multilingual branch against ``src ==
    tgt`` and not the bilingual one, so a self-pair that ever reached the
    graph became a real edge.
    """
    entry = dict(next(e for e in TRANSLATE.values() if e.get("pair")))
    entry["pair"] = [entry["pair"][0], entry["pair"][0]]
    capability = capability_from_entry(entry)
    tag = capability.pair[0]
    assert not capability.covers(tag, tag)


# ---------------------------------------------------------------------------
# 3. Coverage stays inside what the publisher claimed
# ---------------------------------------------------------------------------

def _declared(hf_repo: str):
    """The card's language list, normalised, or None when it names none."""
    raw = CARD_LANGUAGES.get(hf_repo)
    if not isinstance(raw, list) or not raw or "multilingual" in raw:
        return None
    return {normalize_tag(str(code)) for code in raw}


def test_card_language_cache_covers_every_registered_repo():
    """The offline card cache must not silently go stale.

    A repo missing from the cache would make the subset test below skip
    exactly the entry it exists to catch.
    """
    missing = sorted({entry["hf_repo"] for entry in TRANSLATE.values()}
                     - set(CARD_LANGUAGES))
    assert not missing, (
        "test/data/hub_card_languages.json is missing "
        f"{len(missing)} registered repo(s); re-run scripts/sync_registry.py "
        "and refresh it:\n  " + "\n  ".join(missing))


#: Entries whose coverage is deliberately *not* bounded by their Hub card,
#: with the reason. The card is a good upper bound for a fine-tune of a
#: multilingual base, which is what this test is for; it is not authoritative
#: everywhere.
_CARD_IS_NOT_THE_BOUND = {
    # The TigreGotico export cards for these are written by the export script
    # and collapse script variants: they name 23 languages where the model's
    # own tag set has 26, distinguishing `kas_Arab`/`kas_Deva`,
    # `mni_Beng`/`mni_Mtei` and `snd_Arab`/`snd_Deva`. The tag set is quoted
    # from AI4Bharat's own card in `INDICTRANS2_TAGS`, with the citation.
    "indictrans2": "hand-verified tag set, cited in INDICTRANS2_TAGS",
    # opus-mt group models take their coverage from the `>>xxx<<` tokens in
    # their own vocab.json, which is what actually selects a target. The
    # export card disagrees in both directions - `opus-mt-itc-itc`'s card
    # lists `rm`/`sc`/`co` that have no token, omits `ast`/`mwl`/`pap` that
    # do, and lists the family code `itc` itself as if it were a language.
    "marian-group": "coverage read from the model's own >>xxx<< token set",
}


def _card_bound_exempt(entry: dict):
    if entry["arch"] == "indictrans2":
        return _CARD_IS_NOT_THE_BOUND["indictrans2"]
    if entry.get("target_token_template", "").startswith(">>"):
        return _CARD_IS_NOT_THE_BOUND["marian-group"]
    return None


def test_languages_stay_inside_the_card_s_claim():
    """No entry claims more languages than its own Hub card names.

    A fine-tune of a multilingual base keeps the base's whole tokenizer, so
    reading the covering set off ``special_tokens_map.json`` publishes the
    base's coverage for weights that no longer have it. The old guard used a
    count threshold - "three languages or fewer means fine-tune" - which
    ``m2m100-418M-smugri`` walked straight past with eight, publishing 104.

    A count cannot tell a narrow fine-tune from a general model. A subset test
    can, and it needs no threshold: a general model declares ``multilingual``
    (or nothing) and is not checked at all.
    """
    offenders = []
    for model_id, entry in _entries():
        declared = _declared(entry["hf_repo"])
        if declared is None or _card_bound_exempt(entry):
            continue
        claimed = {normalize_tag(code) for _, code in _codes(entry)}
        extra = sorted(claimed - declared)
        if extra:
            offenders.append(
                f"{model_id}: card names {len(declared)} languages, the entry "
                f"claims {len(claimed)}; {len(extra)} unclaimed "
                f"(e.g. {', '.join(extra[:5])})")
    assert not offenders, (
        "these entries claim coverage their publisher never claimed - a "
        "fine-tune wearing its base's token inventory:\n  "
        + "\n  ".join(offenders))


# ---------------------------------------------------------------------------
# 4. One code system
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="the languages-list generation path (MADLAD/NLLB/IndicTrans2 "
           "'languages'/'src_languages'/'tgt_languages') is now normalised "
           "at generation time, but two other code paths are not: (1) the "
           "single-pair NLLB fine-tunes (aina-*) still store 'pair' in raw "
           "FLORES codes ('spa_Latn', 'arn_Latn', 'arg_Latn', 'ast_Latn'); "
           "(2) m2m100/opus-mt 'pair'/'languages' still carry the model's "
           "own vocabulary spelling ('ns', 'bam', 'ewe', 'sh', 'tl') which "
           "MODEL_CODE_ALIASES/langcodes resolve away from at lookup time "
           "but which are never rewritten in the committed JSON. Turns "
           "green when sync_registry.py normalises those two paths too.",
    strict=False)
def test_every_code_is_already_normalised():
    """Registry codes are byte-identical to ``normalize_tag`` of themselves.

    The graph names its nodes with ``normalize_tag``. A code that survives
    normalisation unchanged is reachable; one that does not creates a node the
    caller's own tag never resolves to, which reads as "unsupported language"
    for a model that covers it.

    ``native_codes`` is the sanctioned escape hatch: it maps the normalised
    tag back to the model's own spelling, so the model still gets ``>>ita<<``
    while the graph only ever sees ``it``.
    """
    offenders = [
        f"{model_id}.{key}: {code!r} normalises to {normalize_tag(code)!r}"
        for model_id, entry in _entries()
        for key, code in _codes(entry)
        if normalize_tag(code) != code
    ]
    assert not offenders, (
        "the registry has to speak one code system:\n  " + "\n  ".join(offenders))


# ---------------------------------------------------------------------------
# 5. No node shadows another
# ---------------------------------------------------------------------------

#: Node pairs where one tag *is* the other's macrolanguage or its bare base,
#: and both are kept on purpose. Every entry states why.
_SHADOWING_ALLOWED = {
    # Serbian and Chinese are the two languages `_ALWAYS_KEEP_SCRIPT` exists
    # for: the script subtag is informative to a reader, not redundant, so the
    # scripted and unscripted nodes are genuinely different targets.
    ("sr", "sr-Cyrl"),
    ("sr", "sr-Latn"),
    ("zh", "zh-Hans"),
    ("zh", "zh-Hant"),
}


def _shadowing_pairs():
    nodes = {code for entry in TRANSLATE.values() for _, code in _codes(entry)}
    for node in sorted(nodes):
        base = node.split("-")[0]
        if base != node and base in nodes:
            yield base, node
            continue
        try:
            macro = langcodes.Language.get(node).macrolanguage
        except Exception:
            macro = None
        if macro and macro in nodes:
            yield macro, node


@pytest.mark.xfail(
    reason="the hyphen/underscore + known-script-default work landed and "
           "closed the pairs it targeted (crh, ko, acm/acq/ajp/azb, tzm), "
           "but #47's registry regeneration added ~43 base models (MADLAD, "
           "IndicTrans2, more NLLB) this PR never had in scope, and they "
           "surface ~30 new base/base-Script pairs (e.g. ace/ace-Arab, "
           "bjn/bjn-Arab, sat/sat-Beng, mni/mni-Mtei, sd/sd-Deva, "
           "la/la-Grek, orv/orv-Cyrl, taq/taq-Tfng, nds/nds-NL). Some are "
           "genuinely distinct varieties that belong in _SHADOWING_ALLOWED "
           "(the romanised MADLAD targets bg-Latn/el-Latn/bn-Latn/gom-Latn/"
           "ru-Latn/... and the regional az-RU/fa-AF/fr-CA already are, by "
           "design); others are a redundant script CLDR/langcodes failed to "
           "drop and belong in _KNOWN_SCRIPT_DEFAULTS. Each needs the same "
           "one-by-one verification crh/ko/acm got - a follow-up, not a "
           "hand-wave into _SHADOWING_ALLOWED.",
    strict=False)
def test_no_node_shadows_another():
    """No graph node is another node's macrolanguage or bare-base sibling.

    Two nodes for one language split the graph: a caller who asks for ``ar``
    never reaches a model that registered ``arb``, and ``available_languages``
    lists both as if they were separate languages. Anything genuinely
    different lives in ``_SHADOWING_ALLOWED`` with its reason.
    """
    offenders = [f"{a} shadows {b}" for a, b in _shadowing_pairs()
                 if (a, b) not in _SHADOWING_ALLOWED]
    assert not offenders, (
        "two graph nodes name one language:\n  " + "\n  ".join(offenders)
        + "\nEither normalise the code at generation time or add the pair to "
          "_SHADOWING_ALLOWED with a reason.")


# ---------------------------------------------------------------------------
# 6. Routing agrees with the download budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ["int8", "fp32", None])
def test_routable_models_stay_inside_the_download_budget(precision):
    """The default graph never routes a model the download path will refuse.

    ``size_mb`` used to reach routing only as a tie-breaker in the cost tuple,
    while the budget was enforced later, in ``ensure_model_files``. So with
    ``precision="fp32"`` the graph planned a hop through the 19.7 GB
    ``madlad400-3b-mt``, ``can_translate`` answered True for the 270 languages
    only MADLAD serves, and ``translate`` raised ``DownloadTooLargeError`` -
    the same lying-``can_translate`` failure this library already fixed for
    unrunnable architectures.

    ``count_cached_as_free=False`` so the answer does not depend on what
    happens to be in the developer's cache.
    """
    entries = _select_entries(precision, include_noncommercial=True, models=None)
    graph = TranslationGraph(
        [capability_from_entry(e) for e in entries.values()],
        count_cached_as_free=False)
    excluded = {cap.model_id for cap in graph.oversized_capabilities}
    excluded |= {cap.model_id for cap in graph.unrunnable_capabilities}
    still_routed = sorted(
        cap.model_id for cap in graph.capabilities
        if cap.size_mb > DEFAULT_MAX_DOWNLOAD_MB and cap.model_id not in excluded)
    assert not still_routed, (
        f"models over the {DEFAULT_MAX_DOWNLOAD_MB} MB download budget are in "
        f"the default {precision or 'all-precision'} routing index; "
        f"can_translate would promise a route translate cannot fetch:\n  "
        + "\n  ".join(still_routed))
    assert graph.languages, "the budget left nothing routable at all"


def test_the_budget_comment_matches_the_data():
    """The constant is documented against the registry it actually guards.

    This started as a comment claiming "the biggest registry model is ~7.4 GB,
    so the default refuses nothing" - true when it was written, false for
    years afterwards. The assertion is the part a comment cannot do.
    """
    biggest = max(entry["size_mb"] for entry in TRANSLATE.values())
    assert biggest > DEFAULT_MAX_DOWNLOAD_MB, (
        "the registry no longer holds a model over the download budget, so "
        "the budget-aware routing this guards is untested; keep the test but "
        "revisit the constant's docstring in linguonnx/model_manager.py")


# ---------------------------------------------------------------------------
# 7. Big graphs carry their external weights
# ---------------------------------------------------------------------------

#: ONNX is protobuf, and protobuf refuses a message over 2 GiB. A graph past
#: that limit has to be exported with its weights in a sibling
#: ``*.onnx_data`` blob, which ONNX Runtime finds by relative path. An entry
#: that is over the limit and lists no such blob either was exported wrong or
#: had its blob dropped from the entry - and the second one downloads cleanly
#: and then fails at session build, on the caller's machine.
#:
#: ``size_mb`` is the whole entry, not one graph, and an entry always carries
#: three graphs (encoder, decoder, decoder-with-past). So the threshold that
#: *proves* at least one graph is over the limit is three times it; anything
#: between the two is possible either way and is not asserted here.
_PROTOBUF_LIMIT_MB = 2048
_UNAMBIGUOUSLY_OVER_MB = 3 * _PROTOBUF_LIMIT_MB


def test_big_models_ship_their_external_weight_blobs():
    offenders = [
        f"{model_id}: {entry['size_mb']} MB, extra_files is empty"
        for model_id, entry in _entries()
        if entry["size_mb"] > _UNAMBIGUOUSLY_OVER_MB
        and not entry.get("extra_files")
    ]
    assert not offenders, (
        "a model past the 2 GiB protobuf limit must reference its "
        "*.onnx_data blobs:\n  " + "\n  ".join(offenders))


def test_external_blobs_belong_to_a_declared_graph():
    """Every ``extra_files`` blob is ``<some declared graph>_data``.

    A blob with no graph is a file the entry downloads and never uses; a graph
    whose blob was dropped is a session build that fails after the download
    succeeded.
    """
    offenders = []
    for model_id, entry in _entries():
        graphs = set((entry.get("graphs") or {}).values())
        for blob in entry.get("extra_files") or ():
            if not blob.endswith("_data") or blob[:-len("_data")] not in graphs:
                offenders.append(f"{model_id}: {blob} matches no declared graph")
    assert not offenders, "\n  ".join(offenders)


# --------------------------------------------------------------------------
# Masakhane's reused-token fine-tunes: a real content check, not a
# key-exists-in-metadata check.
# --------------------------------------------------------------------------

#: Vendored, not re-downloaded: every Masakhane ``m2m100_418M_*_rel_news_ft``
#: entry ships the exact same unmodified ``facebook/m2m100_418M``
#: ``special_tokens_map.json`` (byte-identical across the ten fine-tunes -
#: none of them adds a token for the language it was fine-tuned on). A single
#: snapshot is therefore representative of all of them, and checking it needs
#: no network.
_MASAKHANE_BASE_SPECIAL_TOKENS = json.loads(
    (Path(__file__).parent / "data"
     / "m2m100_masakhane_base_special_tokens_map.json")
    .read_text(encoding="utf-8"))
_MASAKHANE_BASE_CODES = set(
    artifact_lang_codes({"special_tokens_map":
                         Path(__file__).parent / "data"
                         / "m2m100_masakhane_base_special_tokens_map.json"}))

_MASAKHANE_RELNEWS = sorted(
    model_id for model_id, entry in _entries()
    if entry.get("provenance_org") == "Masakhane"
    and entry.get("arch") == "m2m100"
    and model_id.endswith("_rel_news_ft"))


def test_the_masakhane_fixture_still_matches_what_these_entries_ship():
    """Guards the test below from passing because the fixture went stale.

    If a future export of one of these fine-tunes actually adds its own
    language token, the fixture must be refreshed - this only checks that
    at least one such entry currently exists to test against.
    """
    assert _MASAKHANE_RELNEWS


@pytest.mark.parametrize("model_id", _MASAKHANE_RELNEWS)
def test_masakhane_reused_tokens_are_declared_in_native_codes(model_id):
    """``native_code`` must answer with a token the export's vocabulary has.

    linguonnx#56's registry test for this shape
    (``test_language_token_models_can_build_a_language_block``) only checks
    that ``side_files`` names a ``special_tokens_map.json`` to fetch - not
    that the file, once fetched, contains a token for the language the entry
    claims to serve. These ten fine-tunes all keep the unmodified base
    M2M100 vocabulary and reuse an existing token as a stand-in (the
    ``TigreGotico/m2m100_418M_bam_fr_rel_news_ft-onnx`` card: `"src_lang is
    'sw', which is *not* the ISO code of Bambara"`); the registry has to
    record that reuse as a ``native_codes`` override, or every call raises.
    """
    entry = TRANSLATE[model_id]
    pair = entry.get("pair") or ()
    native_codes = entry.get("native_codes") or {}
    for iso in pair:
        native = native_codes.get(iso, iso)
        assert native in _MASAKHANE_BASE_CODES, (
            f"{model_id}: {iso!r} resolves to {native!r}, which is not one "
            f"of the {len(_MASAKHANE_BASE_CODES)} tokens this fine-tune's "
            f"own special_tokens_map.json actually carries - every call with "
            f"{iso!r} raises. Needs a native_codes override (see the export's "
            f"Hub card for which existing token it reuses).")
