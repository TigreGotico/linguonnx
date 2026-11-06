"""Specialist-provenance tie-break: the router prefers the institution that
*owns* a language over a generalist that merely covers it, but only as a
tie-break - never overriding hop count, dedicated-vs-multilingual, or an
explicit caller-supplied route.
"""

import pytest

from linguonnx.translate.graph import (Capability, Hop, Route,
                                       TranslationGraph, is_specialist_for)

# --- fixtures ---------------------------------------------------------------


def specialist(model_id, pair, org, license_tier="permissive", size_mb=300,
              release_date=None):
    return Capability(model_id=model_id, arch="marian", license="Apache-2.0",
                      license_tier=license_tier, size_mb=size_mb, pair=pair,
                      provenance_org=org, release_date=release_date)


def generalist(model_id, languages, size_mb=1200, release_date=None):
    return Capability(model_id=model_id, arch="m2m100", license="MIT",
                      license_tier="permissive", size_mb=size_mb,
                      languages=frozenset(languages), release_date=release_date)


HITZ_ES_EU = specialist("mt-hitz-es-eu", ("es", "eu"), "HiTZ")
GENERIC_ES_EU = generalist("generic-multi", {"es", "eu", "en"})

NOS_ES_AST = specialist("nos-mt-es-ast", ("es", "ast"), "Proxecto Nós")
AINA_ES_AST = specialist("aina-es-ast", ("es", "ast"), "Projecte AINA")

AINA_ES_AN = specialist("aina-es-an", ("es", "an"), "Projecte AINA")
NOS_ES_AN = specialist("nos-mt-es-an", ("es", "an"), "Proxecto Nós")


# --- specialist wins over a generalist for its own language -----------------

def test_specialist_wins_over_generalist_for_its_language():
    """HiTZ owns Basque: its dedicated model should already win as
    'dedicated over multilingual'. Prove the specialist tier separately with
    two *equally dedicated* candidates, since that is what isolates the new
    tier from the pre-existing one."""
    generic_dedicated = specialist("some-es-eu-mt", ("es", "eu"), None)
    graph = TranslationGraph([HITZ_ES_EU, generic_dedicated])
    route = graph.route("es", "eu")
    assert route.hops[0].model_id == "mt-hitz-es-eu"
    assert route.hops[0].specialist is True
    assert route.hops[0].provenance_org == "HiTZ"


def test_specialist_beats_multilingual_directly():
    graph = TranslationGraph([GENERIC_ES_EU, HITZ_ES_EU])
    route = graph.route("es", "eu")
    assert route.hops[0].model_id == "mt-hitz-es-eu"


# --- no overreach: a specialist org is not preferred outside its language ---

def test_specialist_does_not_win_for_a_language_it_is_not_specialist_in():
    """HiTZ's provenance is only recognised for `eu`; on an es->en pair (not
    in SPECIALIST_MAP for HiTZ) it must not out-rank a smaller/newer
    non-specialist competitor on the specialist tier."""
    hitz_es_en = specialist("mt-hitz-es-en", ("es", "en"), "HiTZ",
                            release_date="2020-01-01")
    other_es_en = specialist("other-es-en", ("es", "en"), None,
                             release_date="2024-01-01")
    graph = TranslationGraph([hitz_es_en, other_es_en])
    route = graph.route("es", "en")
    # Neither is a specialist for es<->en, so recency (the next tier) decides.
    assert route.hops[0].model_id == "other-es-en"
    assert route.hops[0].specialist is False


def test_is_specialist_for_helper_has_no_overreach():
    assert is_specialist_for("HiTZ", "es", "eu") is True
    assert is_specialist_for("HiTZ", "es", "en") is False
    assert is_specialist_for(None, "es", "eu") is False


# --- recency applies only when everything else ties -------------------------

def test_recency_is_a_last_resort_tie_break():
    older = specialist("old-en-pt", ("en", "pt"), None, size_mb=300,
                       release_date="2019-01-01")
    newer = specialist("new-en-pt", ("en", "pt"), None, size_mb=300,
                       release_date="2024-01-01")
    graph = TranslationGraph([older, newer])
    assert graph.route("en", "pt").hops[0].model_id == "new-en-pt"


def test_recency_never_overrides_size_or_specialist_tier():
    """A newer *non*-specialist, smaller model still loses to the dedicated
    specialist model even though it is more recent - recency is consulted
    only after specialist provenance ties."""
    newer_generic = specialist("newer-generic", ("es", "eu"), None,
                               size_mb=100, release_date="2026-01-01")
    graph = TranslationGraph([HITZ_ES_EU, newer_generic])
    assert graph.route("es", "eu").hops[0].model_id == "mt-hitz-es-eu"


def test_missing_release_date_degrades_gracefully_never_wins():
    dated = specialist("dated", ("en", "pt"), None, release_date="2018-01-01")
    undated = specialist("undated", ("en", "pt"), None, release_date=None)
    graph = TranslationGraph([dated, undated])
    # Undated must not beat even an old, verified date.
    assert graph.route("en", "pt").hops[0].model_id == "dated"


# --- explicit route/model overrides all of it --------------------------------

def test_explicit_route_overrides_specialist_ranking():
    """A caller-built Route that deliberately picks the *generalist* model
    validates and executes as given - the ranking policy never gets a veto
    over an explicit route."""
    graph = TranslationGraph([GENERIC_ES_EU, HITZ_ES_EU])
    hop = Hop(model_id="generic-multi", src="es", tgt="eu", arch="m2m100",
             license="MIT", license_tier="permissive", size_mb=1200,
             dedicated=False)
    manual_route = Route(src="es", tgt="eu", hops=(hop,))
    validated = graph.validate_route(manual_route)
    assert validated.hops[0].model_id == "generic-multi"


# --- the deliberate proximity-rule overlaps ----------------------------------

def test_proxecto_nos_wins_asturian_over_aina_by_proximity():
    """Both institutions ship an es->ast model; Nos wins because Asturian is
    on the Galician-Portuguese continuum (Eonavian), not the Catalan one."""
    graph = TranslationGraph([AINA_ES_AST, NOS_ES_AST])
    route = graph.route("es", "ast")
    assert route.hops[0].model_id == "nos-mt-es-ast"
    assert route.hops[0].provenance_org == "Proxecto Nós"


def test_aina_wins_aragonese_over_nos_by_proximity():
    """Both institutions ship an es->an model; AINA wins because Aragonese
    borders Catalonia, not because Nós lacks a model for it."""
    graph = TranslationGraph([NOS_ES_AN, AINA_ES_AN])
    route = graph.route("es", "an")
    assert route.hops[0].model_id == "aina-es-an"
    assert route.hops[0].provenance_org == "Projecte AINA"
