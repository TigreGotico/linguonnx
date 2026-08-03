#!/usr/bin/env python3
"""Regenerate ``linguonnx/model_index/*.json`` from the HuggingFace API.

The registries are **generated**, not hand-maintained. Exports land in the
``TigreGotico`` org faster than a human can copy file lists into JSON, and a
stale registry is not a cosmetic problem: a missing entry means a language pair
silently routes the long way round, and a *wrong* entry means the router picks
a model that cannot do the pair and returns fluent text in the wrong language.

Run it::

    python scripts/sync_registry.py            # rewrite the JSON in place
    python scripts/sync_registry.py --check    # exit 1 if the committed JSON drifted

What is derived from where
--------------------------

Everything that can be read off the Hub is read off the Hub:

``arch``
    ``config.model_type`` from the repo's own config. ``marian`` is Marian.
    Otherwise the tokenizer decides: a repo with ``vocab.json`` reads ids
    straight out of it (M2M100), a repo without one uses fairseq's
    ``id = sp_id + 1`` convention (NLLB). This is a file-level fact, not a
    guess from the repo name.
``license``
    ``cardData.license`` when the card has YAML front matter. Most opus-mt
    exports do not, so the fallback is the ``**License:** ...`` line the export
    script writes into the README body. There is no third fallback: a repo
    whose licence cannot be read is skipped, because "probably Apache" is not
    a licence claim this library is willing to publish.
``languages`` (multilingual models)
    ``additional_special_tokens`` in ``special_tokens_map.json`` - the model's
    own tokenizer metadata. Never a list typed out here.
``pair`` (bilingual models)
    The repo name, **cross-checked against the base model named in the card**.
    See :func:`_marian_pair` - this is where the sharp edges are.
``size_mb``
    Summed blob sizes of the files the entry actually references. Not the
    repo total: the repos carry a ``decoder_model_merged.onnx`` that linguonnx
    never downloads.

Sharp edges, and why the base-model cross-check exists
------------------------------------------------------

A repo called ``opus-mt-en-pl-onnx`` is not necessarily an export of
``Helsinki-NLP/opus-mt-en-pl``. Several are exports of *group* models:

- ``opus-mt-pt-en-onnx`` is ``opus-mt-ROMANCE-en`` - multi-*source*. Harmless:
  the source language is inferred from the text, so ``pt -> en`` still holds.
- ``opus-mt-en-pl-onnx`` is ``opus-mt-en-sla`` - multi-*target*. **Not**
  harmless: without a ``>>pol<<`` prefix token on the input the model picks a
  Slavic language on its own and you get Czech. The token is recorded as
  ``target_token`` and applied automatically by ``TranslationModel.translate``.
- ``opus-mt-en-pt``/``en-tr``/``en-ko`` are ``tc-big`` variants - a bigger
  model for the same pair, and a different licence (cc-by-4.0, not apache-2.0).

So: when the base model's target segment differs from the repo's target, a
target token is **required**. If the README does not show one, the entry is
skipped rather than published with a coverage claim that cannot be honoured.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import http.client
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HF_API = "/api/models"
HF_RAW = "/{repo}/raw/main/{path}"
AUTHOR = "TigreGotico"

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "linguonnx" / "model_index"

#: Repo name -> registry model id, where stripping ``-onnx`` gives the wrong
#: answer. Kept short on purpose; anything not listed uses the derived id.
MODEL_ID_OVERRIDES: Dict[str, str] = {
    "nllb-200-distilled-600M-onnx": "nllb-600M",
    "aina-translator-es-oc-onnx": "aina-es-oc",
}

#: Bilingual fine-tunes of a multilingual base, where the language codes the
#: model expects are *not* recoverable from the TigreGotico card and had to be
#: read from the upstream source card. Each entry cites where it came from.
#:
#: ``aina-es-oc``: a full fine-tune of NLLB-200-600M for Spanish -> Aranese.
#: Aranese is not in NLLB, so Projecte Aina *added* a token for it - and chose
#: ``arn_Latn``, which collides with the ISO code for Mapudungun. The pair is
#: stated in the model's own codes; ``normalize_tag`` maps ``arn_Latn`` to
#: ``oc`` so the routing graph never sees the collision.
#: Source: https://huggingface.co/projecte-aina/aina-translator-es-oc
#:   "Since the original NLLB-200-600M doesn't support Aranese, we added a new
#:    token ("arn_Latn") to enable translation into Aranese."
BILINGUAL_FINETUNES: Dict[str, Dict] = {
    "aina-es-oc": {
        "pair": ["spa_Latn", "arn_Latn"],
        "notes": "Spanish -> Aranese (Gascon Occitan). NLLB-200-600M fine-tune "
                 "by Projecte Aina; Aranese uses the added token 'arn_Latn'. "
                 "One-way: there is no Aranese -> Spanish direction.",
    },
    #: ProxectoNos' nos-coda_iacobus family: OpenNMT-py checkpoints exported as
    #: a Pegasus-shaped graph (see the arch docstring above `_arch`). Each is a
    #: single hand-verified pair, one-way - there is no reverse-direction repo
    #: for any of them.
    "nos-coda_iacobus-en-gl": {"pair": ["en", "gl"],
                               "notes": "OpenNMT-py English -> Galician."},
    "nos-coda_iacobus-es-gl": {"pair": ["es", "gl"],
                               "notes": "OpenNMT-py Spanish -> Galician."},
    "nos-coda_iacobus-pt-gl": {"pair": ["pt", "gl"],
                               "notes": "OpenNMT-py Portuguese -> Galician."},
    "nos-coda_iacobus-es-pt": {"pair": ["es", "pt"],
                               "notes": "OpenNMT-py Spanish -> Portuguese."},
    "nos-coda_iacobus-en-pt": {"pair": ["en", "pt"],
                               "notes": "OpenNMT-py English -> Portuguese."},
    "nos-coda_iacobus-en-es": {"pair": ["en", "es"],
                               "notes": "OpenNMT-py English -> Spanish."},

    #: Projecte AINA's `aina-translator-XX-YY` family. Verified by inspecting
    #: each repo's own `int8/tokenizer_config.json`: these are plain
    #: `MarianTokenizer` exports with a placeholder `source_lang`/`target_lang`
    #: ("src"/"tgt") and NO `additional_special_tokens` list at all - i.e. a
    #: genuinely bilingual tokenizer, not a multilingual one wearing a
    #: fine-tune. cardData.language (2 codes) agrees with the repo name for
    #: every entry below, so the pair is a cross-checked fact, not a guess.
    #: Source: https://huggingface.co/projecte-aina/aina-translator-<pair>
    "aina-translator-ca-de": {"pair": ["ca", "de"], "notes": "Marian, Catalan -> German."},
    "aina-translator-ca-en": {"pair": ["ca", "en"], "notes": "Marian, Catalan -> English."},
    "aina-translator-ca-es": {"pair": ["ca", "es"], "notes": "Marian, Catalan -> Spanish."},
    "aina-translator-ca-fr": {"pair": ["ca", "fr"], "notes": "Marian, Catalan -> French."},
    "aina-translator-ca-it": {"pair": ["ca", "it"], "notes": "Marian, Catalan -> Italian."},
    "aina-translator-ca-pt": {"pair": ["ca", "pt"], "notes": "Marian, Catalan -> Portuguese."},
    "aina-translator-ca-zh": {"pair": ["ca", "zh"], "notes": "Marian, Catalan -> Chinese."},
    "aina-translator-de-ca": {"pair": ["de", "ca"], "notes": "Marian, German -> Catalan."},
    "aina-translator-en-ca": {"pair": ["en", "ca"], "notes": "Marian, English -> Catalan."},
    "aina-translator-es-ca": {"pair": ["es", "ca"], "notes": "Marian, Spanish -> Catalan."},
    "aina-translator-eu-ca": {"pair": ["eu", "ca"], "notes": "Marian, Basque -> Catalan."},
    "aina-translator-fr-ca": {"pair": ["fr", "ca"], "notes": "Marian, French -> Catalan."},
    "aina-translator-gl-ca": {"pair": ["gl", "ca"], "notes": "Marian, Galician -> Catalan."},
    "aina-translator-it-ca": {"pair": ["it", "ca"], "notes": "Marian, Italian -> Catalan."},
    "aina-translator-pt-ca": {"pair": ["pt", "ca"], "notes": "Marian, Portuguese -> Catalan."},
    "aina-translator-zh-ca": {"pair": ["zh", "ca"], "notes": "Marian, Chinese -> Catalan."},

    #: `aina-translator-es-an` / `-es-ast`: unlike the pairs above, these ARE
    #: NLLB-200-600M fine-tunes and keep the full 256-code inventory in
    #: `additional_special_tokens` - so like `aina-es-oc`, the pair is stated
    #: in the model's own added-token codes, verified from each repo's
    #: `int8/added_tokens.json`. Asturian's `ast_Latn` is already a native
    #: NLLB-200 code (id 256016); Aragonese has none, so AINA added `arg_Latn`
    #: at id 256204, mirroring how they added `arn_Latn` for Aranese.
    #: cc-by-nc-4.0, unlike the Marian pairs above.
    #: Source: https://huggingface.co/projecte-aina/aina-translator-es-an ,
    #:         https://huggingface.co/projecte-aina/aina-translator-es-ast
    "aina-translator-es-an": {
        "pair": ["spa_Latn", "arg_Latn"],
        "notes": "Spanish -> Aragonese. NLLB-200-600M fine-tune by Projecte "
                 "Aina; Aragonese uses the added token 'arg_Latn'. One-way.",
    },
    "aina-translator-es-ast": {
        "pair": ["spa_Latn", "ast_Latn"],
        "notes": "Spanish -> Asturian. NLLB-200-600M fine-tune by Projecte "
                 "Aina; Asturian uses NLLB's native 'ast_Latn'. One-way.",
    },

    #: Masakhane's `m2m100_418M_<lang>_fr_rel_news_ft` / `_fr_<lang>_...`
    #: family: M2M100-418M fine-tunes for ten French <-> West African language
    #: pairs. Verified from each repo's `int8/generation_config.json`: the
    #: target is forced via `forced_bos_token_id` baked into the export
    #: (mirrors this library's own `target_token` mechanism, so no extra field
    #: is needed here), and cardData.language names exactly the two languages,
    #: agreeing with the repo name. The low-resource side (bam/bbj/ewe/fon/mos)
    #: has no dedicated M2M100 vocabulary token - Masakhane's checkpoints read
    #: it from plain text on the source side, so only the *target* forcing
    #: matters, and that already lives in the export's own generation config.
    #: Source: https://huggingface.co/masakhane/m2m100_418M_<pair>
    "m2m100_418M_bam_fr_rel_news_ft": {"pair": ["bam", "fr"], "notes": "M2M100-418M fine-tune, Bambara -> French."},
    "m2m100_418M_bbj_fr_rel_news_ft": {"pair": ["bbj", "fr"], "notes": "M2M100-418M fine-tune, Ghomala -> French."},
    "m2m100_418M_fon_fr_rel_news_ft": {"pair": ["fon", "fr"], "notes": "M2M100-418M fine-tune, Fon -> French."},
    "m2m100_418M_fr_bam_rel_news_ft": {"pair": ["fr", "bam"], "notes": "M2M100-418M fine-tune, French -> Bambara."},
    "m2m100_418M_fr_bbj_rel_news_ft": {"pair": ["fr", "bbj"], "notes": "M2M100-418M fine-tune, French -> Ghomala."},
    "m2m100_418M_fr_ewe_rel_news_ft": {"pair": ["fr", "ewe"], "notes": "M2M100-418M fine-tune, French -> Ewe."},
    "m2m100_418M_fr_fon_rel_news_ft": {"pair": ["fr", "fon"], "notes": "M2M100-418M fine-tune, French -> Fon."},
    "m2m100_418M_fr_mos_rel_news_ft": {"pair": ["fr", "mos"], "notes": "M2M100-418M fine-tune, French -> Mossi."},
    "m2m100_418M_mos_fr_rel_news_ft": {"pair": ["mos", "fr"], "notes": "M2M100-418M fine-tune, Mossi -> French."},

    #: ProxectoNos' `nos-mt-*` family - the same OpenNMT-py/Pegasus-shaped
    #: export as `nos-coda_iacobus`, one hand-verified one-way pair each,
    #: cardData.language agreeing with the repo name.
    #: Source: https://huggingface.co/proxectonos/Nos_MT-CT2-<pair> (and
    #: proxectonos/es-arg, proxectonos/es-arn, proxectonos/es-ast).
    "nos-mt-en-gl": {"pair": ["en", "gl"], "notes": "OpenNMT-py English -> Galician."},
    "nos-mt-es-gl": {"pair": ["es", "gl"], "notes": "OpenNMT-py Spanish -> Galician."},
    "nos-mt-gl-en": {"pair": ["gl", "en"], "notes": "OpenNMT-py Galician -> English."},
    "nos-mt-gl-es": {"pair": ["gl", "es"], "notes": "OpenNMT-py Galician -> Spanish."},
    "nos-mt-es-ast": {"pair": ["es", "ast"], "notes": "OpenNMT-py Spanish -> Asturian."},
    "nos-mt-es-arg": {"pair": ["es", "an"], "notes": "OpenNMT-py Spanish -> Aragonese (card: 'an')."},
    "nos-mt-es-arn": {"pair": ["es", "oc"], "notes": "OpenNMT-py Spanish -> Aranese, card records it as 'oc'."},

    #: Softcatalà's Marian exports into Catalan, same tokenizer shape as the
    #: `aina-translator-ca-*` pairs above (verified the same way: card
    #: language agrees with the repo name, no multilingual token inventory).
    #: Source: https://huggingface.co/softcatala/translate-eus-cat ,
    #:         https://huggingface.co/softcatala/translate-oci-cat
    "translate-eus-cat": {"pair": ["eu", "ca"], "notes": "Marian, Basque -> Catalan."},
    "translate-oci-cat": {"pair": ["oc", "ca"], "notes": "Marian, Occitan -> Catalan."},
}

#: `opus-mt-tc-big-*` repos that cover more than two languages and cannot be
#: resolved to one pair by name + base-model cross-check (see
#: `_non_opus_marian_pair`): these are genuine *group* models (Ibero-Romance
#: <-> English/Catalan, or all-pairs-within-Italic), not bilingual fine-tunes,
#: so they do not belong in BILINGUAL_FINETUNES - representing them correctly
#: needs the same multi-language + prefix-token machinery as `liv4ever-mt`
#: (`_marian_multilingual_languages`), which does not yet handle the
#: underscore-joined macro-language grouping these repo names use
#: (`cat_oci_spa`, `itc-itc`). Left out on purpose rather than guessed at;
#: tracked as a known gap, not silently dropped.
_KNOWN_UNRESOLVED_GROUP_MODELS = (
    "opus-mt-tc-big-cat_oci_spa-en",
    "opus-mt-tc-big-en-cat_oci_spa",
    "opus-mt-tc-big-itc-itc",
)

#: Real architecture-classification gaps, found while closing the "43 missing
#: models" registry-completeness bug. `_arch()` keys the tokenizer class
#: purely off `config.model_type`, and for these two families that field
#: disagrees with the tokenizer the repo actually ships:
#:
#: - `aina-translator-ca-<xx>` / `<xx>-ca` (14 of the 18 Catalan pairs; the
#:   other 4 - `ca-zh`, `zh-ca`, `es-an`, `es-ast` - resolve fine): the graph
#:   reports `model_type: m2m_100`, which `_arch()` reads as the M2M100
#:   tokenizer family (a single merged `sentencepiece.bpe.model`, loaded by
#:   `SpmSeq2SeqTokenizer`). The actual tokenizer shipped is `MarianTokenizer`
#:   over a *dual* `source.spm` + `target.spm` + `vocab.json` - the Marian
#:   tokenizer shape, loaded by a different class entirely
#:   (`linguonnx/translate/tokenizers.py`). Forcing `arch="marian"` onto the
#:   registry entry without knowing whether the ONNX graph itself was also
#:   exported Marian-shaped (decoder I/O, `forced_bos` handling) risks a
#:   wrong-but-silent runtime pickup for real users - worse than staying
#:   unregistered. Needs a maintainer to inspect one of these repos' actual
#:   ONNX graph I/O signature, not just its tokenizer, before either fixing
#:   `_arch()` or adding a new hybrid tokenizer variant.
#: - `translate-eus-cat`, `translate-oci-cat` (Softcatalà): `model_type:
#:   pegasus`, same as ProxectoNos' `nos-mt-*`/`nos-coda_iacobus-*` - but
#:   those ship OpenNMT BPE vocab files and these ship a SentencePiece
#:   `tokenizer.json` instead. A third tokenizer shape hiding under the same
#:   `model_type`, same reasoning as above for not guessing.
_KNOWN_ARCH_MISMATCH: Tuple[str, ...] = (
    "aina-translator-ca-de", "aina-translator-ca-en", "aina-translator-ca-es",
    "aina-translator-ca-fr", "aina-translator-ca-it", "aina-translator-ca-pt",
    "aina-translator-de-ca", "aina-translator-en-ca", "aina-translator-es-ca",
    "aina-translator-eu-ca", "aina-translator-fr-ca", "aina-translator-gl-ca",
    "aina-translator-it-ca", "aina-translator-pt-ca",
    "translate-eus-cat", "translate-oci-cat",
)

#: Hand-verified fix-ups for a *multilingual* Marian export whose target is
#: chosen by a `<2xx>` prefix token read straight out of its own vocab.json
#: (see :func:`_marian_multilingual_languages`). Keyed by model id.
#:
#: ``liv4ever-mt``: TartuNLP's vocabulary spells Livonian's token `<2li>`.
#: ISO assigns `li` to Limburgish, so publishing the raw code as-is would make
#: the graph believe this model translates Limburgish. `native_codes` records
#: the true mapping so `TranslationModel` still sends the model its own
#: `<2li>` token, while the graph only ever sees the correct BCP-47 `liv`.
#: Source: https://huggingface.co/tartuNLP/liv4ever-mt (model card language table).
MARIAN_MULTILINGUAL_OVERRIDES: Dict[str, Dict] = {
    "liv4ever-mt": {
        "code_fixups": {"li": "liv"},
        "notes": "en/et/lv <-> Livonian (liv). Target chosen by a <2xx> "
                 "prefix token; the vocabulary spells Livonian '<2li>', which "
                 "collides with ISO's 'li' (Limburgish) - fixed up to 'liv'.",
    },
}

#: Canonical spelling for the licence ids the Hub reports lowercased.
LICENSE_DISPLAY: Dict[str, str] = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-3.0": "CC-BY-SA-3.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "gpl-3.0": "GPL-3.0",
    "afl-3.0": "AFL-3.0",
}

#: Licence -> routing tier. The graph will not put a more restrictive tier into
#: a caller's output than they asked for, so this classification is load-bearing.
LICENSE_TIERS: Dict[str, str] = {
    "Apache-2.0": "permissive",
    "MIT": "permissive",
    "CC-BY-4.0": "permissive",
    "CC-BY-SA-3.0": "share-alike",
    "CC-BY-SA-4.0": "share-alike",
    "GPL-3.0": "share-alike",
    "AFL-3.0": "permissive",
    "CC-BY-NC-4.0": "non-commercial",
}

# ---------------------------------------------------------------------------
# Specialist provenance
# ---------------------------------------------------------------------------
#
# `linguonnx.translate.graph.SPECIALIST_MAP` breaks routing ties in favour of
# the institution that owns a language over a generalist that merely covers
# it. Two registry fields feed that: `provenance_org` (who trained/fine-tuned
# it) and `release_date` (when the *upstream* model was first published).
#
# Both are derived here from the model_id, by prefix - the same naming
# convention the registry already relies on elsewhere (`MODEL_ID_OVERRIDES`,
# `BILINGUAL_FINETUNES`) - never guessed from `cardData`, which does not
# reliably name the training institution for most of these repos.
#
# `release_date` is the upstream repo's own `createdAt` on the Hub, read by
# hand from each institution's *source* repo - never `TigreGotico`'s own
# mirror, whose `createdAt`/`lastModified` are always close to "now" (that is
# when the export happened, not when the weights were trained) and would
# silently promote whatever this project re-uploaded most recently. Not every
# institution's date could be verified in one pass; a prefix with no date
# below stays undated on purpose (`_recency_sort_value` treats that as
# "unknown", which never wins a recency comparison) rather than guess.
#
# Ordered longest-prefix-first-in-effect by construction: `_provenance_for`
# walks this in order and returns the first prefix match, and no prefix here
# is itself a prefix of another, so match order does not matter today - kept
# as a tuple (not a dict) so a future ambiguous prefix pair has to be ordered
# on purpose instead of silently depending on dict insertion order.
PROVENANCE_BY_PREFIX: Tuple[Tuple[str, str, Optional[str]], ...] = (
    # HiTZ (Basque). Sampled from `HiTZ/mt-hitz-es-eu` on the Hub.
    ("mt-hitz-", "HiTZ", "2024-06-17"),
    # Proxecto Nós (Galician, and Asturian by proximity - see
    # linguonnx.translate.graph.SPECIALIST_MAP). Covers both the `nos-mt-*`
    # and `nos-coda_iacobus-*` OpenNMT-py families. No upstream date verified
    # yet for the `proxectonos` org repos; left undated rather than guessed.
    ("nos-", "Proxecto Nós", None),
    # Projecte AINA (Catalan, and Aragonese/Occitan-Aranese by proximity).
    # Sampled from `projecte-aina/aina-translator-es-ca`.
    ("aina-", "Projecte AINA", "2024-02-09"),
    # AI4Bharat / IndicTrans2. Sampled from
    # `ai4bharat/indictrans2-en-indic-dist-200M`.
    ("indictrans2", "AI4Bharat", "2023-09-12"),
    # TartuNLP's smugri (Finno-Ugric minority languages) M2M100 fine-tune.
    # Date not verified against the actual upstream `tartuNLP` repo yet.
    ("m2m100-418m-smugri", "TartuNLP", None),
    # liv4ever (Livonian). Sampled from `tartuNLP/liv4ever-mt` - the project
    # is TartuNLP-hosted but is its own named specialist for `liv`
    # specifically, distinct from the general smugri model above.
    ("liv4ever", "liv4ever", "2022-03-10"),
    # Masakhane's `_rel_news_ft` M2M100 fine-tunes (West-African pairs).
    # Sampled from `masakhane/m2m100_418M_fr_bam_rel_news_ft`; all of the
    # family share one release wave, so one verified date stands for it.
    ("m2m100_418m_", "Masakhane", "2022-04-09"),
)


def _provenance_for(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """``(org, release_date)`` for a model_id, or ``(None, None)``.

    Most registry entries have no specialist claim at all - a plain OPUS-MT,
    NLLB, or M2M100-base export - and get ``(None, None)``, which is exactly
    :class:`linguonnx.translate.graph.Capability`'s own default: no provenance
    field is written into the JSON for those, so old registries and new ones
    read the same absence the same way.
    """
    lowered = model_id.lower()
    for prefix, org, release_date in PROVENANCE_BY_PREFIX:
        if lowered.startswith(prefix):
            return org, release_date
    return None, None

#: Side files per architecture: registry key -> filename in the repo.
#: A file that is absent from the repo is dropped from the entry, except for
#: the ones listed in REQUIRED_SIDE_FILES.
SIDE_FILES: Dict[str, Dict[str, str]] = {
    "marian": {
        "source_spm": "source.spm",
        "target_spm": "target.spm",
        "vocab": "vocab.json",
        "config": "config.json",
        "generation_config": "generation_config.json",
    },
    "m2m100": {
        "spm": "sentencepiece.bpe.model",
        "config": "config.json",
        "generation_config": "generation_config.json",
        "special_tokens_map": "special_tokens_map.json",
        "vocab": "vocab.json",
        "added_tokens": "added_tokens.json",
    },
    "nllb": {
        "spm": "sentencepiece.bpe.model",
        "config": "config.json",
        "generation_config": "generation_config.json",
        "special_tokens_map": "special_tokens_map.json",
    },
    # T5 (MADLAD): a single SentencePiece model with the `<2xx>` target tokens
    # baked in as ordinary pieces - nothing in special_tokens_map.json names
    # them, so there is no separate vocab/special-tokens file to record.
    "madlad": {
        "spm": "spiece.model",
        "config": "config.json",
        "generation_config": "generation_config.json",
        "tokenizer_config": "tokenizer_config.json",
    },
    # IndicTrans2: a custom HF architecture with separate source/target
    # SentencePiece models and BPE merge dictionaries - see `_arch`.
    "indictrans2": {
        "spm_src": "model.SRC",
        "spm_tgt": "model.TGT",
        "dict_src": "dict.SRC.json",
        "dict_tgt": "dict.TGT.json",
        "config": "config.json",
        "generation_config": "generation_config.json",
        "tokenizer_config": "tokenizer_config.json",
    },
    # ProxectoNos' OpenNMT-py exports, shaped as Pegasus graphs (see `_arch`).
    # `bpe_code` is added per-entry in `translate_entries`, since its filename
    # is `{src}_35k.code` - it names the source language, not the pair.
    "opennmt-bpe": {
        "vocab": "onmt_vocab.json",
        "config": "config.json",
        "generation_config": "generation_config.json",
    },
}

#: Architectures whose inference pipeline this library does not implement yet,
#: and why. The entries they produce carry ``"runnable": false``, which keeps
#: them in the registry - listed, licensed, sized - while `can_translate`,
#: `route` and `available_languages` all agree that nothing can run them.
#: **Delete an architecture from this table when its pipeline lands**, re-run
#: this script, and the models start routing again with no code change in
#: `linguonnx/translate/graph.py`.
#:
#: Empty since the IndicTrans2 and OpenNMT-BPE pipelines landed in
#: `linguonnx/translate/preprocess.py`. Both were listed here; both now route.
UNRUNNABLE_ARCHS: Dict[str, str] = {}

REQUIRED_SIDE_FILES: Dict[str, Tuple[str, ...]] = {
    "marian": ("source_spm", "target_spm", "vocab", "config"),
    "m2m100": ("spm", "config", "special_tokens_map", "vocab"),
    "nllb": ("spm", "config", "special_tokens_map"),
    "madlad": ("spm", "config"),
    "indictrans2": ("spm_src", "spm_tgt", "dict_src", "dict_tgt", "config"),
    "opennmt-bpe": ("vocab", "config", "bpe_code"),
}

GRAPHS: Dict[str, str] = {
    "encoder": "encoder_model.onnx",
    "decoder": "decoder_model.onnx",
    "decoder_with_past": "decoder_with_past_model.onnx",
}


class SkipRepo(Exception):
    """This repo will not get a registry entry, and why."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

#: Sent on every request. The Hub throttles hard on an unidentified agent.
USER_AGENT = "linguonnx-sync-registry/1.0 (+https://github.com/TigreGotico/linguonnx)"

HOST = "huggingface.co"

#: One kept-alive connection for the whole sync. A fresh TLS handshake per
#: request turns a 141-repo crawl into a twenty-minute one; reusing the socket
#: brings it under three. Reset to None on any error so the next call redials.
_CONNECTION = None


def _get(path: str, retries: int = 4) -> bytes:
    """GET an absolute path on the Hub, over a reused HTTPS connection."""
    global _CONNECTION
    last = None
    for attempt in range(retries):
        if attempt:
            time.sleep(2 ** attempt)  # back off; the Hub rate-limits bursts
        try:
            if _CONNECTION is None:
                _CONNECTION = http.client.HTTPSConnection(HOST, timeout=30)
            _CONNECTION.request("GET", path, headers={
                "User-Agent": USER_AGENT, "Accept": "*/*"})
            response = _CONNECTION.getresponse()
            body = response.read()
            if response.status == 404:
                raise FileNotFoundError(path)
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {path}")
            return body
        except FileNotFoundError:
            raise
        except Exception as err:
            # A half-consumed or reset socket cannot be reused.
            try:
                if _CONNECTION is not None:
                    _CONNECTION.close()
            finally:
                _CONNECTION = None
            last = err
    raise RuntimeError(f"GET {path} failed after {retries} tries: {last}")


def _get_json(path: str):
    return json.loads(_get(path))


def list_repos() -> List[dict]:
    return _get_json(f"{HF_API}?author={AUTHOR}&limit=1000")


def repo_detail(repo_id: str) -> dict:
    return _get_json(f"{HF_API}/{repo_id}?blobs=true")


def raw_file(repo_id: str, path: str) -> str:
    return _get(HF_RAW.format(repo=repo_id, path=path)).decode("utf-8")


# ---------------------------------------------------------------------------
# Shared parsing
# ---------------------------------------------------------------------------

_LICENSE_BODY_RE = re.compile(r"[Ll]icen[cs]e:?\**\s*[`\"]?([A-Za-z0-9.\-]+)")
_BASE_MODEL_RE = re.compile(r"Helsinki-NLP/([A-Za-z0-9._\-]+)")
_TARGET_TOKEN_RE = re.compile(r">>(\w+)<<")


def _license_of(detail: dict, readme: str) -> str:
    """Licence id, from card metadata first and the README body second."""
    raw = (detail.get("cardData") or {}).get("license")
    if not raw:
        match = _LICENSE_BODY_RE.search(readme)
        raw = match.group(1) if match else None
    if not raw:
        raise SkipRepo("no licence in cardData or README")
    key = str(raw).strip().lower()
    if key not in LICENSE_DISPLAY:
        raise SkipRepo(f"unrecognised licence {raw!r}")
    return LICENSE_DISPLAY[key]


def _files_of(detail: dict) -> Dict[str, int]:
    return {s["rfilename"]: int(s.get("size") or 0)
            for s in detail.get("siblings", [])}


def _resolve(files: Dict[str, int], prefix: str, name: str) -> Optional[str]:
    """``int8/vocab.json`` if the variant ships one, else the root copy.

    Several int8 subdirectories carry only the graphs, and share the root
    tokenizer files. Both layouts have to produce a working entry.
    """
    if prefix and (prefix + name) in files:
        return prefix + name
    return name if name in files else None


def _with_external_data(files: Dict[str, int], paths: Iterable[str]) -> List[str]:
    """``*.onnx_data`` blobs for the given graphs.

    They are never opened by name; ONNX Runtime finds them by relative path
    next to the graph, so they only have to be fetched, not keyed.
    """
    extra = []
    for path in paths:
        blob = path + "_data"
        if blob in files:
            extra.append(blob)
    return sorted(extra)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(files: Dict[str, int]) -> Optional[str]:
    """``"lid"``, ``"translate"`` or None for a repo we do not index."""
    if "labels.json" in files and any(f.endswith("_hash.py") for f in files):
        return "lid"
    if "encoder_model.onnx" in files:
        return "translate"
    return None


#: IndicTrans2's fixed 22-language + English tag set. Not recoverable from the
#: export's own files - `config.json` carries vocab sizes, not a language
#: list, and the custom tokenizer ships no `special_tokens_map.json` for the
#: two directional variants. Read from the model's own published card:
#: https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
INDICTRANS2_TAGS: Tuple[str, ...] = (
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "eng_Latn", "gom_Deva",
    "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva",
    "mal_Mlym", "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya",
    "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml",
    "tel_Telu", "urd_Arab",
)
_INDICTRANS2_INDIC_TAGS = tuple(t for t in INDICTRANS2_TAGS if t != "eng_Latn")

#: model id suffix -> (src tags, tgt tags). `indic-indic` is symmetric: both
#: sides are the 22 Indic tags, English excluded from either side, because the
#: distilled indic-indic checkpoint was never trained on English at all.
INDICTRANS2_DIRECTIONS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "indictrans2-en-indic": (("eng_Latn",), _INDICTRANS2_INDIC_TAGS),
    "indictrans2-indic-en": (_INDICTRANS2_INDIC_TAGS, ("eng_Latn",)),
    "indictrans2-indic-indic": (_INDICTRANS2_INDIC_TAGS, _INDICTRANS2_INDIC_TAGS),
}

#: `<2xx>` / `>>xx<<` target-selector tokens, shared by MADLAD (T5, SentencePiece
#: pieces) and a multilingual Marian export like liv4ever-mt (plain vocab.json
#: keys). Both spell it `<2code>`; opus-mt group models spell it `>>code<<`
#: instead (see `_TARGET_TOKEN_RE`).
_PREFIX_TARGET_TOKEN_RE = re.compile(r"^<2([A-Za-z]{2,3}(?:_[A-Za-z]+)?)>$")

#: MADLAD's vocabulary also carries a handful of bare ISO-3166 REGION tokens
#: (`<2CA>`, `<2IR>`, `<2NL>`, `<2RU>`, `<2ZW>`). They match the target-token
#: shape but name a country, not a language, so claiming them as coverage
#: would advertise translation into "Zimbabwe". Language subtags are
#: lowercase by convention; an all-uppercase 2-letter code is a region.
_REGION_ONLY_TOKEN_RE = re.compile(r"^[A-Z]{2}$")


def _arch(detail: dict, files: Dict[str, int]) -> str:
    model_type = (detail.get("config") or {}).get("model_type", "")
    if model_type == "marian":
        return "marian"
    if model_type == "t5":
        return "madlad"
    if model_type == "IndicTrans":
        return "indictrans2"
    if model_type == "pegasus":
        return "opennmt-bpe"
    # M2M100 and NLLB share `model_type: m2m_100`; the tokenizer tells them
    # apart. M2M100 ships a vocab.json and reads ids from it; NLLB has none and
    # uses fairseq's sp_id + 1 offset instead.
    if "vocab.json" in files:
        return "m2m100"
    return "nllb"


# ---------------------------------------------------------------------------
# Translation entries
# ---------------------------------------------------------------------------

def _non_opus_marian_pair(repo_name: str, detail: dict
                          ) -> Tuple[List[str], Optional[str], Optional[str]]:
    """A bilingual Marian export that is not from the OPUS-MT project.

    ``mt-hitz-es-eu-onnx`` (HiTZ's Spanish -> Basque model) is a plain
    single-pair Marian with a properly filled card, so there is no group-model
    ambiguity and no prefix token. The pair still is not *taken* from the repo
    name: the name proposes it and ``cardData.language`` has to agree, as a
    set. A name and a card that disagree mean nobody actually knows which
    direction the weights run, and that is a skip.
    """
    card = detail.get("cardData") or {}
    declared = [str(code) for code in (card.get("language") or [])]
    if len(declared) != 2:
        raise SkipRepo(
            f"cannot read a language pair from the repo name {repo_name!r} and "
            f"the card declares {len(declared)} languages")
    stem = repo_name[:-len("-onnx")] if repo_name.endswith("-onnx") else repo_name
    parts = stem.split("-")
    if len(parts) < 2:
        raise SkipRepo(f"cannot read a language pair from {repo_name!r}")
    src, tgt = parts[-2], parts[-1]
    if {src, tgt} != set(declared):
        raise SkipRepo(
            f"repo name {repo_name!r} says {src}->{tgt} but the card declares "
            f"{declared}; refusing to guess the direction")
    base = card.get("base_model")
    if not base:
        raise SkipRepo("no base_model in the card to verify the pair against")
    return [src, tgt], str(base), None


def _marian_pair(repo_name: str, readme: str, detail: dict
                 ) -> Tuple[List[str], Optional[str], Optional[str]]:
    """``(pair, base_model, target_token)`` for one Marian export.

    The repo name gives the intended pair; the base model named in the card
    says whether the export is actually that pair or a slice of a group model.
    Disagreement on the *target* side means a prefix token is mandatory.
    """
    match = re.fullmatch(r"opus-mt-([a-z]{2,3})-([a-z]{2,3})-onnx", repo_name)
    if not match:
        return _non_opus_marian_pair(repo_name, detail)
    src, tgt = match.group(1), match.group(2)

    base_match = _BASE_MODEL_RE.search(readme)
    if not base_match:
        raise SkipRepo("no Helsinki-NLP base model named in the card")
    base = base_match.group(1)

    # Strip the `tc-big` marker: a bigger model for the very same pair.
    stem = base[len("opus-mt-"):] if base.startswith("opus-mt-") else base
    stem = stem[len("tc-big-"):] if stem.startswith("tc-big-") else stem
    parts = stem.split("-")
    if len(parts) != 2:
        raise SkipRepo(f"cannot parse base model {base!r}")
    base_src, base_tgt = parts

    token = None
    if base_tgt != tgt:
        # Multi-target group model: the target is chosen by a prefix token, and
        # is wrong by default. Only the card can tell us which token.
        tokens = sorted(set(_TARGET_TOKEN_RE.findall(readme)))
        if len(tokens) != 1:
            raise SkipRepo(
                f"base model {base!r} is multi-target for {src}->{tgt} but the "
                f"card shows {len(tokens)} target tokens; refusing to guess")
        token = f">>{tokens[0]}<<"
    # base_src != src is a multi-*source* group model (e.g. ROMANCE-en). The
    # source is inferred from the text, so the pair claim still holds as-is.
    return [src, tgt], base, token


def _assert_not_narrow_finetune(detail: dict, model_id: str) -> None:
    """Refuse to read a covering set off a *bilingual fine-tune*.

    A fine-tune keeps the whole base tokenizer, so ``special_tokens_map.json``
    still lists all 100 (or 202) languages long after the weights stopped being
    able to serve them. Masakhane's ``m2m100_418M_bbj_fr_rel_news_ft`` is a
    French <-> Ghomala' model wearing M2M100's full token inventory; trusting
    the tokenizer would publish a 100-language claim for a two-language model
    and let the router send anything through it.

    The tell is ``cardData.language``: general models declare
    ``["multilingual"]``, fine-tunes name their two or three languages. Those
    need a hand-verified :data:`BILINGUAL_FINETUNES` entry stating the pair in
    the model's own codes, because which token a fine-tune reused for a
    language the base never had is not recoverable from the export.
    """
    languages = (detail.get("cardData") or {}).get("language") or []
    if not isinstance(languages, list) or "multilingual" in languages:
        return
    if 0 < len(languages) <= 3:
        raise SkipRepo(
            f"bilingual fine-tune ({'/'.join(languages)}) of a multilingual "
            f"base: its tokenizer still lists every base language, so coverage "
            f"cannot be derived. Add a verified BILINGUAL_FINETUNES entry for "
            f"{model_id!r} to register it")


def _multilingual_languages(repo_id: str, files: Dict[str, int]) -> List[str]:
    """The model's covered set, from its own tokenizer metadata.

    ``additional_special_tokens`` in ``special_tokens_map.json`` is the list
    the model was actually built with - M2M100's ``__pt__``, NLLB's
    ``por_Latn``. Nothing is added to or removed from it here.
    """
    if "special_tokens_map.json" not in files:
        raise SkipRepo("no special_tokens_map.json to read languages from")
    data = json.loads(raw_file(repo_id, "special_tokens_map.json"))
    tokens = data.get("additional_special_tokens") or []
    codes = []
    for token in tokens:
        code = token[2:-2] if token.startswith("__") and token.endswith("__") \
            else token
        if code and code != "<mask>":
            codes.append(code)
    if not codes:
        raise SkipRepo("special_tokens_map.json lists no language tokens")
    # NLLB spells its set in FLORES codes (``kea_Latn``); the router speaks
    # BCP-47, so a raw FLORES tag makes the language unreachable even though
    # the model covers it. Normalise, keeping the raw tag when it does not
    # resolve rather than dropping the language.
    from linguonnx.detect.labels import to_bcp47
    normalized = set()
    for code in codes:
        try:
            tag = to_bcp47(code)
        except Exception:
            tag = code
        normalized.add(tag or code)
    return sorted(normalized)


def _madlad_languages(repo_id: str, files: Dict[str, int]) -> List[str]:
    """MADLAD's target set, read off its own SentencePiece vocabulary.

    Every language MADLAD was trained on gets a `<2xx>` piece baked into the
    tokenizer at train time; there is no separate special-tokens list. This is
    a file-level fact, not a guess: `sentencepiece` loads the model straight
    from bytes (no temp file needed) and every piece is scanned.
    """
    import urllib.request
    import sentencepiece as spm

    if "spiece.model" not in files:
        raise SkipRepo("no spiece.model to read <2xx> target tokens from")
    # spiece.model is Git-LFS/xet backed: `/raw/main/` only returns the LFS
    # pointer text for a file this size, not the bytes. `/resolve/main/`
    # redirects to the real blob, so this goes through urllib (which follows
    # redirects) instead of the raw-file connection used for small text files.
    url = f"https://{HOST}/{repo_id}/resolve/main/spiece.model"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        blob = response.read()
    processor = spm.SentencePieceProcessor()
    processor.LoadFromSerializedProto(blob)
    codes = sorted({
        match.group(1)
        for i in range(processor.get_piece_size())
        if (match := _PREFIX_TARGET_TOKEN_RE.match(processor.IdToPiece(i)))
        and not _REGION_ONLY_TOKEN_RE.match(match.group(1))
    })
    if not codes:
        raise SkipRepo("spiece.model has no <2xx> target-prefix pieces")
    return codes


def _marian_multilingual_languages(repo_id: str, files: Dict[str, int],
                                   model_id: str
                                   ) -> Tuple[List[str], Dict[str, str]]:
    """Codes (+ any hand-verified fix-up) for a `<2xx>`-prefix Marian export.

    liv4ever-mt is multilingual (en/et/lv/liv), not one dedicated pair, and
    picks its target the same way MADLAD does - a prefix token - just spelled
    out in a plain `vocab.json` instead of the SentencePiece model itself.
    """
    if "vocab.json" not in files:
        raise SkipRepo("no vocab.json to read <2xx> target tokens from")
    vocab = json.loads(raw_file(repo_id, "vocab.json"))
    codes = sorted({
        match.group(1) for key in vocab
        if (match := _PREFIX_TARGET_TOKEN_RE.match(key))
        and not _REGION_ONLY_TOKEN_RE.match(match.group(1))
    })
    if not codes:
        raise SkipRepo("vocab.json has no <2xx> target-prefix tokens")
    fixups = MARIAN_MULTILINGUAL_OVERRIDES.get(model_id, {}).get("code_fixups", {})
    codes = sorted({fixups.get(code, code) for code in codes})
    native_codes = {fixups[raw]: raw for raw in fixups}
    return codes, native_codes


def translate_entries(repo_id: str, detail: dict, readme: str) -> Dict[str, dict]:
    """Every precision variant of one translation repo."""
    name = repo_id.split("/")[1]
    model_id = MODEL_ID_OVERRIDES.get(
        name, name[:-len("-onnx")] if name.endswith("-onnx") else name)
    files = _files_of(detail)
    arch = _arch(detail, files)
    license_id = _license_of(detail, readme)

    shared: Dict[str, object] = {"arch": arch}
    declared_langs = (detail.get("cardData") or {}).get("language") or []
    opus_named = re.fullmatch(r"opus-mt-[a-z]{2,3}-[a-z]{2,3}-onnx", name) is not None

    marian_multilingual = None
    if arch == "marian" and not opus_named and isinstance(declared_langs, list) \
            and len(declared_langs) > 2:
        # A multilingual Marian export (liv4ever-mt): any-to-any within its
        # set, target chosen by a <2xx> prefix token - not the single
        # dedicated pair `_marian_pair` expects. The `tc-big`/ROMANCE-style
        # group models also declare >2 languages but pick their target with a
        # `>>xxx<<` token instead, so this only fires when `<2xx>` tokens are
        # actually present; otherwise it falls through to `_marian_pair`,
        # which knows what to do with (or skip) a `>>xxx<<` group model.
        try:
            marian_multilingual = _marian_multilingual_languages(repo_id, files, model_id)
        except SkipRepo:
            marian_multilingual = None

    if model_id in BILINGUAL_FINETUNES:
        shared.update(BILINGUAL_FINETUNES[model_id])
    elif marian_multilingual is not None:
        codes, native_codes = marian_multilingual
        shared["languages"] = codes
        shared["target_token_template"] = "<2{code}>"
        if native_codes:
            shared["native_codes"] = native_codes
    elif arch == "marian":
        pair, base, token = _marian_pair(name, readme, detail)
        shared["pair"] = pair
        shared["base_model"] = base if "/" in base else f"Helsinki-NLP/{base}"
        if token:
            shared["target_token"] = token
    elif arch == "madlad":
        shared["languages"] = _madlad_languages(repo_id, files)
    elif arch == "indictrans2":
        directions = None
        for suffix, dirs in INDICTRANS2_DIRECTIONS.items():
            if name.startswith(suffix):
                directions = dirs
                break
        if directions is None:
            raise SkipRepo(
                f"{name!r} does not match a known IndicTrans2 direction "
                f"({', '.join(INDICTRANS2_DIRECTIONS)})")
        shared["src_languages"], shared["tgt_languages"] = \
            list(directions[0]), list(directions[1])
    else:
        _assert_not_narrow_finetune(detail, model_id)
        shared["languages"] = _multilingual_languages(repo_id, files)

    entries: Dict[str, dict] = {}
    for suffix, prefix in (("", ""), ("-int8", "int8/")):
        if prefix and (prefix + GRAPHS["encoder"]) not in files:
            continue
        graphs = {}
        for key, filename in GRAPHS.items():
            path = _resolve(files, prefix, filename)
            if path is None:
                raise SkipRepo(f"missing graph {filename}")
            graphs[key] = path

        side: Dict[str, str] = {}
        for key, filename in SIDE_FILES[arch].items():
            path = _resolve(files, prefix, filename)
            if path is not None:
                side[key] = path
        if arch == "opennmt-bpe":
            # Two generations of OpenNMT-py export from ProxectoNos, same
            # graph shape, different side-file naming:
            #   nos-coda_iacobus-*: `onmt_vocab.json` + `<src>_35k.code`
            #     (BPE merges named after the source language, not the pair).
            #   nos-mt-*: `nos_vocab.json` + a single unqualified `source.bpe`
            #     shared by fp32/int8 (never duplicated under `int8/`).
            if "vocab" not in side:
                vocab_path = _resolve(files, prefix, "nos_vocab.json")
                if vocab_path is not None:
                    side["vocab"] = vocab_path
            if "pair" in shared:
                bpe_path = _resolve(files, prefix, f"{shared['pair'][0]}_35k.code")
                if bpe_path is None:
                    bpe_path = _resolve(files, prefix, "source.bpe")
                if bpe_path is not None:
                    side["bpe_code"] = bpe_path
        missing = [k for k in REQUIRED_SIDE_FILES[arch] if k not in side]
        if missing:
            raise SkipRepo(f"missing required side files: {', '.join(missing)}")

        extra = _with_external_data(files, graphs.values())
        size = sum(files.get(p, 0)
                   for p in list(graphs.values()) + list(side.values()) + extra)

        entry = {
            "model_id": model_id + suffix,
            "hf_repo": repo_id,
            "arch": arch,
            "graphs": graphs,
            "side_files": side,
            "extra_files": extra,
        }
        entry.update({k: v for k, v in shared.items() if k != "arch"})
        entry["license"] = license_id
        entry["license_tier"] = LICENSE_TIERS[license_id]
        entry["size_mb"] = max(1, round(size / 1e6))
        entry["precision"] = "int8" if prefix else "fp32"
        provenance_org, release_date = _provenance_for(entry["model_id"])
        if provenance_org:
            entry["provenance_org"] = provenance_org
        if release_date:
            entry["release_date"] = release_date
        if arch in UNRUNNABLE_ARCHS:
            entry["runnable"] = False
            entry["unrunnable_reason"] = UNRUNNABLE_ARCHS[arch]
        entries[entry["model_id"]] = entry
    return entries


# ---------------------------------------------------------------------------
# LID entries
# ---------------------------------------------------------------------------

def lid_entries(repo_id: str, detail: dict, readme: str) -> Dict[str, dict]:
    name = repo_id.split("/")[1]
    model_id = MODEL_ID_OVERRIDES.get(
        name, name[:-len("-onnx")] if name.endswith("-onnx") else name)
    files = _files_of(detail)
    license_id = _license_of(detail, readme)

    config = json.loads(raw_file(repo_id, "config.json"))
    labels = json.loads(raw_file(repo_id, "labels.json"))
    num_labels = len(labels) if isinstance(labels, (list, dict)) else None
    if not num_labels:
        raise SkipRepo("labels.json is empty")
    # lid.176 is hierarchical-softmax; everything else is a plain softmax head.
    loss = str(config.get("loss", "softmax")).lower()
    if loss not in ("softmax", "hs"):
        raise SkipRepo(f"unknown loss {loss!r}")

    side = {}
    for key, filename in (("vocab", "vocab.txt"), ("labels", "labels.json"),
                          ("config", "config.json")):
        if filename not in files:
            raise SkipRepo(f"missing {filename}")
        side[key] = filename
    # A hierarchical-softmax model (lid.176) cannot decode without its tree:
    # the ONNX graph emits scores over internal nodes, not labels. Required
    # exactly when loss == "hs", and refused when it is missing, because the
    # entry would download cleanly and then fail at the first prediction.
    if loss == "hs":
        if "hs_tree.json" not in files:
            raise SkipRepo("loss is 'hs' but the repo has no hs_tree.json")
        side["hs_tree"] = "hs_tree.json"
    elif "hs_tree.json" in files:
        raise SkipRepo("repo ships hs_tree.json but declares a softmax loss")

    entries = {}
    for suffix, candidates in (
            ("", [f for f in files if f.endswith(".onnx")
                  and ".int8." not in f]),
            ("-int8", [f for f in files if f.endswith(".onnx")
                       and ".int8." in f])):
        if not candidates:
            continue
        onnx_file = sorted(candidates)[0]
        entries[model_id + suffix] = {
            "model_id": model_id + suffix,
            "hf_repo": repo_id,
            "onnx_file": onnx_file,
            "side_files": dict(side),
            "license": license_id,
            "engine": "fasttext-onnx",
            "loss": loss,
            "num_labels": num_labels,
            "size_mb": max(1, round(
                (files.get(onnx_file, 0)
                 + sum(files.get(f, 0) for f in side.values())) / 1e6)),
            "precision": "int8" if suffix else "fp32",
        }
    return entries


# ---------------------------------------------------------------------------
# Merge + write
# ---------------------------------------------------------------------------

def merge_preserving(existing: Dict[str, dict],
                     generated: Dict[str, dict]) -> Dict[str, dict]:
    """Generated values win; hand-authored *extra* keys survive.

    A curated ``notes``, a pinned override, anything a human added to an entry
    and the API knows nothing about is carried across. Only keys this script
    produces are allowed to change, so a re-run never silently discards
    editorial work.
    """
    merged = {}
    for model_id, entry in generated.items():
        previous = existing.get(model_id, {})
        kept = {k: v for k, v in previous.items() if k not in entry}
        combined = dict(entry)
        combined.update(kept)
        merged[model_id] = combined
    return merged


def _sorted_json(registry: Dict[str, dict]) -> str:
    """Stable text: entries sorted by id, keys sorted inside each entry.

    Re-running on an unchanged Hub must produce a byte-identical file, or
    ``--check`` is worthless and every sync makes a noisy diff.
    """
    ordered = {model_id: dict(sorted(registry[model_id].items()))
               for model_id in sorted(registry)}
    return json.dumps(ordered, indent=2, ensure_ascii=False,
                      sort_keys=False) + "\n"


def build() -> Tuple[Dict[str, dict], Dict[str, dict], List[Tuple[str, str]]]:
    translate: Dict[str, dict] = {}
    lid: Dict[str, dict] = {}
    skipped: List[Tuple[str, str]] = []

    repos = list_repos()
    for index, summary in enumerate(sorted(repos, key=lambda r: r["id"]), 1):
        repo_id = summary["id"]
        print(f"[{index}/{len(repos)}] {repo_id}", file=sys.stderr)
        try:
            detail = repo_detail(repo_id)
        except Exception as err:
            skipped.append((repo_id, f"detail fetch failed: {err}"))
            continue
        files = _files_of(detail)
        kind = classify(files)
        if kind is None:
            continue
        try:
            readme = raw_file(repo_id, "README.md") if "README.md" in files else ""
            if kind == "translate":
                translate.update(translate_entries(repo_id, detail, readme))
            else:
                lid.update(lid_entries(repo_id, detail, readme))
        except SkipRepo as err:
            skipped.append((repo_id, str(err)))
        except Exception as err:
            skipped.append((repo_id, f"{type(err).__name__}: {err}"))
    return translate, lid, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the committed registry "
                             "differs from what the Hub now says")
    args = parser.parse_args(argv)

    translate, lid, skipped = build()
    if not translate or not lid:
        print("refusing to write an empty registry; is the Hub reachable?",
              file=sys.stderr)
        return 2

    drift = False
    for kind, generated in (("translate", translate), ("lid", lid)):
        path = INDEX_DIR / f"{kind}.json"
        existing = json.loads(path.read_text(encoding="utf-8")) \
            if path.exists() else {}
        text = _sorted_json(merge_preserving(existing, generated))
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if text == current:
            print(f"{kind}.json: up to date ({len(generated)} entries)")
            continue
        drift = True
        added = sorted(set(generated) - set(existing))
        removed = sorted(set(existing) - set(generated))
        if args.check:
            print(f"{kind}.json: DRIFT - {len(added)} new, {len(removed)} "
                  f"stale, content differs", file=sys.stderr)
            for model_id in added:
                print(f"  + {model_id}", file=sys.stderr)
            for model_id in removed:
                print(f"  - {model_id}", file=sys.stderr)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{kind}.json: wrote {len(generated)} entries "
                  f"(+{len(added)} new, -{len(removed)} stale)")

    for repo_id, reason in skipped:
        print(f"skipped {repo_id}: {reason}", file=sys.stderr)

    # A silently-skipped repo is drift too: `--check` diffs the committed
    # JSON against a freshly generated one, and a repo that fails extraction
    # on *every* run is missing from both sides of that diff every time - the
    # comparison alone can never catch it (this is exactly how 43 published
    # models went unnoticed: skip, not drift). So any skip outside the
    # documented, hand-reviewed allowlist fails `--check` on its own,
    # independent of whether the JSON text changed.
    def _stem(repo_id: str) -> str:
        name = repo_id.split("/")[-1]
        return name[:-len("-onnx")] if name.endswith("-onnx") else name

    _allowlisted = set(_KNOWN_UNRESOLVED_GROUP_MODELS) | set(_KNOWN_ARCH_MISMATCH)
    unexpected_skips = [
        (repo_id, reason) for repo_id, reason in skipped
        if _stem(repo_id) not in _allowlisted
    ]
    if args.check and unexpected_skips:
        print(f"{len(unexpected_skips)} translate/lid repo(s) on the Hub "
              f"cannot be registered and are not in the documented "
              f"allowlist:", file=sys.stderr)
        for repo_id, reason in unexpected_skips:
            print(f"  ! {repo_id}: {reason}", file=sys.stderr)
        return 1

    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
