# The model registry

Two JSON files under `linguonnx/model_index/` say what exists: `lid.json` for
language identification and `translate.json` for translation. They hold 10 and
369 entries. Every entry names a HuggingFace repo, the exact files to fetch,
the languages covered, the licence and the size.

Both files are **generated**, not edited by hand. See
[Keeping it in sync](#keeping-it-in-sync) below for why that matters more than
it sounds.

## What is in the translation registry

369 entries is 184 int8 and 185 fp32 entries across 185 models. `load_translator()` defaults to
`precision="int8"`; pass `precision="fp32"` or `precision=None` for both.

| model | arch | coverage | size (int8) | licence |
|---|---|---|---|---|
| `madlad400-3b-mt` | T5 (MADLAD) | 450, any-to-any | 4.9 GB | Apache-2.0 |
| `nllb-600M` | NLLB-200 | 202, any-to-any | 1.9 GB | **CC-BY-NC-4.0** |
| `m2m100-418M` | M2M100 | 100, any-to-any | 1.2 GB | MIT |
| `m2m100-1.2B` | M2M100 | 100, any-to-any | 2.3 GB | MIT |
| `m2m100-418M-smugri` | M2M100 | 104, any-to-any; adds Livonian, Võro, Sami | 1.2 GB | MIT |
| `liv4ever-mt` | Marian, multi-target | 4, any-to-any (en, et, lv, Livonian) | 569 MB | Apache-2.0 |
| `indictrans2-en-indic-dist-200M` | IndicTrans2 | English → 25 Indic tags, **one-way** | 480 MB | MIT |
| `indictrans2-indic-en-dist-200M` | IndicTrans2 | 25 Indic tags → English, **one-way** | 342 MB | MIT |
| `indictrans2-indic-indic-dist-320M` | IndicTrans2 | 25 Indic tags, any-to-any | 532 MB | MIT |
| `aina-es-oc` | NLLB fine-tune | Spanish → Aranese, **one-way** | 1.9 GB | **CC-BY-NC-4.0** |
| `nos-coda_iacobus-*` (6) | OpenNMT | one pair each; en/es/pt → gl, es↔pt, en↔es | 499 MB – 814 MB | MIT |
| `mt-hitz-*` (6) | Marian | `ca-eu`, `en-eu`, `es-eu`, `eu-en`, `eu-es`, `gl-eu` | 90 MB – 153 MB | Apache-2.0 |
| `opus-mt-*` (124) | Marian | one pair each | 78 MB – 716 MB | Apache-2.0 or CC-BY-4.0 |

The opus-mt pairs published as int8:

```
af-en  ar-en  az-en  bg-en  bn-en  ca-en  ca-es  ca-fr
ca-it  ceb-en  cs-en  cy-en  da-en  de-en  en-af  en-ar
en-az  en-bg  en-ca  en-cs  en-cy  en-da  en-de  en-el
en-es  en-et  en-eu  en-fi  en-fr  en-ga  en-gl  en-gmq
en-he  en-hi  en-hu  en-hy  en-id  en-is  en-it  en-jap
en-ml  en-mr  en-nl  en-pl  en-pt  en-ro  en-ru  en-sk
en-sq  en-sv  en-ti  en-tr  en-uk  en-ur  en-vi  en-xh
en-zh  es-ca  es-en  es-eu  es-gl  et-en  eu-en  eu-es
fi-en  fr-ca  fr-en  ga-en  gl-en  gl-es  gl-pt  gmq-en
hi-en  hu-en  hy-en  id-en  is-en  it-en  itc-itc  ja-en
ka-en  ko-en  lv-en  mg-en  mk-en  ml-en  mr-en  mt-en
nl-en  pa-en  pl-en  pt-ca  pt-en  pt-gl  ru-en  sk-en
sm-en  sn-en  sq-en  st-en  sv-en  tc-base-en-sh  tc-big-cat_oci_spa-en  tc-big-el-en
tc-big-en-cat_oci_spa  tc-big-en-el  tc-big-en-ro  tc-big-en-zle  tc-big-gmw-gmw  tc-big-he-en  tc-big-itc-itc  tc-big-sh-en
tc-big-zle-en  th-en  tl-en  tn-en  tr-az  tr-en  ts-en  uk-en
ur-en  vi-en  xh-en  zh-en
```

Together they reach 586 languages over the default graph. MADLAD supplies most
of the tail, including the ones no other model here has: Mirandese, Aragonese,
Occitan and several hundred more.

Four entries — `nllb-600M` and `aina-es-oc`, in both precisions — are
non-commercial and out of the default graph. See [licences.md](licences.md).

Sizes are the sum of the blobs an entry actually references, not the repo
total: these repos also carry a `decoder_model_merged.onnx` that `linguonnx`
never downloads.

`opus-mt-en-ko` is **not** in the registry. Its upstream export
(`Helsinki-NLP/opus-mt-tc-big-en-ko`) ships a `vocab.json` that spells only
21% of the pieces its own `source.spm` produces, so most English words reach
the encoder as `<unk>` and the model answers fluent Korean nonsense.
`transformers` builds the same broken input ids from the same files, so this
is an upstream defect, not an ONNX one. English → Korean routes through
M2M100 instead, which scores chrF 30.4 on that pair (FLORES-200 devtest,
n=100, beam4).

Three opus-mt group models translate **from English only**, into the
languages their vocabulary carries a `>>xxx<<` token for:
`opus-mt-tc-big-en-zle` (be, ru, rue, uk), `opus-mt-en-gmq`
(da, fo, is, nb, nn, sv) and `opus-mt-tc-big-en-cat_oci_spa` (ca, es, oc).
The token set names the target side; the encoder reads English and nothing
else. `opus-mt-tc-big-en-zle` also carries `>>orv<<`/`>>orv_Cyrl<<` (Old East
Slavic) in its vocabulary, but both produce modern Russian, so neither is
claimed.

## What is in the LID registry

| model_id | labels | size | licence |
|---|---|---|---|
| `glotlid` / `glotlid-int8` | 2102 | 1.68 GB / 425 MB | Apache-2.0 |
| `lid176` / `lid176-int8` | 176 | 131 MB / 33 MB | CC-BY-SA-3.0 |
| `lid218e` / `lid218e-int8` | 218 | 1 MB / 295 MB | **CC-BY-NC-4.0** |
| `openlid` / `openlid-int8` | 201 | 1.23 GB / 310 MB | GPL-3.0 |
| `openlid-v2` / `openlid-v2-int8` | 200 | 1.22 GB / 305 MB | GPL-3.0 |

Each entry also carries the `loss` the model was trained with, which is how
`linguonnx` knows to walk a Huffman tree for lid.176 rather than reading
softmax outputs. See [detect.md](detect.md).

## Reading the registry from Python

```python
from linguonnx.model_manager import list_models, registry_entry

entry = registry_entry("opus-mt-pt-en-int8", kind="translate")
print(entry["arch"], entry["pair"], entry["license"], entry["size_mb"])
# marian ['pt', 'en'] Apache-2.0 172

print(len(list_models(kind="translate")), len(list_models(kind="lid")))
# 369 10
```

## The download cache

Files are cached under `$LINGUONNX_CACHE/models/<model_id>/`, defaulting to
`~/.cache/linguonnx`. The variable is read once at import, so set it before
importing linguonnx. Point it at bulk storage on a server: the whole
translation registry is well over 100 GB, and a symlink at `~/.cache/linguonnx`
is not a substitute — it is invisible to anyone reading the code and the root
disk fills up the moment it goes missing or a service runs as another user.

Downloads go
through `huggingface_hub.hf_hub_download`, which does its own resumable and
checksummed download, and are then copied into the cache atomically — written
to a temp sibling and `os.replace`'d into place, so a killed process never
leaves a truncated file at the final path. A zero-byte file there is always
treated as "not cached" and fetched again.

The temp name is unique per process and per call. That is what makes the
guarantee hold when two workers cold-start the same model at the same time:
they write to different temp files, and the second `os.replace` publishes a
complete file. A shared temp name lets them interleave, and the result is a
corrupt file that is not zero bytes — which no later run would notice.

A registry filename must stay inside its model's cache directory. Absolute
paths and `..` components are refused, so an edited registry file cannot turn
a download into a write anywhere else on the host.

### Pinning and verification

If a registry entry carries a `revision` (a commit SHA — HuggingFace tags and
branches are mutable, so they pin nothing), it is passed to the hub, and every
client then fetches the same bytes. If an entry carries a `sha256` map, each
downloaded file is verified against it and a mismatch raises before anything is
published to the cache. Both fields are optional and no entry carries them yet;
`scripts/sync_registry.py` has to start emitting them.

### Download budget and warm-up

A cold fetch bigger than `LINGUONNX_MAX_DOWNLOAD_MB` raises
`DownloadTooLargeError` instead of holding a request thread for an hour on a
slow link. The default of 8192 refuses nothing in the current registry; set it
lower on a server where the request path must stay predictable. `0` disables
the check. A warm cache never trips it.

To keep downloads off the request path entirely, warm the cache at startup:

```python
from linguonnx.model_manager import prefetch

prefetch("glotlid-int8")
prefetch("opus-mt-pt-en-int8", kind="translate")
```

`prefetch()` ignores the budget, because it does not run in a request.

### How many models stay loaded

A `Translator` keeps at most `model_cache_size` loaded models alive, four by
default, and evicts the least recently used one. Eviction releases that model's
three ONNX sessions.

The bound matters because the whole default graph is 73 models and about
25 GB. Without it, a long-lived server that routes over many language pairs
converges on loading all of them, gets OOM-killed, restarts cold, and pays
every download again.

```python
from linguonnx import load_translator

tx = load_translator(model_cache_size=8)   # more RAM, fewer reloads
print(tx.loaded_models)                    # least recently used first
```

Raise it when one process serves a few hot pairs and has the RAM; lower it on a
small device. An evicted model reloads from the disk cache on next use, so
eviction costs session-build time, not a download.

## Keeping it in sync

Exports land in the `TigreGotico` HuggingFace org faster than anyone can copy
file lists into JSON, and a stale registry is not a cosmetic problem. A missing
entry makes a language pair route the long way round. A *wrong* entry makes the
router pick a model that cannot do the pair, which then answers fluently in the
wrong language and raises nothing.

So the registries are regenerated from the Hub:

```bash
python scripts/sync_registry.py            # rewrite both registries
python scripts/sync_registry.py --check    # exit 1 if the committed JSON drifted
```

`--check` writes nothing and exits 1 on drift. It prints which entries are new
and which are stale, so "someone published a model and forgot the registry"
shows up as a failing check instead of a silent gap. Re-running the generator
on an unchanged Hub produces a byte-identical file.

It needs a network and a full 280-repo crawl, so no workflow runs it — it is a
command a maintainer runs, not a gate CI applies. Everything that *can* be
checked offline against the committed registry is a test in
`test/test_registry_sync.py`, and those do run in CI.

A sync never destroys hand-verified work. The script owns the fields it
derives from the Hub — `GENERATED_KEYS` in `scripts/sync_registry.py`: the file
listing, the licence, the size, the `>>xxx<<` coverage read out of each
export's own `vocab.json` — and may overwrite or remove those. Every other
field in an entry is carried across untouched, whether or not anyone declared
it. `notes` and `quality` go further and win outright: the script writes a
default one-line `notes`, but the committed text may carry a caveat somebody
found by reading 100 translations, and `quality` holds chrF numbers measured
against a FLORES-200/MAFAND reference that no crawl can re-derive.

Where the generator disagrees with a curated field, the curated value is kept
— and the value the generator wanted is committed too, to
`model_index/<kind>_overruled.json`. That file is what stops a curated field
from freezing the entry: the registry text no longer moves when an upstream
card is rewritten, so nothing else would report it, but `_overruled.json`
changes and `--check` fails on it like any other drift (`OVERRULED-DRIFT` on
stderr). Clearing it is the normal loop — run the sync, read the diff, fold in
anything real, commit. Failing on the disagreement itself would never clear: a
curated note differs from the generated one by definition.

Fields in neither list — a typo'd `note`, or one whose generator was deleted —
are preserved and printed with a `CURATED` prefix. They are immortal by
design; the print is so they are not also invisible.

`human_owned_losses` refuses to write if a run would fail to reproduce a
curated value, naming each entry and field. It is a tripwire on the merge
function, not a runtime guard: while `merge_preserving` is correct it cannot
fire, because the merge copies exactly the set of keys it inspects. It exists
because that function is the thing that broke.

The allow-list is of generated keys, not of human ones, on purpose: a list of
human-owned keys fails silently the first time somebody curates a field nobody
remembered to add to it. Adding a derived field means adding it to
`GENERATED_KEYS`; the script refuses to run until you do.

See [routing.md#measured-quality](routing.md#measured-quality) for what
`quality` means and how it is used.

Every repo the script refuses is written to `linguonnx/model_index/skipped.json`
alongside the registries, with the reason. A skip used to reach stderr and
nowhere else, which is how 43 published models stayed invisible: `--check`
diffs the committed registry against a fresh one, and a repo that fails
extraction on every run is missing from both sides of that diff every time.

### What is derived from where

Everything the Hub can answer is read from the Hub, never typed out:

| field | source |
|---|---|
| `arch` | `config.model_type` from the repo's own config; then the tokenizer files. A repo with `vocab.json` reads ids straight out of it (M2M100); one without uses fairseq's `id = sp_id + 1` (NLLB). |
| `license` | `cardData.license` when the card has YAML front matter, else the `**License:**` line in the README body — most opus-mt exports have no front matter. |
| `languages` | `additional_special_tokens` in the model's own `special_tokens_map.json`. |
| `pair` | The repo name, cross-checked against the base model named in the card. Never a language *family*: `itc`, `sla`, `mul` and the other ISO 639-5 collection codes are refused as pair sides, and the repo takes the group-model path below. |
| `target_token_template` + `native_codes` | For a model that picks its target with a prefix token. The token set is read from the model's own vocabulary — `<2xx>` for MADLAD and `liv4ever-mt`, `>>xxx<<` for the opus-mt group models — and `native_codes` maps the BCP-47 tag the graph uses back to the model's own spelling, so the graph sees `it` and the model still gets `>>ita<<`. |
| `size_mb` | Summed blob sizes of the files the entry actually references. Load-bearing twice over: it breaks ties in the route ranking, and it is what `max_model_mb` compares against, so an entry that under-reports its size gets routed onto hosts that cannot afford it. See [routing](routing.md#size-budget). |
| `runnable` | Written as `false`, with an `unrunnable_reason`, for an architecture whose inference pipeline this library does not implement. The router excludes those models. Delete the architecture from `UNRUNNABLE_ARCHS` in the script when its pipeline lands. |

There is no third fallback for the licence. A repo whose licence cannot be read
is skipped, because "probably Apache" is not a licence claim this library is
willing to publish on someone else's behalf.

### Two things the script refuses to guess

**Group models.** A repo called `opus-mt-en-pl-onnx` is an export of
`opus-mt-en-sla`, a model that serves several Slavic languages from one decoder
and needs a `>>pol<<` prefix token on the input. The script cross-checks the
repo name against the base model named in the card; when the target side
disagrees it records the `target_token` the export was verified against, and
skips the repo entirely if the card does not show one. Publishing a coverage
claim that cannot be honoured is worse than publishing nothing.

A repo whose *name* is a family — `opus-mt-itc-itc`, `opus-mt-mul-en` — is not
a pair at all and is not treated as one. `itc` is the Italic family, so no
caller can ask for it; the entry that used to say `pair: ["itc", "itc"]` was an
edge nobody could reach, and the mandatory prefix token was missing because the
"base target differs from repo target" test cannot fire when both sides say
`itc`. These take the token path instead: the `>>xxx<<` keys in the export's own
`vocab.json` are the coverage, because a language with no token cannot be
selected whatever the card lists.

**Bilingual fine-tunes of multilingual bases.** A fine-tune keeps the whole
base tokenizer, so `special_tokens_map.json` still lists all 100 or 202
languages long after the weights stopped serving them. Those need a verified
entry in `BILINGUAL_FINETUNES` stating the pair in the model's own codes, and
are skipped with a recorded reason until someone adds one. The test is a
subset test, not a count: the entry may not claim a language its own Hub card
does not. A threshold cannot tell a narrow fine-tune from a general model —
`m2m100-418M-smugri` is a Finno-Ugric fine-tune declaring **eight** languages,
which walked past the old "three or fewer" rule and published a 104-language
claim including `th -> sw`. A model whose real set is narrower than its
tokenizer states it in `MULTILINGUAL_LANGUAGE_OVERRIDES`. `aina-es-oc` is one
of these: an NLLB-600M fine-tune whose tokenizer claims 202 languages and whose
weights do Spanish into Aranese, one way.

There is a third hand-maintained fix-up, `MARIAN_MULTILINGUAL_OVERRIDES`, for a
model whose own code collides with a different language's ISO tag —
`liv4ever-mt`'s `<2li>` means Livonian, not Limburgish. Left alone, the router
would offer that model for a language it has never seen.
