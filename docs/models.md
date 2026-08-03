# The model registry

Two JSON files under `linguonnx/model_index/` say what exists: `lid.json` for
language identification and `translate.json` for translation. They hold 8 and
150 entries. Every entry names a HuggingFace repo, the exact files to fetch,
the languages covered, the licence and the size.

Both files are **generated**, not edited by hand. See
[Keeping it in sync](#keeping-it-in-sync) below for why that matters more than
it sounds.

## What is in the translation registry

150 entries is fp32 and int8 of 75 models. `load_translator()` defaults to
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
| `mt-hitz-*` (3) | Marian | `ca-eu`, `es-eu`, `eu-es` | 153 MB each | Apache-2.0 |
| `opus-mt-*` (56) | Marian | one pair each | 78 MB – 448 MB | Apache-2.0 or CC-BY-4.0 |

The opus-mt pairs published as int8:

```
ar-en  ca-en  ca-es  ca-fr  ca-it  de-en  en-ar  en-bg  en-ca  en-cs
en-da  en-de  en-el  en-es  en-eu  en-fi  en-fr  en-gl  en-he  en-hi
en-hu  en-id  en-it  en-ko  en-nl  en-pl  en-pt  en-ro  en-ru  en-sv
en-tr  en-uk  en-vi  en-zh  es-ca  es-en  es-eu  es-gl  eu-en  eu-es
fr-ca  fr-en  gl-en  gl-es  gl-pt  it-en  itc-itc  nl-en  pl-en  pt-ca
pt-en  pt-gl  ru-en  tr-en  uk-en  zh-en
```

Together they reach 459 languages over the default graph. MADLAD supplies most
of the tail, including the ones no other model here has: Mirandese, Aragonese,
Occitan and several hundred more.

Four entries — `nllb-600M` and `aina-es-oc`, in both precisions — are
non-commercial and out of the default graph. See [licences.md](licences.md).

Sizes are the sum of the blobs an entry actually references, not the repo
total: these repos also carry a `decoder_model_merged.onnx` that `linguonnx`
never downloads.

## What is in the LID registry

| model_id | labels | size | licence |
|---|---|---|---|
| `glotlid` / `glotlid-int8` | 2102 | 1.68 GB / 425 MB | Apache-2.0 |
| `lid176` / `lid176-int8` | 176 | 131 MB / 33 MB | CC-BY-SA-3.0 |
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
# 150 8
```

## The download cache

Files are cached under `~/.cache/linguonnx/models/<model_id>/`. Downloads go
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

`--check` writes nothing and is the CI guard. It prints which entries are new
and which are stale, so "someone published a model and forgot the registry"
shows up as a failing check instead of a silent gap. Re-running the generator
on an unchanged Hub produces a byte-identical file, and hand-authored keys the
script does not generate — a curated `notes`, a pinned default — survive
regeneration.

### What is derived from where

Everything the Hub can answer is read from the Hub, never typed out:

| field | source |
|---|---|
| `arch` | `config.model_type` from the repo's own config; then the tokenizer files. A repo with `vocab.json` reads ids straight out of it (M2M100); one without uses fairseq's `id = sp_id + 1` (NLLB). |
| `license` | `cardData.license` when the card has YAML front matter, else the `**License:**` line in the README body — most opus-mt exports have no front matter. |
| `languages` | `additional_special_tokens` in the model's own `special_tokens_map.json`. |
| `pair` | The repo name, cross-checked against the base model named in the card. |
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

**Bilingual fine-tunes of multilingual bases.** A fine-tune keeps the whole
base tokenizer, so `special_tokens_map.json` still lists all 100 or 202
languages long after the weights stopped serving them. Those need a verified
entry in `BILINGUAL_FINETUNES` stating the pair in the model's own codes, and
are skipped with a printed reason until someone adds one. `aina-es-oc` is one
of these: an NLLB-600M fine-tune whose tokenizer claims 202 languages and whose
weights do Spanish into Aranese, one way.

There is a third hand-maintained fix-up, `MARIAN_MULTILINGUAL_OVERRIDES`, for a
model whose own code collides with a different language's ISO tag —
`liv4ever-mt`'s `<2li>` means Livonian, not Limburgish. Left alone, the router
would offer that model for a language it has never seen.
