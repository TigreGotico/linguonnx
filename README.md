# linguonnx
<!-- agent-hint: language identification and machine translation on ONNX Runtime, no torch. Verify the examples: `python scripts/check_docs.py README.md` (downloads models). -->

Language identification and machine translation on CPU, with ONNX Runtime and
no torch.

- `linguonnx.detect` identifies the language of a text with ONNX exports of
  four fastText classifiers: GlotLID, lid.176, OpenLID and OpenLID-v2.
- `linguonnx.translate` translates with Marian (opus-mt), M2M100, NLLB-200 and
  MADLAD exports. A routing graph selects the model, or the chain of models,
  for a language pair.

## Install

```bash
pip install linguonnx
```

## Quickstart

```python
from linguonnx import load_detector, load_translator

det = load_detector()            # glotlid-int8, 425 MB on first use
print(det.detect("Egun on, zer moduz?"))              # 'eu'
print(det.detect("وش لونك يا خوي", collapse_varieties=True))   # 'ar'

tx = load_translator()
print(tx.translate("bom dia, como estás?", src="pt", tgt="en"))
# 'Good morning, how are you?'

route = tx.route("pt", "eu", prefer="dedicated")
print(route.model_ids)      # ('aina-translator-pt-ca-int8', 'mt-hitz-ca-eu-int8')
print(route.pivots)         # ('ca',)
```

A pair that no single model covers goes through a third language. The `Route`
shows each model and each pivot language.

## Documentation

| Page | Contents |
| --- | --- |
| [docs/index.md](docs/index.md) | What ships, and the reading order |
| [docs/install.md](docs/install.md) | Extras, GPU, the model cache, OpenVoiceOS, development |
| [docs/detect.md](docs/detect.md) | Language identification |
| [docs/translate.md](docs/translate.md) | The translation API |
| [docs/routing.md](docs/routing.md) | How the router selects a route |
| [docs/models.md](docs/models.md) | The model registry |
| [docs/licences.md](docs/licences.md) | Licence tiers and non-commercial models |

## Licence

`linguonnx` is Apache-2.0. Some models have other licences: OpenLID is
GPL-3.0, and NLLB-200 is CC-BY-NC-4.0. The default translation graph has no
non-commercial model. See [docs/licences.md](docs/licences.md).
