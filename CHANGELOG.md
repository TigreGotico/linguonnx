# Changelog

## [0.12.0a1](https://github.com/TigreGotico/linguonnx/tree/0.12.0a1) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.11.1a2...0.12.0a1)

**Merged pull requests:**

- feat: bound the model cache by size, and peak memory by concurrency [\#79](https://github.com/TigreGotico/linguonnx/pull/79) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.1a2](https://github.com/TigreGotico/linguonnx/tree/0.11.1a2) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.11.1a1...0.11.1a2)

**Merged pull requests:**

- docs: catalogue the aina-translator, nos-mt, m2m100\_418M African and eus/oci-cat model families [\#82](https://github.com/TigreGotico/linguonnx/pull/82) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.1a1](https://github.com/TigreGotico/linguonnx/tree/0.11.1a1) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.11.0a1...0.11.1a1)

**Merged pull requests:**

- fix: correct stale registry-derived numbers in docs, pin them with a test [\#81](https://github.com/TigreGotico/linguonnx/pull/81) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.0a1](https://github.com/TigreGotico/linguonnx/tree/0.11.0a1) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.10.0a1...0.11.0a1)

**Merged pull requests:**

- feat: per-language quality flags \(phase 1: schema, API, preservation\) [\#77](https://github.com/TigreGotico/linguonnx/pull/77) ([JarbasAl](https://github.com/JarbasAl))

## [0.10.0a1](https://github.com/TigreGotico/linguonnx/tree/0.10.0a1) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.9.4a1...0.10.0a1)

**Merged pull requests:**

- feat: max\_model\_mb deprioritises oversized models instead of deleting languages [\#75](https://github.com/TigreGotico/linguonnx/pull/75) ([JarbasAl](https://github.com/JarbasAl))

## [0.9.4a1](https://github.com/TigreGotico/linguonnx/tree/0.9.4a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.9.3a2...0.9.4a1)

**Merged pull requests:**

- fix: derive and validate ONNX external-data from graphs, not extra\_files [\#73](https://github.com/TigreGotico/linguonnx/pull/73) ([JarbasAl](https://github.com/JarbasAl))

## [0.9.3a2](https://github.com/TigreGotico/linguonnx/tree/0.9.3a2) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.9.3a1...0.9.3a2)

**Merged pull requests:**

- fix: a registry re-sync no longer destroys hand-verified data [\#71](https://github.com/TigreGotico/linguonnx/pull/71) ([JarbasAl](https://github.com/JarbasAl))
- Register the DSFSI Northern Sotho pair; record two arch gaps and two unengineable classifiers [\#70](https://github.com/TigreGotico/linguonnx/pull/70) ([JarbasAl](https://github.com/JarbasAl))

## [0.9.3a1](https://github.com/TigreGotico/linguonnx/tree/0.9.3a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.9.2a1...0.9.3a1)

## [0.9.2a1](https://github.com/TigreGotico/linguonnx/tree/0.9.2a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.9.1a1...0.9.2a1)

**Merged pull requests:**

- fix: sweep\_empty\_output.py must not substitute a fallback English sample [\#67](https://github.com/TigreGotico/linguonnx/pull/67) ([JarbasAl](https://github.com/JarbasAl))
- fix: OpenNMT-BPE segmentation, and an honest caveat for two looping fine-tunes [\#65](https://github.com/TigreGotico/linguonnx/pull/65) ([JarbasAl](https://github.com/JarbasAl))

## [0.9.1a1](https://github.com/TigreGotico/linguonnx/tree/0.9.1a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.9.0a1...0.9.1a1)

**Merged pull requests:**

- fix: three target-language and routing defects from the live quality sweep [\#64](https://github.com/TigreGotico/linguonnx/pull/64) ([JarbasAl](https://github.com/JarbasAl))

## [0.9.0a1](https://github.com/TigreGotico/linguonnx/tree/0.9.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.8.3a1...0.9.0a1)

**Merged pull requests:**

- feat: measured translation quality field in the registry [\#62](https://github.com/TigreGotico/linguonnx/pull/62) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.3a1](https://github.com/TigreGotico/linguonnx/tree/0.8.3a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.8.2a2...0.8.3a1)

**Merged pull requests:**

- fix: address M2M100 language tokens by name, not by list position [\#60](https://github.com/TigreGotico/linguonnx/pull/60) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.2a2](https://github.com/TigreGotico/linguonnx/tree/0.8.2a2) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.8.2a1...0.8.2a2)

**Merged pull requests:**

- test: fix a test module that never ran, and what it was hiding [\#58](https://github.com/TigreGotico/linguonnx/pull/58) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.2a1](https://github.com/TigreGotico/linguonnx/tree/0.8.2a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.8.1a1...0.8.2a1)

**Merged pull requests:**

- fix: keep each model's own language spelling alongside the routing tag [\#56](https://github.com/TigreGotico/linguonnx/pull/56) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.1a1](https://github.com/TigreGotico/linguonnx/tree/0.8.1a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.8.0a2...0.8.1a1)

**Merged pull requests:**

- fix: repair mangled version.py so the published version is the real one [\#53](https://github.com/TigreGotico/linguonnx/pull/53) ([JarbasAl](https://github.com/JarbasAl))
- fix: unify language-tag spelling systems in translation routing [\#40](https://github.com/TigreGotico/linguonnx/pull/40) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.0a2](https://github.com/TigreGotico/linguonnx/tree/0.8.0a2) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.8.0a1...0.8.0a2)

**Merged pull requests:**

- ci: pass PYPI\_TOKEN explicitly to the reusable publish workflows [\#51](https://github.com/TigreGotico/linguonnx/pull/51) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.0a1](https://github.com/TigreGotico/linguonnx/tree/0.8.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.7.1a1...0.8.0a1)

**Merged pull requests:**

- test: regression-guard IndicTrans2 state isolation across architectures [\#44](https://github.com/TigreGotico/linguonnx/pull/44) ([JarbasAl](https://github.com/JarbasAl))
- feat: specialist-institution provenance ranking tie-break [\#41](https://github.com/TigreGotico/linguonnx/pull/41) ([JarbasAl](https://github.com/JarbasAl))

## [0.7.1a1](https://github.com/TigreGotico/linguonnx/tree/0.7.1a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.7.0a1...0.7.1a1)

**Merged pull requests:**

- fix: resolve architecture/tokenizer mismatch for 19 unregistered models [\#47](https://github.com/TigreGotico/linguonnx/pull/47) ([JarbasAl](https://github.com/JarbasAl))

## [0.7.0a1](https://github.com/TigreGotico/linguonnx/tree/0.7.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.6.4a1...0.7.0a1)

**Merged pull requests:**

- feat: relocate the model cache with LINGUONNX\_CACHE [\#43](https://github.com/TigreGotico/linguonnx/pull/43) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.4a1](https://github.com/TigreGotico/linguonnx/tree/0.6.4a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.6.3a1...0.6.4a1)

**Merged pull requests:**

- fix: mt-hitz-gl-eu returned empty translations \(bad\_words\_ids not enforced\) [\#45](https://github.com/TigreGotico/linguonnx/pull/45) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.3a1](https://github.com/TigreGotico/linguonnx/tree/0.6.3a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.6.2a1...0.6.3a1)

**Merged pull requests:**

- fix: refuse language-family codes, bound coverage by the card, make routing budget-aware [\#38](https://github.com/TigreGotico/linguonnx/pull/38) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.2a1](https://github.com/TigreGotico/linguonnx/tree/0.6.2a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.6.1a2...0.6.2a1)

**Merged pull requests:**

- fix: MADLAD ignored the requested target language and returned English [\#36](https://github.com/TigreGotico/linguonnx/pull/36) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.1a2](https://github.com/TigreGotico/linguonnx/tree/0.6.1a2) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.6.1a1...0.6.1a2)

**Merged pull requests:**

- docs: point at the OVOS plugin that wraps this library [\#30](https://github.com/TigreGotico/linguonnx/pull/30) ([JarbasAl](https://github.com/JarbasAl))
- feat: bound routing by per-model download size [\#29](https://github.com/TigreGotico/linguonnx/pull/29) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.1a1](https://github.com/TigreGotico/linguonnx/tree/0.6.1a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.6.0a1...0.6.1a1)

**Merged pull requests:**

- fix: CI blockers for PyPI release \(network tests, registry sync, license false-positive, doc bugs\) [\#31](https://github.com/TigreGotico/linguonnx/pull/31) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.0a1](https://github.com/TigreGotico/linguonnx/tree/0.6.0a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.4a2...0.6.0a1)

**Merged pull requests:**

- feat: run IndicTrans2 and OpenNMT-BPE models instead of only routing them [\#27](https://github.com/TigreGotico/linguonnx/pull/27) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.4a2](https://github.com/TigreGotico/linguonnx/tree/0.5.4a2) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.4a1...0.5.4a2)

**Merged pull requests:**

- fix: validate a caller-supplied route before executing it [\#25](https://github.com/TigreGotico/linguonnx/pull/25) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.4a1](https://github.com/TigreGotico/linguonnx/tree/0.5.4a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.3a1...0.5.4a1)

**Merged pull requests:**

- fix: make routing agree with what translation can execute [\#23](https://github.com/TigreGotico/linguonnx/pull/23) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.3a1](https://github.com/TigreGotico/linguonnx/tree/0.5.3a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.2a1...0.5.3a1)

**Merged pull requests:**

- fix: bound untrusted input and validate decoding knobs [\#20](https://github.com/TigreGotico/linguonnx/pull/20) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.2a1](https://github.com/TigreGotico/linguonnx/tree/0.5.2a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.1...0.5.2a1)

**Merged pull requests:**

- fix: harden model download, caching and licence enforcement [\#19](https://github.com/TigreGotico/linguonnx/pull/19) ([JarbasAl](https://github.com/JarbasAl))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
