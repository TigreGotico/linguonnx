# Changelog

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

## [0.5.1](https://github.com/TigreGotico/linguonnx/tree/0.5.1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.1a1...0.5.1)

## [0.5.1a1](https://github.com/TigreGotico/linguonnx/tree/0.5.1a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.0a1...0.5.1a1)

**Merged pull requests:**

- Release 0.5.1a1 [\#18](https://github.com/TigreGotico/linguonnx/pull/18) ([github-actions[bot]](https://github.com/apps/github-actions))
- docs: split README into docs/, add examples, record short-text caveat [\#17](https://github.com/TigreGotico/linguonnx/pull/17) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.0a1](https://github.com/TigreGotico/linguonnx/tree/0.5.0a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.5.0...0.5.0a1)

## [0.5.0](https://github.com/TigreGotico/linguonnx/tree/0.5.0) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.4.0a1...0.5.0)

**Merged pull requests:**

- Release 0.5.0a1 [\#15](https://github.com/TigreGotico/linguonnx/pull/15) ([github-actions[bot]](https://github.com/apps/github-actions))
- feat: route the big multilingual models, and say when a licence blocks a pair [\#14](https://github.com/TigreGotico/linguonnx/pull/14) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.0a1](https://github.com/TigreGotico/linguonnx/tree/0.4.0a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.3.0a1...0.4.0a1)

**Merged pull requests:**

- Release 0.4.0a1 [\#13](https://github.com/TigreGotico/linguonnx/pull/13) ([github-actions[bot]](https://github.com/apps/github-actions))
- feat: generate the model registry from the HuggingFace API [\#12](https://github.com/TigreGotico/linguonnx/pull/12) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.0a1](https://github.com/TigreGotico/linguonnx/tree/0.3.0a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.2.0a1...0.3.0a1)

**Merged pull requests:**

- Release 0.3.0a1 [\#11](https://github.com/TigreGotico/linguonnx/pull/11) ([github-actions[bot]](https://github.com/apps/github-actions))
- feat: rank translation pivots by phonological distance [\#10](https://github.com/TigreGotico/linguonnx/pull/10) ([JarbasAl](https://github.com/JarbasAl))

## [0.2.0a1](https://github.com/TigreGotico/linguonnx/tree/0.2.0a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.2.0...0.2.0a1)

**Merged pull requests:**

- Release 0.2.0a1 [\#9](https://github.com/TigreGotico/linguonnx/pull/9) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.2.0](https://github.com/TigreGotico/linguonnx/tree/0.2.0) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.1.2a1...0.2.0)

**Merged pull requests:**

- feat: translation, designed graph-first [\#8](https://github.com/TigreGotico/linguonnx/pull/8) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.2a1](https://github.com/TigreGotico/linguonnx/tree/0.1.2a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.1.1a1...0.1.2a1)

**Merged pull requests:**

- Release 0.1.2a1 [\#7](https://github.com/TigreGotico/linguonnx/pull/7) ([github-actions[bot]](https://github.com/apps/github-actions))
- docs: link related projects [\#6](https://github.com/TigreGotico/linguonnx/pull/6) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.1a1](https://github.com/TigreGotico/linguonnx/tree/0.1.1a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.1.1...0.1.1a1)

**Merged pull requests:**

- Release 0.1.1a1 [\#5](https://github.com/TigreGotico/linguonnx/pull/5) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.1.1](https://github.com/TigreGotico/linguonnx/tree/0.1.1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.1.0...0.1.1)

**Merged pull requests:**

- chore: rename package to linguonnx [\#4](https://github.com/TigreGotico/linguonnx/pull/4) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.0](https://github.com/TigreGotico/linguonnx/tree/0.1.0) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.1.0a1...0.1.0)

## [0.1.0a1](https://github.com/TigreGotico/linguonnx/tree/0.1.0a1) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/0.0.0...0.1.0a1)

**Merged pull requests:**

- Release 0.1.0a1 [\#3](https://github.com/TigreGotico/linguonnx/pull/3) ([github-actions[bot]](https://github.com/apps/github-actions))
- feat: support lid.176, OpenLID and OpenLID-v2 [\#2](https://github.com/TigreGotico/linguonnx/pull/2) ([JarbasAl](https://github.com/JarbasAl))

## [0.0.0](https://github.com/TigreGotico/linguonnx/tree/0.0.0) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/linguonnx/compare/f7d9950aee8fd5f80ca9b28e8ba50693aa2dfac5...0.0.0)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
