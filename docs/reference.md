# Reference
<!-- agent-hint: every public name of linguonnx, from each module's __all__. Regenerate: `python scripts/docs_reference.py`. -->

Each table lists the names in the `__all__` of one module, with the first line of the docstring. A name with no docstring shows its kind only. The task pages explain how to use the names: [detect.md](detect.md), [translate.md](translate.md), [routing.md](routing.md) and [models.md](models.md).


## `linguonnx`

| Name | Summary |
| :--- | :--- |
| `__version__` | constant (`str`) |
| `load_detector` | Load a language-identification model, downloading it on first use. |
| `load_translator` | Build a translator over the registry in ``linguonnx/model_index/translate.json``. |

## `linguonnx.translate`

| Name | Summary |
| :--- | :--- |
| `GenerationConfig` | Decoding knobs. ``num_beams=1`` means greedy. |
| `Hop` | One model doing one directed language pair. |
| `NoRouteError` | No chain of models within ``max_hops`` connects the two languages. |
| `Route` | An ordered chain of hops from ``src`` to ``tgt``, plus the policy that chose it. |
| `Translator` | A routing graph plus the models it can route through. |
| `is_language_flagged` | Whether ``model_id`` carries a language flag for ``lang``. |
| `is_quality_flagged` | Whether ``model_id`` carries any :func:`quality_flag_reasons`. |
| `language_flag_reason_for` | :func:`language_flag_reason` for a registry id, **including its |
| `load_translator` | Build a :class:`Translator` over the registry. |
| `quality_flag_reasons_for` | :func:`quality_flag_reasons` for a registry id, looking up its own |

## `linguonnx.translate.graph`

| Name | Summary |
| :--- | :--- |
| `Capability` | What one model claims it can translate. |
| `DEFAULT_PIVOT_PREFERENCE` | constant (`tuple`) |
| `Hop` | One model doing one directed language pair. |
| `InvalidRouteError` | A caller-supplied :class:`Route` is not executable as written. |
| `LICENSE_TIERS` | constant (`dict`) |
| `MalformedTagError` | A caller passed something that is not a language tag at all. |
| `NoRouteError` | No chain of models within ``max_hops`` connects the two languages. |
| `PIVOT_RANKINGS` | constant (`tuple`) |
| `REGIONAL_PIVOTS` | constant (`dict`) |
| `Route` | An ordered chain of hops from ``src`` to ``tgt``, plus the policy that chose it. |
| `SPECIALIST_MAP` | constant (`dict`) |
| `TranslationGraph` | Resolve ``src -> tgt`` into a :class:`Route` over the registered models. |
| `UNSET` | constant (`_Unset`) |
| `entry_runnability` | ``(runnable, reason)`` for one ``translate.json`` entry. |
| `is_specialist_for` | Whether an org is *the* named specialist for either side of a pair. |
| `normalize_tag` | Normalise any language tag shape to the graph's BCP-47 node name. |

## `linguonnx.translate.models`

| Name | Summary |
| :--- | :--- |
| `TranslationModel` | A single loaded translation model. Graphs load lazily, on first use. |
| `capability_from_entry` | Turn one ``translate.json`` entry into a graph :class:`Capability`. |

## `linguonnx.translate.quality`

| Name | Summary |
| :--- | :--- |
| `LANGUAGE_FLAG_SIDES` | constant (`tuple`) |
| `QUALITY_ABSOLUTE_FLOOR` | constant (`float`) |
| `QUALITY_INT8_GAP` | constant (`float`) |
| `counterpart_model_id` | The registry id of the same export in the other precision. |
| `entry_target_languages` | Every BCP-47 tag ``entry`` claims it can **write**. |
| `flagged_languages` | Every language ``entry`` is flagged for in the direction ``side``. |
| `flagged_target_languages` | ``lang -> reasons`` for every language ``entries`` covers as a target |
| `is_language_flagged` | Whether ``model_id`` carries a language flag for ``lang``. |
| `is_quality_flagged` | Whether ``model_id`` carries any :func:`quality_flag_reasons`. |
| `language_flag_reason` | Why ``entry`` is flagged for ``lang``, or ``None`` when it is not. |
| `language_flag_reason_for` | :func:`language_flag_reason` for a registry id, **including its |
| `quality_flag_reasons` | Why ``entry`` is quality-flagged, or ``()`` when it is not. |
| `quality_flag_reasons_for` | :func:`quality_flag_reasons` for a registry id, looking up its own |

## `linguonnx.translate.distance`

| Name | Summary |
| :--- | :--- |
| `available` | True when orthography2ipa can be imported. |
| `clear_cache` | Drop the memo. For tests, and for reloading after an install. |
| `pair_distance` | Phonological distance between two language tags, lower = closer. |

## `linguonnx.translate.preprocess`

| Name | Summary |
| :--- | :--- |
| `Pipeline` | How one architecture turns text into ids and ids back into text. |
| `pipeline_for` | function |
| `register` | Class decorator: bind a :class:`Pipeline` to one or more ``arch`` names. |

## `linguonnx.translate.decode`

| Name | Summary |
| :--- | :--- |
| `CausalLMDecoder` | Greedy and beam search over one decoder-only ONNX Runtime session. |
| `GenerationConfig` | Decoding knobs. ``num_beams=1`` means greedy. |
| `Seq2SeqDecoder` | Greedy and beam-search generation over three ONNX Runtime sessions. |

## `linguonnx.translate.tokenizers`

| Name | Summary |
| :--- | :--- |
| `ByteLevelBpeTokenizer` | A decoder-only model's byte-level BPE ``tokenizer.json`` (the GPT-2 |
| `FastUnigramTokenizer` | Softcatalà's ``translate-eus-cat``/``translate-oci-cat``: a Marian- |
| `IndicTransTokenizer` | IndicTrans2. Two SentencePiece models and two plain JSON dictionaries. |
| `MBartTokenizer` | mBART: ``pieces + [</s>, src_lang]``. |
| `MarianTokenizer` | opus-mt. No language token: the model is the pair. |
| `OpenNmtBpeTokenizer` | Proxecto Nós ``nos-coda_iacobus``. Moses tokenisation + subword-nmt BPE. |
| `SpmSeq2SeqTokenizer` | M2M100 and NLLB. |
| `T5SpmTokenizer` | MADLAD (T5). One SentencePiece model; ids are its own piece ids. |
| `T5TextPrefixTokenizer` | An instruction-prefixed T5/UMT5. One SentencePiece model, and the task |
| `UnigramTextPrefixTokenizer` | The same instruction-prefixed T5, where the export ships a |
| `artifact_lang_codes` | The language codes *this export actually carries*, in its own id order. |
| `bare_lang_code` | ``__fr__`` -> ``fr``. Anything that is not a wrapped token -> ``None``. |
| `load_tokenizer` | Build the right tokenizer for ``arch`` from the downloaded ``files``. |
| `normalize_punctuation` | Moses punctuation normalisation, as Marian models were trained with. |
| `tokenizer_json_added_tokens` | The real ids of an export's added tokens, read from ``tokenizer.json``. |

## `linguonnx.limits`

| Name | Summary |
| :--- | :--- |
| `CorruptModelFileError` | A downloaded side file does not match the model it belongs to. |
| `DEFAULT_MAX_MODEL_MB` | constant (`int`) |
| `DecodeError` | Generation produced no usable sequence. |
| `EmptyInputError` | Input has no visible content, so there is nothing to work on. |
| `InputTooLongError` | Input crossed a configured size bound. |
| `MAX_DETECT_CHARS` | constant (`int`) |
| `MAX_ENCODER_TOKENS` | constant (`int`) |
| `MAX_LENGTH_PENALTY` | constant (`float`) |
| `MAX_LINE_TOKENS` | constant (`int`) |
| `MAX_MODEL_MB` | constant (`NoneType`) |
| `MAX_NUM_BEAMS` | constant (`int`) |
| `OutputTruncatedError` | Generation hit ``max_new_tokens`` before the model finished. |
| `check_length` | Raise :class:`InputTooLongError` if ``size`` is over ``limit``. |
| `has_visible_content` | True if ``text`` holds anything a reader would see. |

---
[← Prev](licences.md) · [Home](index.md)
