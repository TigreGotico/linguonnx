from linguonnx.version import __version__


def load_detector(model_id: str = "glotlid-int8"):
    """Load a language-identification model, downloading it on first use.

    ``model_id`` is one of the entries in ``linguonnx/model_index/lid.json``:
    "glotlid"/"glotlid-int8" (Apache-2.0, the default), "lid176"/"lid176-int8"
    (CC-BY-SA-3.0), "openlid"/"openlid-int8" and "openlid-v2"/"openlid-v2-int8"
    (both GPL-3.0). The default is GlotLID because it is the only
    Apache-2.0-licensed option; the others must be asked for by name.
    """
    from linguonnx.detect import LanguageDetector
    return LanguageDetector(model_id=model_id)


def load_translator(*args, **kwargs):
    """Build a translator over the registry in ``linguonnx/model_index/translate.json``.

    See :func:`linguonnx.translate.load_translator` for the full signature. The
    default graph is every permissive-licensed int8 model - M2M100-418M plus
    the opus-mt bilingual pairs - routed by fewest hops, capped at two.
    """
    from linguonnx.translate import load_translator as _load
    return _load(*args, **kwargs)


__all__ = ["load_detector", "load_translator", "__version__"]
