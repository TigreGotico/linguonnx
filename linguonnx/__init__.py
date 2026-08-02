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


__all__ = ["load_detector", "__version__"]
