from lingonnx.version import __version__


def load_detector(model_id: str = "glotlid-int8"):
    """Load a language-identification model, downloading it on first use.

    ``model_id`` is one of the entries in ``lingonnx/model_index/lid.json``
    ("glotlid" for the fp32 graph, "glotlid-int8" for the quantized default).
    """
    from lingonnx.detect import LanguageDetector
    return LanguageDetector(model_id=model_id)


__all__ = ["load_detector", "__version__"]
