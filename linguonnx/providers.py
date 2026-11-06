"""
ONNX Runtime execution-provider selection.

A provider list is resolved from the first of these that yields something:

1. an explicit ``providers=[...]`` argument (the caller controls the order,
   including its own fallbacks);
2. the ``LINGUONNX_ONNX_PROVIDERS`` environment variable — a comma-separated
   provider list, or ``auto``;
3. auto-detection — :data:`PREFERRED_PROVIDERS` intersected with what the
   installed ``onnxruntime`` build actually offers.

Requested providers that the installed runtime does not offer are dropped with
a warning, and ``CPUExecutionProvider`` is always appended, so a session never
fails because of the provider list. With nothing configured, resolution
auto-detects, and on a CPU-only ``onnxruntime`` build - the default install -
that still comes out to ``["CPUExecutionProvider"]``, so existing deployments
are unaffected.

GPU providers come from the ONNX Runtime build, not from linguonnx: the
default ``onnxruntime`` wheel is CPU-only (plus a few platform providers),
CUDA needs ``onnxruntime-gpu``, ROCm needs ``onnxruntime-rocm``, and DirectML
needs ``onnxruntime-directml``. ``onnxruntime`` and ``onnxruntime-gpu`` both
provide the same ``onnxruntime`` import namespace and must never be installed
together.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime

LOG = logging.getLogger(__name__)

#: A provider is either a name or a ``(name, options)`` pair.
ProviderSpec = Union[str, Tuple[str, Dict[str, Any]]]

CPU_PROVIDER = "CPUExecutionProvider"

#: Environment variable holding a comma-separated provider list, or ``auto``.
PROVIDERS_ENV_VAR = "LINGUONNX_ONNX_PROVIDERS"

#: Auto-detection preference order, best first. Only providers the installed
#: runtime reports as available are kept.
PREFERRED_PROVIDERS: List[str] = [
    "CUDAExecutionProvider",  # NVIDIA
    "ROCMExecutionProvider",  # AMD
    "MIGraphXExecutionProvider",  # AMD
    "DmlExecutionProvider",  # DirectML (Windows)
    "CoreMLExecutionProvider",  # Apple
    "OpenVINOExecutionProvider",  # Intel
    CPU_PROVIDER,
]

#: Default session options per provider, applied when a provider is requested
#: by name only.
PROVIDER_OPTIONS: Dict[str, Dict[str, Any]] = {
    "CUDAExecutionProvider": {"cudnn_conv_algo_search": "HEURISTIC"},
}


def available_providers() -> List[str]:
    """Providers offered by the installed ONNX Runtime build."""
    try:
        return list(onnxruntime.get_available_providers())
    except Exception as err:  # pragma: no cover - defensive, ORT always answers
        LOG.warning(f"could not query onnxruntime providers: {err}")
        return [CPU_PROVIDER]


def _name(provider: ProviderSpec) -> str:
    return provider[0] if isinstance(provider, tuple) else provider


def _with_options(provider: ProviderSpec) -> ProviderSpec:
    """Attach the default options of a provider given by bare name."""
    if isinstance(provider, tuple):
        return provider
    options = PROVIDER_OPTIONS.get(provider)
    return (provider, dict(options)) if options else provider


def _from_env() -> Optional[Sequence[str]]:
    raw = os.environ.get(PROVIDERS_ENV_VAR, "").strip()
    if not raw or raw.lower() == "auto":
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def _autodetect() -> List[ProviderSpec]:
    available = available_providers()
    return [_with_options(p) for p in PREFERRED_PROVIDERS if p in available]


def resolve_providers(
        providers: Optional[Sequence[ProviderSpec]] = None,
) -> List[ProviderSpec]:
    """
    Resolve the execution providers to run a session with.

    Parameters:
        providers: Ordered provider list; names or ``(name, options)`` pairs.
            When omitted, ``LINGUONNX_ONNX_PROVIDERS`` is consulted and
            otherwise the best available provider is auto-detected.

    Returns:
        A provider list, filtered to what the runtime offers and always ending
        in ``CPUExecutionProvider``.
    """
    if providers is None:
        providers = _from_env()

    if providers is None:
        resolved = _autodetect()
    else:
        available = available_providers()
        resolved = []
        for provider in providers:
            if _name(provider) in available:
                resolved.append(_with_options(provider))
            else:
                LOG.warning(
                    f"'{_name(provider)}' is not available in this onnxruntime "
                    f"build (available: {available}), skipping it. Install the "
                    f"matching onnxruntime package to enable it."
                )

    if not any(_name(p) == CPU_PROVIDER for p in resolved):
        resolved.append(CPU_PROVIDER)

    LOG.debug(f"onnxruntime execution providers: {[_name(p) for p in resolved]}")
    return resolved


def _warn_on_provider_fallback(
        requested: Sequence[ProviderSpec],
        session: "onnxruntime.InferenceSession",
) -> None:
    """Warn when the first requested provider silently fell back to another.

    onnxruntime does not raise when a provider fails to initialize (e.g. a
    CUDA build advertising ``CUDAExecutionProvider`` while missing a shared
    library like ``libcublasLt``); it just falls back to the next provider
    in the list with no log line. This surfaces that so a "GPU" run that is
    actually running on CPU is not mistaken for one that is not.
    """
    if not requested:
        return
    try:
        actual = list(session.get_providers())
    except Exception:  # pragma: no cover - defensive, ORT always answers
        return
    requested_first = _name(requested[0])
    if requested_first not in actual:
        LOG.warning(
            f"requested onnxruntime provider '{requested_first}' is not "
            f"active for this session; falling back to "
            f"'{actual[0] if actual else 'unknown'}' (active providers: {actual})"
        )


def make_session(
        model_path: Any,
        providers: Optional[Sequence[ProviderSpec]] = None,
        sess_options: Optional["onnxruntime.SessionOptions"] = None,
) -> "onnxruntime.InferenceSession":
    """Create an ``InferenceSession`` on the resolved execution providers."""
    resolved = resolve_providers(providers)
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=sess_options or onnxruntime.SessionOptions(),
        providers=resolved,
    )
    _warn_on_provider_fallback(resolved, session)
    return session
