import os
import unittest
from unittest.mock import MagicMock, patch

from linguonnx import providers
from linguonnx.providers import (
    CPU_PROVIDER,
    PROVIDERS_ENV_VAR,
    make_session,
    resolve_providers,
)

CPU_ONLY = ["CPUExecutionProvider"]
ROCM_BOX = ["ROCMExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider"]
CUDA_BOX = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _available(names):
    """Pretend the installed onnxruntime build offers exactly *names*."""
    return patch.object(providers, "available_providers", lambda: list(names))


def _names(resolved):
    return [p[0] if isinstance(p, tuple) else p for p in resolved]


class TestDefaultIsCpuOnly(unittest.TestCase):
    """The regression this whole module exists to prevent: with nothing
    configured, resolution must still land on CPU-only, so an existing
    deployment pinning this package is unaffected."""

    def setUp(self):
        os.environ.pop(PROVIDERS_ENV_VAR, None)

    def test_default_resolves_to_cpu_only_on_a_cpu_build(self):
        with _available(CPU_ONLY):
            resolved = resolve_providers()
        self.assertEqual(_names(resolved), [CPU_PROVIDER])

    def test_default_prefers_gpu_when_the_runtime_offers_it(self):
        with _available(CUDA_BOX):
            resolved = resolve_providers()
        self.assertEqual(_names(resolved), ["CUDAExecutionProvider", CPU_PROVIDER])


class TestExplicitProviders(unittest.TestCase):
    def setUp(self):
        os.environ.pop(PROVIDERS_ENV_VAR, None)

    def test_explicit_list_is_kept_in_order(self):
        with _available(ROCM_BOX):
            resolved = resolve_providers(
                ["MIGraphXExecutionProvider", "ROCMExecutionProvider", CPU_PROVIDER])
        self.assertEqual(
            _names(resolved),
            ["MIGraphXExecutionProvider", "ROCMExecutionProvider", CPU_PROVIDER])

    def test_cpu_fallback_is_appended_when_omitted(self):
        with _available(ROCM_BOX):
            resolved = resolve_providers(["ROCMExecutionProvider"])
        self.assertEqual(_names(resolved), ["ROCMExecutionProvider", CPU_PROVIDER])

    def test_provider_options_are_preserved(self):
        with _available(CUDA_BOX):
            resolved = resolve_providers(
                [("CUDAExecutionProvider", {"device_id": 1})])
        self.assertEqual(resolved[0], ("CUDAExecutionProvider", {"device_id": 1}))

    def test_named_cuda_gets_default_options(self):
        with _available(CUDA_BOX):
            resolved = resolve_providers(["CUDAExecutionProvider"])
        self.assertEqual(
            resolved[0],
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))


class TestAutoDetection(unittest.TestCase):
    def setUp(self):
        os.environ.pop(PROVIDERS_ENV_VAR, None)

    def test_best_available_provider_wins(self):
        with _available(ROCM_BOX):
            resolved = resolve_providers()
        self.assertEqual(
            _names(resolved),
            ["ROCMExecutionProvider", "MIGraphXExecutionProvider", CPU_PROVIDER])

    def test_unknown_available_providers_are_ignored(self):
        with _available(["MadeUpExecutionProvider", "CPUExecutionProvider"]):
            resolved = resolve_providers()
        self.assertEqual(_names(resolved), [CPU_PROVIDER])


class TestUnavailableProviderFallback(unittest.TestCase):
    def setUp(self):
        os.environ.pop(PROVIDERS_ENV_VAR, None)

    def test_unavailable_provider_is_dropped_with_a_warning(self):
        with _available(CPU_ONLY):
            with patch.object(providers.LOG, "warning") as warn:
                resolved = resolve_providers(["ROCMExecutionProvider"])
        self.assertEqual(_names(resolved), [CPU_PROVIDER])
        warned = " ".join(str(c) for c in warn.call_args_list)
        self.assertIn("ROCMExecutionProvider", warned)

    def test_partially_available_list_keeps_what_works(self):
        with _available(CUDA_BOX):
            resolved = resolve_providers(
                ["ROCMExecutionProvider", "CUDAExecutionProvider", CPU_PROVIDER])
        self.assertEqual(_names(resolved), ["CUDAExecutionProvider", CPU_PROVIDER])


class TestEnvVar(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(PROVIDERS_ENV_VAR, None)

    def test_env_var_provider_list(self):
        with patch.dict("os.environ",
                        {PROVIDERS_ENV_VAR: "ROCMExecutionProvider, CPUExecutionProvider"}):
            with _available(ROCM_BOX):
                resolved = resolve_providers()
        self.assertEqual(_names(resolved), ["ROCMExecutionProvider", CPU_PROVIDER])

    def test_env_var_auto_means_autodetect(self):
        with patch.dict("os.environ", {PROVIDERS_ENV_VAR: "auto"}):
            with _available(ROCM_BOX):
                resolved = resolve_providers()
        self.assertEqual(_names(resolved)[0], "ROCMExecutionProvider")

    def test_explicit_providers_win_over_env_var(self):
        with patch.dict("os.environ", {PROVIDERS_ENV_VAR: "CPUExecutionProvider"}):
            with _available(CUDA_BOX):
                resolved = resolve_providers(["CUDAExecutionProvider"])
        self.assertEqual(_names(resolved)[0], "CUDAExecutionProvider")

    def test_unavailable_env_provider_warns_and_falls_back(self):
        with patch.dict("os.environ", {PROVIDERS_ENV_VAR: "DmlExecutionProvider"}):
            with _available(CPU_ONLY):
                with patch.object(providers.LOG, "warning") as warn:
                    resolved = resolve_providers()
        self.assertEqual(_names(resolved), [CPU_PROVIDER])
        self.assertTrue(warn.called)


class TestMakeSession(unittest.TestCase):
    def setUp(self):
        os.environ.pop(PROVIDERS_ENV_VAR, None)

    def test_session_is_built_with_resolved_providers(self):
        with _available(ROCM_BOX):
            with patch.object(providers.onnxruntime, "InferenceSession") as session:
                make_session("model.onnx", providers=["ROCMExecutionProvider"])
        _, kwargs = session.call_args
        self.assertEqual(
            _names(kwargs["providers"]), ["ROCMExecutionProvider", CPU_PROVIDER])

    def test_default_make_session_is_cpu_only_on_a_cpu_build(self):
        with _available(CPU_ONLY):
            with patch.object(providers.onnxruntime, "InferenceSession") as session:
                make_session("model.onnx")
        _, kwargs = session.call_args
        self.assertEqual(_names(kwargs["providers"]), [CPU_PROVIDER])

    def test_fallback_warning_fires_when_active_first_provider_differs(self):
        fake_session = MagicMock()
        fake_session.get_providers.return_value = [CPU_PROVIDER]
        with _available(CUDA_BOX):
            with patch.object(providers.onnxruntime, "InferenceSession",
                              return_value=fake_session):
                with patch.object(providers.LOG, "warning") as warn:
                    make_session("model.onnx", providers=["CUDAExecutionProvider"])
        self.assertTrue(warn.called)
        warned = " ".join(str(c) for c in warn.call_args_list)
        self.assertIn("CUDAExecutionProvider", warned)

    def test_no_fallback_warning_when_requested_provider_is_active(self):
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider", CPU_PROVIDER]
        with _available(CUDA_BOX):
            with patch.object(providers.onnxruntime, "InferenceSession",
                              return_value=fake_session):
                with patch.object(providers.LOG, "warning") as warn:
                    make_session("model.onnx", providers=["CUDAExecutionProvider"])
        self.assertFalse(warn.called)


if __name__ == "__main__":
    unittest.main()
