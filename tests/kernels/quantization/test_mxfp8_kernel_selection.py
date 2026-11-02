# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp8_kernel_selection.py`.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from vllm.model_executor.kernels.linear import (
    EmulationMxfp8LinearKernel,
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    MarlinMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test

# Kernels backed by FlashInfer mm_mxfp8, which requires N, K >= 128 and
# K % 32 == 0.
_MM_MXFP8_KERNELS = [
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
]

_SUPPORTED_SHAPE = (4096, 4096)
# (N, K): N < 128, K < 128, K % 32 != 0.
_UNSUPPORTED_SHAPES = [(64, 4096), (4096, 64), (4096, 4112)]

# (N, K) with N < 128 or K < 128, which only Marlin runs.
_MARLIN_ONLY_SHAPES = [(32, 256), (64, 256), (96, 256), (256, 64)]
# (N, K): N <= 0, K <= 0, K % 32 != 0.
_MARLIN_UNSUPPORTED_SHAPES = [(0, 256), (256, 0), (256, 100)]

# W8A8 (dynamic activation) selection.
# Marlin is weight-only and rejects kMxfp8Dynamic, so shapes FlashInfer cannot
# implement fall through to Emulation rather than Marlin.
_SELECTION_CASES = [
    (100, (128, 128), FlashInferCutedslMxfp8LinearKernel),
    (100, (130, 256), FlashInferCutedslMxfp8LinearKernel),
    (103, (128, 128), FlashInferCutedslMxfp8LinearKernel),
    (103, (130, 256), FlashInferCutedslMxfp8LinearKernel),
    (120, (128, 128), FlashInferCutlassMxfp8LinearKernel),
    (120, (130, 256), EmulationMxfp8LinearKernel),
    (121, (128, 128), FlashInferCutlassMxfp8LinearKernel),
    (121, (130, 256), EmulationMxfp8LinearKernel),
    *[
        (capability, shape, EmulationMxfp8LinearKernel)
        for capability in (100, 103, 120, 121)
        for shape in _MARLIN_ONLY_SHAPES
    ],
]

# Weight-only (activation_quant_key=None) must skip self-quantizing W8A8 kernels.
_WEIGHT_ONLY_SELECTION_CASES = [
    (100, (128, 128), MarlinMxfp8LinearKernel),
    (100, (4096, 4096), MarlinMxfp8LinearKernel),
    (103, (128, 128), MarlinMxfp8LinearKernel),
    (120, (128, 128), MarlinMxfp8LinearKernel),
]


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
def test_mm_mxfp8_kernels_accept_supported_shape(kernel_cls):
    config = Mxfp8LinearLayerConfig(weight_shape=_SUPPORTED_SHAPE)
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
@pytest.mark.parametrize("weight_shape", _UNSUPPORTED_SHAPES)
def test_mm_mxfp8_kernels_reject_unsupported_shape(kernel_cls, weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("weight_shape", _MARLIN_UNSUPPORTED_SHAPES)
def test_marlin_mxfp8_rejects_unsupported_shape(weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = MarlinMxfp8LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


@contextmanager
def _patch_cuda_platform(capability: int):
    """Patch the platform so kernel selection runs as on a CUDA GPU with the
    given compute capability. The FlashInfer and Marlin kernels report as
    supported, CuTe-DSL only on SM100/103."""
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.current_platform."
            "is_device_capability_family",
            side_effect=lambda cap, device_id=0: cap // 10 == capability // 10,
        ),
        patch.object(
            FlashInferCutedslMxfp8LinearKernel,
            "is_supported",
            return_value=(capability in (100, 103), None),
        ),
        patch.object(
            FlashInferCutlassMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(
            MarlinMxfp8LinearKernel, "is_supported", return_value=(True, None)
        ),
    ):
        platform._enum = PlatformEnum.CUDA
        yield


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(100, True), (103, True), (120, False), (121, False)],
)
def test_cutlass_mxfp8_requires_aligned_n_only_on_sm12x(capability, expected):
    """(130, 256) passes the generic mm_mxfp8 limits. Only the SM12x CUTLASS
    backend needs N % 32 == 0, so SM100/103 must keep accepting it."""
    config = Mxfp8LinearLayerConfig(weight_shape=(130, 256))
    with _patch_cuda_platform(capability):
        can_implement, _ = FlashInferCutlassMxfp8LinearKernel.can_implement(config)
    assert can_implement == expected


@pytest.mark.parametrize(
    ("capability", "weight_shape", "expected_kernel_cls"), _SELECTION_CASES
)
def test_init_mxfp8_linear_kernel(capability, weight_shape, expected_kernel_cls):
    """A layer that mm_mxfp8 cannot handle must fall through to the next
    kernel in the CUDA priority list instead of being selected and failing
    in apply_weights.

    Existing cases historically assumed W8A8; pass ``kMxfp8Dynamic`` so
    FlashInfer remains selectable after activation-scheme gating.
    """
    with _patch_cuda_platform(capability):
        kernel = init_mxfp8_linear_kernel(
            weight_shape=weight_shape,
            activation_quant_key=kMxfp8Dynamic,
        )

    assert isinstance(kernel, expected_kernel_cls)
    assert kernel.config == Mxfp8LinearLayerConfig(
        weight_shape=weight_shape,
        activation_quant_key=kMxfp8Dynamic,
    )


@pytest.mark.parametrize(
    ("capability", "weight_shape", "expected_kernel_cls"),
    _WEIGHT_ONLY_SELECTION_CASES,
)
def test_init_mxfp8_weight_only_skips_flashinfer(
    capability, weight_shape, expected_kernel_cls
):
    """Weight-only Quark MXFP8 (activation_quant_key=None) must not select a
    self-quantizing FlashInfer W8A8 kernel on SM100."""
    with _patch_cuda_platform(capability):
        kernel = init_mxfp8_linear_kernel(
            weight_shape=weight_shape,
            activation_quant_key=None,
        )

    assert isinstance(kernel, expected_kernel_cls)
    assert kernel.input_quant_key() is None
    assert kernel.config == Mxfp8LinearLayerConfig(
        weight_shape=weight_shape,
        activation_quant_key=None,
    )


@pytest.mark.parametrize("weight_shape", [(128, 128), (4096, 4096)])
def test_init_mxfp8_dynamic_still_selects_flashinfer_cutedsl_on_sm100(weight_shape):
    with _patch_cuda_platform(100):
        kernel = init_mxfp8_linear_kernel(
            weight_shape=weight_shape,
            activation_quant_key=kMxfp8Dynamic,
        )

    assert isinstance(kernel, FlashInferCutedslMxfp8LinearKernel)
    assert kernel.input_quant_key() is kMxfp8Dynamic
