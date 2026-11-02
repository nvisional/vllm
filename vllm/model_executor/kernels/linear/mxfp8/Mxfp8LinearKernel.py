# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)


@dataclass
class Mxfp8LinearLayerConfig:
    """Configuration for an MXFP8 linear layer.

    All MXFP8 layers share the same structure: FP8-E4M3 weights with
    uint8 (E8M0) per-block scales at block size 32.

    Attributes:
        weight_shape: The layer's `(out_features, in_features)`, i.e. `(N, K)`.
        activation_quant_key: Identifies the activation quantization format,
            or `None` when activations must not be quantized.

    """

    weight_shape: tuple[int, int]
    bmm_batch_size: int | None = None
    activation_quant_key: QuantKey | None = None


class Mxfp8LinearKernel(ABC):
    """Base class for MXFP8 quantized linear kernels.

    Each subclass implements a specific GEMM backend (FlashInfer CUTLASS,
    Marlin, emulation).
    The kernel selection mechanism iterates over registered subclasses in
    priority order, calling ``is_supported``, ``_supports_quant_scheme`` and
    ``can_implement`` to find the best match for the current hardware.
    """

    supports_pre_processed_weights: bool = False
    """True if ``process_weights_after_loading`` only rewrites parameters, so
    weights exported by the weight cache daemon can be used as-is."""

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c

    def input_quant_key(self) -> QuantKey | None:
        """Return the input quantization key supported by this kernel. If the kernel
        does not support input quantization outside of the kernel, return None.
        """
        return None

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    def _supports_quant_scheme(
        cls,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether this kernel implements *weight_key* x *activation_key*.

        Selection uses this to dispatch on the requested activation format
        (from the checkpoint or ``--quantization-config.linear.activation``).
        The default covers MXFP8 weights with MXFP8 or unquantized activations;
        kernels for other activation formats override it.
        """
        return weight_key == kMxfp8Static and activation_key in (None, kMxfp8Dynamic)

    @classmethod
    @abstractmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
