import mlx.core as mx
from .quantize import quantized_matmul
from typing import Any


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return mx.matmul(x, w.T) + bias
    else:
        return mx.matmul(x, w.T)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return (
            quantized_matmul(
                w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
            )
            + bias
        )
    else:
        return quantized_matmul(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )


def silu(x: mx.array) -> mx.array:
    return x / (1 + mx.exp(-x))
