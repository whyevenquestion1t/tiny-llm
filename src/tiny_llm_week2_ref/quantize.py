import mlx.core as mx
from typing import Any

from extensions_ref import tiny_llm_ext_ref


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w

def qmm(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return tiny_llm_ext_ref.quantized_matmul(scales, biases, group_size, bits, a, b, transpose_b)

def quantized_linear(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    pass
