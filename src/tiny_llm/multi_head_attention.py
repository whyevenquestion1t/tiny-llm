import mlx.core as mx
from .basics import (
    linear,
)
from .attention import scaled_dot_product_attention
# TODO: add license for those heavily based on mlx-lm/PyTorch


class MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        pass

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        pass
