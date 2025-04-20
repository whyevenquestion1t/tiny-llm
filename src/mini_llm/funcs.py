import mlx.core as mx
import math


def softmax(
    x: mx.array, axis: int, stream: mx.Stream | mx.Device | None = None
) -> mx.array:
    return mx.softmax(x, axis=axis, stream=stream)


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    attn_mask: mx.array | None = None,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """
    Compute scaled dot-product attention.

    query: batch_size x
    """
    factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    scores = (
        mx.matmul(query, key.swapaxes(-2, -1, stream=stream), stream=stream) * factor
    )
    if attn_mask is not None:
        scores = scores + attn_mask
    return mx.matmul(softmax(scores, axis=-1, stream=stream), value, stream=stream)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    if bias is not None:
        return mx.matmul(x, w.T, stream=stream) + bias
    else:
        return mx.matmul(x, w.T, stream=stream)
