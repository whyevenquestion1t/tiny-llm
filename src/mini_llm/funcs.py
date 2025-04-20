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
    mask: mx.array | None = None,
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
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1, stream=stream), value, stream=stream)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """
    Compute scaled dot-product attention.

    query: batch_size x
    """
    factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    expected_shape = query.shape
    query = query.reshape(-1, query.shape[-3], query.shape[-2], query.shape[-1])
    key = key.reshape(-1, key.shape[-3], key.shape[-2], key.shape[-1])
    value = value.reshape(-1, value.shape[-3], value.shape[-2], value.shape[-1])
    B, H_q, L, E = query.shape
    _, H, S, _ = key.shape
    assert H_q % H == 0
    n_repeats = H_q // H
    query = query.reshape((B, H, n_repeats, L, E))
    key = key.reshape((B, H, 1, S, E))
    value = value.reshape((B, H, 1, S, E))
    scores = (
        mx.matmul(query, key.swapaxes(-2, -1, stream=stream), stream=stream) * factor
    )
    if mask is not None:
        mask = mask.reshape(-1, H, n_repeats, mask.shape[-2], mask.shape[-1])
        scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1, stream=stream), value, stream=stream)
    return result.reshape(expected_shape)


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
