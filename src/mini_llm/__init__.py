import mlx.core as mx
import math

def softmax(
    x: mx.array, axis: int, stream: mx.Stream | mx.Device | None = None
) -> mx.array:
    return mx.softmax(x, axis=axis, stream=stream)


def attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """
    Compute scaled dot-product attention.

    query: batch_size x
    """
    factor = math.sqrt(1 / query.shape[-1])
    scores = mx.matmul(
        query, key.swapaxes(-2, -1, stream=stream), stream=stream
    ) * factor
    return mx.matmul(softmax(scores, axis=-1, stream=stream), value, stream=stream)
