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
    scores = (
        mx.matmul(query, key.swapaxes(-2, -1, stream=stream), stream=stream) * factor
    )
    return mx.matmul(softmax(scores, axis=-1, stream=stream), value, stream=stream)


def linear(
    x: mx.array, w: mx.array, stream: mx.Stream | mx.Device | None = None
) -> mx.array:
    return mx.matmul(x, w.T, stream=stream)


class MultiHeadAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        assert wq.shape == (d_model, d_model)
        assert wk.shape == (d_model, d_model)
        assert wv.shape == (d_model, d_model)
        assert wo.shape == (d_model, d_model)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        n_batches = query.shape[0]
        batch_size = query.shape[1]
        projection_q = (
            linear(query, self.wq, stream=stream)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .swapaxes(0, 1)
        )
        projection_k = (
            linear(key, self.wk, stream=stream)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .swapaxes(0, 1)
        )
        projection_v = (
            linear(value, self.wv, stream=stream)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .swapaxes(0, 1)
        )
        x = attention(projection_q, projection_k, projection_v, stream=stream)
        x = x.swapaxes(0, 1).reshape(n_batches, batch_size, self.d_model)
        return linear(x, self.wo, stream=stream)
