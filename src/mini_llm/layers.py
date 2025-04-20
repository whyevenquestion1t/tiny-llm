import mlx.core as mx
from .funcs import linear, scaled_dot_product_attention


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
            .transpose(1, 0, 2)
        )
        projection_k = (
            linear(key, self.wk, stream=stream)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        projection_v = (
            linear(value, self.wv, stream=stream)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        x = scaled_dot_product_attention(
            projection_q, projection_k, projection_v, stream=stream
        )
        x = x.transpose(1, 0, 2).reshape(n_batches, batch_size, self.d_model)
        return linear(x, self.wo, stream=stream)
