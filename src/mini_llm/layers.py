import mlx.core as mx
from .funcs import linear, scaled_dot_product_attention


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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
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
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
            stream=stream,
        )
        x = x.transpose(1, 0, 2).reshape(n_batches, batch_size, self.hidden_size)
        return linear(x, self.wo, stream=stream)


class GroupedMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_kv_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_kv_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
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
            .reshape(n_batches, self.num_kv_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        projection_v = (
            linear(value, self.wv, stream=stream)
            .reshape(n_batches, self.num_kv_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        x = scaled_dot_product_attention(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
            stream=stream,
        )
        x = x.transpose(1, 0, 2).reshape(
            n_batches, batch_size, self.num_heads * batch_size
        )
        return linear(x, self.wo, stream=stream)
