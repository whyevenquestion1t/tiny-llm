import math
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


class QwenMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_kv_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_kv_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        assert bq.shape == (hidden_size,)
        assert bk.shape == (hidden_size,)
        assert bv.shape == (hidden_size,)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        projection_q = (
            linear(x, self.wq, bias=self.bq, stream=stream)
            .reshape(B, L, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(x, self.wk, bias=self.bk, stream=stream)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(x, self.wv, bias=self.bv, stream=stream)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
            stream=stream,
        )
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo, stream=stream)


class RoPE:
    def __init__(self, dims: int, seq_len: int, base: int = 10000):
        self.dims = dims
        self.seq_len = seq_len
        freqs = 1.0 / (base ** (mx.arange(0, dims, 2)[: (dims // 2)] / dims))
        t = mx.arange(seq_len)
        freqs = mx.outer(t, freqs)
        self.basis = mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)
        assert self.basis.shape == (seq_len, dims // 2, 2)

    def __call__(
        self, x: mx.array, stream: mx.Stream | mx.Device | None = None
    ) -> tuple[mx.array, mx.array]:
        # input x: (b, s, n_heads, head_dim)
        orig_shape = x.shape
        s = x.shape[-3]
        basis = self.basis[:s, :]
        # reshape x: (b, s, n_heads, head_dim // 2, 2)
        x = x.reshape(*x.shape[:-1], -1, 2)
        # reshape basis: (1, s, 1, dims // 2, 2)
        basis = basis.reshape(1, s, 1, self.dims // 2, 2)
        real = x[..., 0] * basis[..., 0] - x[..., 1] * basis[..., 1]
        imag = x[..., 1] * basis[..., 0] + x[..., 0] * basis[..., 1]
        y = mx.stack([real, imag], axis=-1)
        y = y.reshape(orig_shape)
        return y
