import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from .funcs import (
    linear,
    scaled_dot_product_attention,
    scaled_dot_product_attention_grouped,
    silu,
)

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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
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
    ) -> mx.array:
        n_batches = query.shape[0]
        batch_size = query.shape[1]
        projection_q = (
            linear(query, self.wq)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(n_batches, self.num_heads * batch_size, self.head_dim)
            .transpose(1, 0, 2)
        )
        x = scaled_dot_product_attention(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(1, 0, 2).reshape(n_batches, batch_size, self.hidden_size)
        return linear(x, self.wo)


class Qwen2MultiHeadAttention:
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
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        orig_dtype = x.dtype
        projection_q = (
            linear(x, self.wq, bias=self.bq)
            .reshape(B, L, self.num_heads, self.head_dim)
            .astype(mx.float32)
        )
        projection_k = (
            linear(x, self.wk, bias=self.bk)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
            .astype(mx.float32)
        )
        projection_v = (
            linear(x, self.wv, bias=self.bv)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
            .astype(mx.float32)
        )
        offset = cache.offset
        projection_q = self.rope(projection_q, offset=slice(offset, offset + L))
        projection_k = self.rope(projection_k, offset=slice(offset, offset + L))
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        # TODO: it is possible to get a sensible result without using a kv-cache? Otherwise we have to include kv-cache in week 1.
        # mlx-lm's KvCache seems to do more than just caching, we could extract something out of it.
        projection_k, projection_v = cache.update_and_fetch(projection_k, projection_v)
        assert (
            projection_k.dtype == mx.float32
        )  # TODO: can we use float16? also a test framework to ensure all data types are casted correctly.
        assert projection_v.dtype == mx.float32
        x = scaled_dot_product_attention_grouped(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        ).astype(orig_dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        half_dims = dims // 2
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        freqs = mx.power(base, -inner)
        t = mx.arange(seq_len)
        freqs = mx.outer(t, freqs)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional
    def __call__(
        self, x: mx.array, offset: slice | None = None
    ) -> tuple[mx.array, mx.array]:
        # input x: (b, s, n_heads, head_dim)
        *N, S, H, D = x.shape
        # if offset is not None:
        #     assert len(offset) == S, f"offset {len(offset)} must be of length {s}"
        cos_basis = (
            self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
        )
        sin_basis = (
            self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]
        )
        # reshape x: (b, s, n_heads, head_dim // 2, 2)
        if self.traditional:
            x = x.reshape(*N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., 0:self.half_dims]
            x2 = x[..., self.half_dims:self.dims]
        # reshape basis: (1, s, 1, dims // 2, 2)
        cos_basis = cos_basis.reshape(S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(S, 1, self.half_dims)
        # manually doing complex number multiplication..
        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(*N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(*N, S, H, D)
        return y


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # TODO: tests to ensure the precision of this function
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        return (
            self.weight
            * x
            * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        ).astype(orig_dtype)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen2MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


def dequantize_linear(mx_layer: Any) -> tuple[mx.array, mx.array | None]:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class Qwen2Model:
    def __init__(
        self,
        mlx_model: Any,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        assert self.vocab_size > 0
        self.embed_tokens = mlx_model.model.embed_tokens
        self.layers_inner = []
        precision = mx.float16
        self.precision = precision

        for i in range(mlx_model.args.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq.astype(precision),
                wk=wk.astype(precision),
                wv=wv.astype(precision),
                wo=wo.astype(precision),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate.astype(precision),
                w_up=w_up.astype(precision),
                w_down=w_down.astype(precision),
                w_input_layernorm=mlx_model.model.layers[
                    i
                ].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, None, cache[layer])
        h = self.norm(h)
        return linear(h, self.w_lm_head)

    def sanitize(self, weights: dict):
        assert False, "not implemented"

    @property
    def layers(self):
        return self.layers_inner
