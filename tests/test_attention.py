import pytest
import mlx.core as mx
import torch
from mini_llm import attention, softmax, MultiHeadAttention
import numpy as np

AVAILABLE_STREAMS = [mx.cpu, mx.gpu]
AVAILABLE_STREAMS_IDS = ["cpu", "gpu"]
PRECISIONS = [np.float32, np.float16]
PRECISION_IDS = ["f32", "f16"]
TORCH_DEVICE = torch.device("cpu")

def assert_allclose(a: mx.array, b: torch.Tensor, precision: np.dtype):
    a = np.array(a)
    b = b.cpu().numpy()
    if precision == np.float32:
        rtol = 1.0e-5
        atol = 1.0e-8
    elif precision == np.float16:
        rtol = 1.0e-2
        atol = 1.0e-4
    assert a.shape == b.shape
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        print("a=", a)
        print("b=", b)
        diff = np.invert(np.isclose(a, b, rtol=rtol, atol=atol))
        print("diff_a=", a * diff)
        print("diff_b=", b * diff)
        assert False, f"result mismatch"


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_softmax(stream: mx.Stream, precision: np.dtype):
    BATCH_SIZE = 10
    DIM = 10
    for _ in range(100):
        x = np.random.rand(BATCH_SIZE, DIM).astype(precision)
        user_output = softmax(mx.array(x), axis=-1, stream=stream)
        reference_output = torch.nn.functional.softmax(
            torch.tensor(x, device=TORCH_DEVICE), dim=-1
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_attention(stream: mx.Stream, precision: np.dtype):
    BATCH_SIZE = 3
    DIM_N = 4
    DIM_M = 5
    for _ in range(100):
        query = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
        key = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
        value = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
        reference_output = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(query, device=TORCH_DEVICE),
            torch.tensor(key, device=TORCH_DEVICE),
            torch.tensor(value, device=TORCH_DEVICE),
        )
        user_output = attention(
            mx.array(query),
            mx.array(key),
            mx.array(value),
            stream=stream,
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_multi_head_attention(stream: mx.Stream, precision: np.dtype):
    BATCH_SIZE = 3
    DIM_N = 4
    DIM_M = 9
    NUM_HEADS = 3
    for _ in range(100):
        query = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
        key = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
        value = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
        q_proj_weight = np.random.rand(DIM_M, DIM_M).astype(precision)
        k_proj_weight = np.random.rand(DIM_M, DIM_M).astype(precision)
        v_proj_weight = np.random.rand(DIM_M, DIM_M).astype(precision)
        out_proj_weight = np.random.rand(DIM_M, DIM_M).astype(precision)
        reference_output, _ = torch.nn.functional.multi_head_attention_forward(
            torch.tensor(query, device=TORCH_DEVICE),
            torch.tensor(key, device=TORCH_DEVICE),
            torch.tensor(value, device=TORCH_DEVICE),
            num_heads=NUM_HEADS,
            q_proj_weight=torch.tensor(q_proj_weight, device=TORCH_DEVICE),
            k_proj_weight=torch.tensor(k_proj_weight, device=TORCH_DEVICE),
            v_proj_weight=torch.tensor(v_proj_weight, device=TORCH_DEVICE),
            out_proj_weight=torch.tensor(out_proj_weight, device=TORCH_DEVICE),
            embed_dim_to_check=DIM_M,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_bias=None,
            use_separate_proj_weight=True,
        )
        user_output = MultiHeadAttention(
            DIM_M,
            NUM_HEADS,
            mx.array(q_proj_weight),
            mx.array(k_proj_weight),
            mx.array(v_proj_weight),
            mx.array(out_proj_weight),
        )(
            mx.array(query),
            mx.array(key),
            mx.array(value),
            stream=stream,
        )
        assert_allclose(user_output, reference_output, precision=precision)
