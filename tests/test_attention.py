import pytest
import mlx.core as mx
import torch
from mini_llm import attention, softmax
import numpy as np

AVAILABLE_STREAMS = [mx.cpu, mx.gpu]
AVAILABLE_STREAMS_IDS = ["cpu", "gpu"]
TORCH_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_softmax(stream: mx.Stream):
    BATCH_SIZE = 10
    DIM = 10
    for i in range(100):
        x = np.random.randn(BATCH_SIZE, DIM).astype(np.float32)
        user_output = softmax(mx.array(x, dtype=mx.float32), axis=-1, stream=stream)
        reference_output = torch.nn.functional.softmax(torch.Tensor(x, device=TORCH_DEVICE), dim=-1)
        assert_allclose(user_output, reference_output)

def torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    import math
    import torch
    """
    Compute scaled dot-product attention.

    query: batch_size x
    """
    factor = math.sqrt(1 / query.shape[-1])
    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    return torch.matmul(torch.nn.functional.softmax(scores, dim=-1), value)

def assert_allclose(a: mx.array, b: torch.Tensor):
    a = np.array(a)
    b = b.numpy()
    if not np.allclose(a, b, rtol=1.e-5, atol=1.e-8):
        print("a=", a)
        print("b=", b)
        print("diff=", a - b)
        assert False, f"result mismatch"

@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_attention(stream: mx.Stream):
    BATCH_SIZE = 3
    DIM_1 = 4
    DIM_2 = 5
    for _ in range(100):
        query = np.random.randn(BATCH_SIZE, DIM_1, DIM_2).astype(np.float32)
        key = np.random.randn(BATCH_SIZE, DIM_1, DIM_2).astype(np.float32)
        value = np.random.randn(BATCH_SIZE, DIM_1, DIM_2).astype(np.float32)
        reference_output = torch_attention(
            torch.Tensor(query, device=TORCH_DEVICE),
            torch.Tensor(key, device=TORCH_DEVICE),
            torch.Tensor(value, device=TORCH_DEVICE),
        )
        user_output = attention(
            mx.array(query, dtype=mx.float32),
            mx.array(key, dtype=mx.float32),
            mx.array(value, dtype=mx.float32),
            stream=stream,
        )
        assert_allclose(user_output, reference_output)
