import pytest
import mlx.core as mx
import torch
from tiny_llm.funcs import *
from tiny_llm.layers import *
import numpy as np
from .utils import *
import torchtune


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_silu(stream: mx.Stream, precision: np.dtype):
    SIZE = 100

    with mx.stream(stream):
        for _ in range(100):
            data = np.random.rand(SIZE).astype(precision)
            reference_output = torch.nn.functional.silu(
                torch.tensor(data, device=TORCH_DEVICE)
            )
            user_output = silu(mx.array(data))
            assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rms_norm(stream: mx.Stream, precision: np.dtype):
    SIZE = 100
    SIZE_Y = 111
    with mx.stream(stream):
        for _ in range(100):
            data = np.random.rand(SIZE, SIZE_Y).astype(precision)
            weight = np.random.rand(SIZE_Y).astype(precision)
            eps = np.finfo(precision).eps
            reference_output = torch.nn.functional.rms_norm(
                torch.tensor(data, device=TORCH_DEVICE),
                (SIZE_Y,),
                torch.tensor(weight, device=TORCH_DEVICE),
                eps=eps,
            )
            user_output = RMSNorm(SIZE_Y, mx.array(weight), eps=eps)(mx.array(data))
            assert_allclose(user_output, reference_output, precision)
