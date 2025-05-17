import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *


@pytest.mark.parametrize("target", ["torch", "mlx"])
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_silu(stream: mx.Stream, precision: np.dtype, target: str):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = np.random.rand(BATCH_SIZE, DIM).astype(precision)
            user_output = silu(mx.array(x))
            if target == "torch":
                reference_output = torch.nn.functional.silu(
                    torch.tensor(x, device=TORCH_DEVICE)
                )
            else:
                reference_output = silu(mx.array(x))
            assert_allclose(user_output, reference_output, precision=precision)
