import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rms_norm(
    stream: mx.Stream,
    precision: np.dtype,
):
    SIZE = 100
    SIZE_Y = 111
    with mx.stream(stream):
        for _ in range(100):
            data = np.random.rand(SIZE, SIZE_Y).astype(precision)
            weight = np.random.rand(SIZE_Y).astype(precision)
            eps = np.finfo(precision).eps
            reference_output = mx.fast.rms_norm(
                mx.array(data),
                mx.array(weight),
                eps=eps,
            )
            user_output = RMSNorm(SIZE_Y, mx.array(weight), eps=eps)(mx.array(data))
            assert_allclose(user_output, reference_output, precision)
