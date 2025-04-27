import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *
import torchtune

@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_quantized_linear(stream: mx.Stream, precision: np.dtype):
    with mx.stream(stream):
        input = mx.array(np.random.randn(10, 64).astype(precision))
        weight = mx.array(np.random.randn(30, 64).astype(precision))
        bias = mx.array(np.random.randn(30).astype(precision))
        w_q, scales, biases = mx.quantize(weight)
        out = quantized_linear(
            scales=scales,
            biases=biases,
            group_size=64,
            bits=4,
            x=input,
            w=w_q,
            bias=bias,
        )
