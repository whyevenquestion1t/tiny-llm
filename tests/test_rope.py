import pytest
import mlx.core as mx
import torch
from mini_llm.funcs import *
from mini_llm.layers import *
import numpy as np
from .utils import *
import torchtune


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_rope(stream: mx.Stream):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    NUM_KV_HEADS = 6
    HEAD_DIM = 4
    MAX_SEQ_LEN = 2
    BASE = 10000

    for _ in range(100):
        reference_layer = (
            torchtune.modules.position_embeddings.RotaryPositionalEmbeddings(
                HEAD_DIM,
                MAX_SEQ_LEN,
                BASE,
            )
        )
        user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE)
        x = np.random.rand(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        y = np.random.rand(BATCH_SIZE, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
        reference_output = reference_layer.forward(torch.tensor(x, device=TORCH_DEVICE))
        user_output = user_layer(mx.array(x), stream=stream)
        assert_allclose(user_output, reference_output, np.float32, rtol=1.0e-3)
        reference_output = reference_layer.forward(torch.tensor(y, device=TORCH_DEVICE))
        user_output = user_layer(mx.array(y), stream=stream)
        assert_allclose(user_output, reference_output, np.float32, rtol=1.0e-3)
