import pytest
import mlx.core as mx
import torch
from tiny_llm.funcs import *
from tiny_llm.layers import *
import numpy as np
from .utils import *
import torchtune


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("with_input", [True, False])
def test_rope(stream: mx.Stream, with_input: bool):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    NUM_KV_HEADS = 6
    HEAD_DIM = 4
    MAX_SEQ_LEN = 2
    BASE = 10000

    with mx.stream(stream):
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
            input_pos = (
                np.random.randint(0, MAX_SEQ_LEN, MAX_SEQ_LEN) if with_input else None
            )
            reference_output = reference_layer.forward(
                torch.tensor(x, device=TORCH_DEVICE),
                input_pos=torch.tensor(input_pos, device=TORCH_DEVICE)
                if input_pos is not None
                else None,
            )
            user_output = user_layer(
                mx.array(x), mx.array(input_pos) if input_pos is not None else None
            )
            assert_allclose(user_output, reference_output, np.float32, atol=1e-6)
            reference_output = reference_layer.forward(
                torch.tensor(y, device=TORCH_DEVICE),
                input_pos=torch.tensor(input_pos, device=TORCH_DEVICE)
                if input_pos is not None
                else None,
            )
            user_output = user_layer(
                mx.array(y), mx.array(input_pos) if input_pos is not None else None
            )
            assert_allclose(user_output, reference_output, np.float32, atol=1e-6)
