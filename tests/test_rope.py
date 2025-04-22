import pytest
import mlx.core as mx
import torch
from tiny_llm.funcs import *
from tiny_llm.layers import *
import numpy as np
from .utils import *
import torchtune
import random


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("with_input", [True, False])
def test_rope(stream: mx.Stream, with_input: bool):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    NUM_KV_HEADS = 6
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000.0

    with mx.stream(stream):
        for _ in range(100):
            reference_layer = (
                torchtune.modules.position_embeddings.RotaryPositionalEmbeddings(
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    BASE,
                )
            )
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=True)
            x = np.random.rand(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)

            if with_input:
                input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
                input_pos_mx = input_pos
                input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
                input_pos_torch = torch.tensor([i for i in range(input_pos, input_pos + SEQ_LEN)], device=TORCH_DEVICE, dtype=torch.int32)
            else:
                input_pos = None
                input_pos_mx = None
                input_pos_user = None
                input_pos_torch = None

            reference_output = reference_layer.forward(
                torch.tensor(x, device=TORCH_DEVICE), input_pos=input_pos_torch
            )
            user_output = user_layer(mx.array(x), input_pos_user)
            assert_allclose(user_output, reference_output, np.float32, atol=1e-6)

            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=False)
            reference_output = mx.fast.rope(
                mx.array(x).transpose(0, 2, 1, 3),
                dims=HEAD_DIM,
                traditional=False,
                base=BASE,
                scale=1.0,
                offset=input_pos_mx or 0,
            ).transpose(0, 2, 1, 3)
            user_output = user_layer(mx.array(x), input_pos_user)
            assert_allclose(user_output, reference_output, np.float32, atol=1e-6)