import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *
import torchtune


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("traditional", [True, False])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rope_week2_batch_offset(
    stream: mx.Stream, traditional: bool, precision: np.dtype
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000.0
    with mx.stream(stream):
        for _ in range(100):
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = np.random.rand(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(
                precision
            )
            input_pos_user = [slice(i, i + SEQ_LEN) for i in range(BATCH_SIZE)]
            user_layer(mx.array(x), input_pos_user)
