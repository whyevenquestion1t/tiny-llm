import mlx.core as mx
from .qwen2 import Qwen2Model
from mlx_lm.tokenizer_utils import TokenizerWrapper


def simple_generate(model: Qwen2Model, tokenizer: TokenizerWrapper, prompt: str) -> str:
    pass
