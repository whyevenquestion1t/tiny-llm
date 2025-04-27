import mlx.core as mx
from .qwen2 import Qwen2Model
from mlx_lm.tokenizer_utils import TokenizerWrapper


def simple_generate(model: Qwen2Model, tokenizer: TokenizerWrapper, prompt: str) -> str:
    def _step(model, y, offset):
        logits = model(y[None], offset)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    # generate/decode
    while True:
        token, _ = _step(model, tokens, tokens.size)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
