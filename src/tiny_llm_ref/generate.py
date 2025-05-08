import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .kv_cache import *
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2


def simple_generate(
    model: Qwen2ModelWeek1, tokenizer: TokenizerWrapper, prompt: str
) -> str:
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
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    offset = 0
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    # generate/decode
    while True:
        token, _ = _step(model, tokens, offset, kv_cache)
        mx.eval(token)
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
        if token.item() == tokenizer.eos_token_id:
            break
        offset += tokens.size
        tokens = token


def batch_generate(model: any, tokenizer: TokenizerWrapper, prompts: list[str]):
    MAX_REQUESTS = 5
    is_idle = [True] * MAX_REQUESTS
    next_tokens = [None] * MAX_REQUESTS
    offsets = [None] * MAX_REQUESTS
    detokenizers = [None] * MAX_REQUESTS
    kv_cache = [
        BatchingKvCache(max_active_requests=MAX_REQUESTS)
        for _ in range(model.num_hidden_layers)
    ]

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    def prefill(model: any, tokenizer: TokenizerWrapper, prompt: str):
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        token, _ = _step(model, tokens, 0, kv_cache)
        mx.eval(token)
        return token, kv_cache, len(tokens)

    print(f"Processing {len(prompts)} prompts")
    prompts = enumerate(prompts)
    more_prompts = True
    while True:
        # prefill until no idle slots
        while any(is_idle) and more_prompts:
            try:
                idx, prompt = next(prompts)
            except StopIteration:
                more_prompts = False
                break
            token, prefill_kv_cache, offset = prefill(model, tokenizer, prompt)
            if token.item() == tokenizer.eos_token_id:
                # if the first token is eos, we skip this prompt
                print("Request finished without producing any token")
                continue
            for i in range(MAX_REQUESTS):
                if is_idle[i]:
                    detokenizers[i] = tokenizer.detokenizer.__class__(
                        tokenizer._tokenizer
                    )
                    detokenizers[i].add_token(token.item())
                    print(f"Prefilling prompt {idx} at request {i}")
                    print(f"{idx}: " + detokenizers[i].last_segment, end="")
                    is_idle[i] = False
                    for prefill_cache, batch_cache in zip(prefill_kv_cache, kv_cache):
                        batch_cache.add_request(prefill_cache, i)
                    next_tokens[i] = (token, offset)
                    break

        # decode
        print(next_tokens)
        next_tokens, _ = _step(model, mx.array(next_tokens), offsets, kv_cache)
        for i in range(MAX_REQUESTS):
            if not is_idle[i]:
                offsets[i] += 1  # we decode one token at a time
                detokenizers[i].add_token(next_tokens[i].item())
                print(f"{idx}: " + detokenizers[i].text, end="")
                if next_tokens[i].item() == tokenizer.eos_token_id:
                    print(f"Removing request {i}")
                    batch_cache.remove_request(i)
                    is_idle[i] = True
                    continue
