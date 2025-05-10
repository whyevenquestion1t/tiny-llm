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
    prefill_max = 64
    total_tokens = tokens.size
    while tokens.size > prefill_max:
        token, _ = _step(model, tokens[:prefill_max], offset, kv_cache)
        for i in kv_cache:
            mx.eval(i.key_values[0])
            mx.eval(i.key_values[1])
        offset += prefill_max
        tokens = tokens[prefill_max:]
        print(f"Prefill progress: {offset}/{total_tokens}", flush=True)
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


def _step(model, y, offsets, kv_cache):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    sampler = lambda x: mx.argmax(x, axis=-1)
    y = sampler(logprobs)
    return y


class PrefillRequest:
    def __init__(
        self, model: any, tokenizer: TokenizerWrapper, prompt: str, max_step: int = 16
    ):
        self.prompt = prompt
        self.kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        self.model = model
        self.prefill_tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
        self.offset = 0
        self.max_step = max_step

    def prefill(self):
        # returns None if prefill is not done
        tokens_to_prefill = min(self.max_step, self.prefill_tokens.size - self.offset)
        token = _step(
            self.model,
            self.prefill_tokens[self.offset : self.offset + tokens_to_prefill][None],
            [self.offset],
            self.kv_cache,
        )
        self.offset += tokens_to_prefill
        for i in self.kv_cache:
            mx.eval(i.key_values[0])
            mx.eval(i.key_values[1])
        if self.offset == self.prefill_tokens.size:
            mx.eval(token)
            return token, self.kv_cache, self.offset
        else:
            return None


def batch_generate(
    model: any, tokenizer: TokenizerWrapper, prompts: list[str], max_seq_len=512
):
    MAX_REQUESTS = 5
    is_idle = [True] * MAX_REQUESTS
    prompt_idx = [0] * MAX_REQUESTS
    next_tokens = mx.array([0] * MAX_REQUESTS)
    offsets = mx.array([0] * MAX_REQUESTS)
    detokenizers = [None] * MAX_REQUESTS
    kv_cache = [
        BatchingKvCache(max_active_requests=MAX_REQUESTS, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_requests = None

    print(f"Processing {len(prompts)} prompts")
    prompts = enumerate(prompts)
    more_prompts = True
    while True:
        if not more_prompts and all(is_idle):
            break
        # prefill until no idle slots
        while any(is_idle) and more_prompts and pending_prefill_requests is None:
            try:
                idx, prompt = next(prompts)
            except StopIteration:
                more_prompts = False
                break
            print(f"Starting prefill for prompt {idx}", flush=True)
            pending_prefill_requests = PrefillRequest(model, tokenizer, prompt)
            break

        if pending_prefill_requests is not None:
            res = pending_prefill_requests.prefill()
            if res is not None:
                print("Prefill done", flush=True)
                pending_prefill_requests = None
                token, prefill_kv_cache, offset = res

                if token.item() == tokenizer.eos_token_id:
                    # if the first token is eos, we skip this prompt
                    print("Request finished without producing any token", flush=True)
                    continue

                for i in range(MAX_REQUESTS):
                    if is_idle[i]:
                        detokenizers[i] = tokenizer.detokenizer.__class__(
                            tokenizer._tokenizer
                        )
                        detokenizers[i].add_token(token.item())
                        print(f"Decoding prompt {idx} at request {i}", flush=True)
                        print(
                            f"(Prefill) {idx}: " + detokenizers[i].last_segment,
                            flush=True,
                        )
                        prompt_idx[i] = idx
                        is_idle[i] = False
                        for prefill_cache, batch_cache in zip(
                            prefill_kv_cache, kv_cache
                        ):
                            batch_cache.add_request(prefill_cache, i)
                        next_tokens[i] = token
                        offsets[i] = offset
                        break
            else:
                print(
                    f"Still prefilling the request: {pending_prefill_requests.offset}/{pending_prefill_requests.prefill_tokens.size}",
                    flush=True,
                )

        if not all(is_idle):
            next_tokens = mx.array(next_tokens)
            # decode
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            offsets += 1
            for i in range(MAX_REQUESTS):
                if not is_idle[i]:
                    detokenizers[i].add_token(next_tokens[i].item())
                    if (
                        next_tokens[i].item() == tokenizer.eos_token_id
                        or offsets[i] >= max_seq_len
                    ):
                        print(
                            f"(Finished) {prompt_idx[i]}: " + detokenizers[i].text,
                            flush=True,
                        )
                        result.append((prompt_idx[i], detokenizers[i].text))
                        print(f"Removing request {i}", flush=True)
                        batch_cache.remove_request(i)
                        is_idle[i] = True
                        continue
                    else:
                        print(
                            f"(In Progress) {prompt_idx[i]}: " + detokenizers[i].text,
                            flush=True,
                        )
        else:
            print("No requests in progress", flush=True)
    return result
