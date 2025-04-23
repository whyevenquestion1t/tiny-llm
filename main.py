from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from tiny_llm.layers import Qwen2Model
import mlx.core as mx

with mx.stream(mx.gpu):
    mlx_model, tokenizer = load(
        "Qwen/Qwen2-7B-Instruct-MLX",
        tokenizer_config={"eos_token": "<|im_end|>"},
        model_config={"tie_word_embeddings": False, "rope_traditional": True},
    )
    tiny_llm_model = Qwen2Model(mlx_model)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    def _step(model, y, offset):
        logits = model(y[None], offset)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)
    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    offset = tokens.size
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    # generate
    while True:
        token, _ = _step(tiny_llm_model, tokens, offset)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
