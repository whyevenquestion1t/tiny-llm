from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load(
    "Qwen/Qwen2-7B-Instruct-MLX", tokenizer_config={"eos_token": "<|im_end|>"}
)

print(model)

print(tokenizer._tokenizer)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

sampler = make_sampler(top_p=0.8, temp=0.7)
logits_processors = make_logits_processors(repetition_penalty=1.05)
response = generate(
    model,
    tokenizer,
    prompt=text,
    verbose=True,
    sampler=sampler,
    logits_processors=logits_processors,
    max_tokens=512,
)
