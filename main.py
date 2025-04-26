from mlx_lm import load
from tiny_llm.qwen2 import Qwen2Model
from tiny_llm.generate import simple_generate
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
    simple_generate(tiny_llm_model, tokenizer, prompt)
