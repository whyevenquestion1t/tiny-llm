from mlx_lm import load
import mlx.core as mx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct-MLX")
parser.add_argument(
    "--prompt",
    action="append",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="tiny_llm")
parser.add_argument("--device", type=str, default="gpu")
args = parser.parse_args()

if args.solution == "tiny_llm":
    from tiny_llm import Qwen2Model, batch_generate

    print("Using your tiny_llm solution")
elif args.solution == "tiny_llm_week2_ref" or args.solution == "week2_ref":
    from tiny_llm_week2_ref import Qwen2Model, batch_generate

    print("Using tiny_llm_week2_ref solution")
else:
    raise ValueError(f"Solution {args.solution} not supported")

mlx_model, tokenizer = load(
    args.model,
    tokenizer_config={"eos_token": "<|im_end|>"},
    model_config={"tie_word_embeddings": False, "rope_traditional": False},
)

with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
    tiny_llm_model = Qwen2Model(mlx_model)
    all_prompts = []
    for prompt in args.prompt:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)
    batch_generate(tiny_llm_model, tokenizer, all_prompts)
