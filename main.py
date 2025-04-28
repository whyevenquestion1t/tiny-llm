from mlx_lm import load
import mlx.core as mx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct-MLX")
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="tiny_llm")
parser.add_argument("--device", type=str, default="gpu")
args = parser.parse_args()

use_mlx = False
if args.solution == "tiny_llm":
    from tiny_llm import Qwen2Model, simple_generate
    print("Using your tiny_llm solution")
elif args.solution == "tiny_llm_week1_ref" or args.solution == "week1_ref":
    from tiny_llm_week1_ref import Qwen2Model, simple_generate
    print("Using tiny_llm_week1_ref solution")
elif args.solution == "tiny_llm_week2_ref" or args.solution == "week2_ref":
    from tiny_llm_week2_ref import Qwen2Model, simple_generate
    print("Using tiny_llm_week2_ref solution")
elif args.solution == "mlx":
    use_mlx = True
    from mlx_lm.generate import stream_generate
    print("Using the original mlx model")
else:
    raise ValueError(f"Solution {args.solution} not supported")

mlx_model, tokenizer = load(
    args.model,
    tokenizer_config={"eos_token": "<|im_end|>"},
    model_config={"tie_word_embeddings": False, "rope_traditional": False},
)

with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
    if use_mlx:
        tiny_llm_model = mlx_model
    else:
        tiny_llm_model = Qwen2Model(mlx_model)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not use_mlx:
        simple_generate(tiny_llm_model, tokenizer, prompt)
    else:
        for resp in stream_generate(tiny_llm_model, tokenizer, prompt):
            print(resp.text, end="", flush=True)
