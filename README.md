# tiny-llm

LLM serving using MLX. The codebase only uses MLX array/matrix APIs and all other stuff is built from scratch.

We test the implementation against PyTorch's CPU implementation to ensure correctness. The main codebase uses MLX
instead of PyTorch because nowadays it's easier to get an Apple Silicon MacBook than an NVIDIA GPU.

## Usage

```bash
poetry install
poetry run pytest
```

## Week 1: LLM from Scratch

### Day 1: Attention is All You Need

Implement `scaled_dot_product_attention`. The function takes key, value, and query of the same dimensions.

```
K: N.. x H x L x E
V: N.. x H x L x E
Q: N.. x H x L x E
```

Where `N..` is zero or some number of dimensions for batches. Within each of the batch, `H` is the number of heads,
`L` is the sequence length, and `E` is the embedding/hidden size.

You may use `softmax` provided by mlx and implement it later in week 2.

**References**

* Annotated Transformer https://nlp.seas.harvard.edu/annotated-transformer/
* PyTorch API (the case where enable_gqa=False) https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
* MLX API https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
* https://arxiv.org/abs/1706.03762

### Day 2: Multi-Head Attention

Implement `MultiHeadAttention`. The layer takes a batch of vectors `x`, maps it through the K,V,Q weight matrixes, and
use the attention function we implemented in day 1 to compute the result. The output needs to be mapped using the O
weight matrix. You will also need to implement the `linear` function.

```
x: N x L x D
D = num_heads x head_dim
```

* Annotated Transformer https://nlp.seas.harvard.edu/annotated-transformer/
* PyTorch API https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
* MLX API https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html

### Day 3: Grouped Query Attention

The Qwen2 models use Grouped Query Attention (GQA). GQA allows different dimensions for query and key/value.

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py
* PyTorch API (the case where enable_gqa=True) https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
* torchtune.modules.MultiHeadAttention https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html
* https://arxiv.org/abs/2305.13245v1

### Day 4: RoPE Embedding

Positional embedding

* https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html
* https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
* https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html
* https://arxiv.org/abs/2104.09864
