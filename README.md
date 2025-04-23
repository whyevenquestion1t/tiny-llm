# tiny-llm

Still WIP and in very early stage. A tutorial on LLM serving using MLX for system engineers. The codebase
is solely (almost!) based on MLX array/matrix APIs without any high-level neural network APIs, so that we
can build the model serving infrastructure from scratch and dig into the optimizations.

We test the implementation against PyTorch's CPU implementation to ensure correctness. The main codebase uses MLX
instead of PyTorch because nowadays it's easier to get an Apple Silicon MacBook than an NVIDIA GPU. In theory you can
implement everything using PyTorch tensor APIs, but we didn't have the test infra to support that.

<sup><sub>(TODO: maybe we should test against MLX? PyTorch APIs sometimes don't align with MLX; but I also want to ensure the computation
precision is enough to load any model directly from PyTorch tensors without converting to MLX format.)</sub></sup>

The goal is to learn the techniques behind efficiently serving an LLM model (i.e., Qwen2 models).

* Week 1: serve Qwen2 with purely Python APIs. No fancy optimizations, just Python.
* Week 2: optimizations, implement C++/Metal custom kernels to make the model run faster.
* Week 3: more optimizations, batch the requests to serve the model with high throughput.

TBD: implement a leaderboard service?

## Usage

```bash
poetry install
poetry run pytest
poetry run python main.py
```

## Week 1: LLM from Scratch

Instead of me explaining everything in this tutorial, it works in a different way: I collect all the materials, blog posts, and source
code I've read when implementing the components. There's API specification and test infra to compare the result with PyTorch/MLX in the
repo. You may read the materials and implement the components :)

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

Implement `MultiHeadAttention`. The layer takes a batch of vectors `x`, maps it through the K,V,Q weight matrixes, and
use the attention function we implemented in day 1 to compute the result. The output needs to be mapped using the O
weight matrix. You will also need to implement the `linear` function.

```
x: N x L x D
D = num_heads x head_dim
```

**References**

* Annotated Transformer https://nlp.seas.harvard.edu/annotated-transformer/
* PyTorch API https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
* MLX API https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html

### Day 2: RoPE Embedding

Note there are traditional and non-traditional ropes.

**References**

* https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html
* https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
* https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
* https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html
* https://arxiv.org/abs/2104.09864

### Day 3: Grouped Query Attention

The Qwen2 models use Grouped Query Attention (GQA). GQA allows different dimensions for query and key/value.

**References**

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py
* PyTorch API (the case where enable_gqa=True) https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
* torchtune.modules.MultiHeadAttention https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html
* https://arxiv.org/abs/2305.13245v1

### Day 4: RMSNorm and MLP

RMSNorm needs to be accumulated over float32

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py
* SiLU https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
* RMSNorm (note that it needs to accumulate at float32)

### Day 5: Transformer Block

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

### Day 6: Load the Model

We will use mlx-lm's loader to load the model. We will _steal_ the loaded parameters from the mlx model and
plug it into our own operators.

### Day 7: Generate Responses

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

Run `python main.py` and it should give you a reasonable response.

On my M4 Pro Mac Mini, my implementation gives 17 tokens per sec on Metal, versus 50 tokens per sec from the mlx-lm
Qwen2 implementation. Sadly, it also takes 4x memory than using the mlx-lm components as it does not support computation
over quantized parameters.

## Week 2

Quantization, implement softmax/linear/silu kernels, implement attention kernels, key-value cache and compression, attention masks, prompt cache.

## Week 3

Continuous batching, OpenAPI HTTP endpoint, integrate with other services.
