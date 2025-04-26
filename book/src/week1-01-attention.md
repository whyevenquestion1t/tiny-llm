# Week 1 Day 1: Attention and Multi-Head Attention

In day 1, we will implement the basic attention layer and the multi-head attention layer. Attention layers take a input
sequence and focus on different parts of the sequence when generating the output. Attention layers are the key building
blocks of the Transformer models.

[📚 Reading: Transformer Architecture](https://huggingface.co/learn/llm-course/chapter1/6)

We use the Qwen2 model for text generation. The model is a decoder-only model. The input of the model is a sequence of
token embeddings. The output of the model is the most likely next token ID.

[📚 Reading: LLM Inference, the Decode Phase](https://huggingface.co/learn/llm-course/chapter1/8)

Back to the attention layer. The attention layer takes a query, a key, and a value. In a classic implementation, all
of them are of the same shape: `N.. x H x L x D`.

`N..` is zero or some number of dimensions for batches. Within each of the batch, `H` is the number of heads, `L` is the
sequence length, and `D` is the dimension of the embedding for a given head in the sequence.

So, for example, if we have a sequence of 1024 tokens, where each of the token has a 4096-dimensional embedding. We split
this 4096-dimensional embedding into 8 heads using the upper layer (i.e., multi-head attention layer), then each of the head
will have a 512-dimensional embedding. For week 1, we assume each of the batch will only have one sequence. In this case,
we will pass a tensor of the shape `1 x 8 x 1024 x 512` to the attention layer.

## Task 1: Implement `scaled_dot_product_attention`

Implement `scaled_dot_product_attention`. The function takes key, value, and query of the same dimensions.

```
K: N.. x H x L x D
V: N.. x H x L x D
Q: N.. x H x L x D
```

You may use `softmax` provided by mlx and implement it later in week 2.

**📚 Readings**

* [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch API](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (assume `enable_gqa=False`, assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [MLX API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## Task 2: Implement `MultiHeadAttention`

Implement `MultiHeadAttention`. The layer takes a batch of vectors `x`, maps it through the K, V, Q weight matrixes, and
use the attention function we implemented in day 1 to compute the result. The output needs to be mapped using the O
weight matrix. You will also need to implement the `linear` function.

For `linear`, it takes a tensor of the shape `N.. x I`, and a weight matrix of the shape `O x I`, and a bias vector of
the shape `O`. The output is of the shape `N.. x O`. `I` is the input dimension and `O` is the output dimension.

For the `MultiHeadAttention` layer, the input tensor `x` has the shape `N x L x E`, where `E` is the dimension of the
embedding for a given head in the sequence. The `K/Q/V` weight matrixes will map the tensor into key, value, and query
separately, where the dimension `E` will be mapped into a dimension of size `H x D`. Then, you will need to reshape it
to `H, D`

```
E is hidden_size or embed_dim or dims or model_dim
H is num_heads
D is head_dim
L is seq_len, in PyTorch API it's S (source len)

x: N x L x E
w_q/k/v: E x (H x D)
q/k/v = linear(x, w_q/w_k/w_v) = N x L x (H x D)
then, reshape it into N x L x H x D then transpose it to get N x H x L x D as the input of the attention function.
o = attention(q, k, v) = N x H x L x D
w_o: (H x D) x O
result = linear(reshaped o, w_o) = N x L x O
```

You can then directly split the `q/k/v` into `H` heads by reshaping the last dimension into `H x D` and apply the
attention function on it. Note that the attention function takes `N.. x H x L x D` as input, so you will need to
transpose it to get the right shape.

**📚 Readings**

* [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch API](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [MLX API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)

At the end of the day, you should be able to pass the following tests:

```
poetry run pytest tests -k test_attention_simple
poetry run pytest tests -k test_attention_with_mask
poetry run pytest tests -k test_multi_head_attention
```

{{#include copyright.md}}
