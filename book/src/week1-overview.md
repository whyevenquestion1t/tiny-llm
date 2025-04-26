# Week 1: From Matmul to Text

In this week, we will start from the basic matrix operations and see how those these matrix manipulations can turn the
Qwen2 model parameters into a model that generates text. We will implement the neural network layers used in the Qwen2
model using mlx's matrix APIs.

## What We will Cover

* Attention, Multi-Head Attention, and Grouped/Multi Query Attention
* Positional Embeddings and RoPE
* Put the attention layers together and implement the whole Transformer block
* Implement the MLP layer and the whole Transformer model
* Load the Qwen2 model parameters and generate text

## What We will Not Cover
To make the journey as interesting as possible, we will skip a few things for now:

* How to quantize/dequantize a model -- that will be part of week 2. The Qwen2 model is quantized so we will need to
  dequantize them before we can use them in our layer implementations.
* Actually we still used some APIs other than matrix manipulations -- like softmax, exp, log, etc. But they are simple
  and not implementing them would not affect the learning experience.
* Tokenizer -- we will not implement the tokenizer from scratch. We will use the `mlx_lm` tokenizer to tokenize the input.
* Loading the model weights -- I don't think it's an interesting thing to learn how to decode those tensor dump files, so
  we will use the `mlx_lm` to load the model and steal the weights from the loaded model into our layer implementations.

{{#include copyright.md}}
