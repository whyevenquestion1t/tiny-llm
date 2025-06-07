import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int):
    def sample(logits: mx.array):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            logits = logits / temp
            return mx.random.categorical(logits, axis=-1)

    return sample
