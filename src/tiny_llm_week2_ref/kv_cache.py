from typing import Optional

import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(self, key: mx.array, value: mx.array, offset: int) -> mx.array:
        pass
