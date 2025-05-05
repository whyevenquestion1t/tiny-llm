from typing import Optional

import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int):
        self.max_active_requests = max_active_requests
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        if self.key_values is None:
            self.key_values = (key, value)
            assert self.offset == 0
            return key, value, 0
        else:
            B, H, S, D = key.shape
            assert key.shape == value.shape
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (B, H, self.offset, D)
            assert prev_values.shape == (B, H, self.offset, D)
            new_keys = mx.concat([prev_keys, key], axis=2)
            new_values = mx.concat([prev_values, value], axis=2)
            self.key_values = (new_keys, new_values)
            start_offset = self.offset
            self.offset += S
            return new_keys, new_values, start_offset

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if self.key_values is None:
            keys, values = prefilled.keys_values
            self.key_values = (
                mx.zeros((self.max_active_requests, *keys.shape)),
                mx.zeros((self.max_active_requests, *values.shape)),
            )
        else:
            keys, values = prefilled.keys_values
            cached_keys, cached_values = self.key_values
            cached_keys[id, :, :, :, :] = keys
            cached_values[id, :, :, :, :] = values
            self.key_values = (cached_keys, cached_values)

    def remove_request(self, id: int):
        if self.key_values is None:
            raise ValueError(f"Request id {id} is not in the cache")
        cached_keys, cached_values = self.key_values
        cached_keys[id, :, :, :, :] = 0
        cached_values[id, :, :, :, :] = 0


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        if self.key_values is None:
            assert self.offset == 0
            self.key_values = (key, value)
            B, H, S, D = key.shape
            self.offset = S
            return key, value, 0
        else:
            B, H, S, D = key.shape
            assert key.shape == value.shape
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (B, H, self.offset, D)
            assert prev_values.shape == (B, H, self.offset, D)
            new_keys = mx.concat([prev_keys, key], axis=2)
            new_values = mx.concat([prev_values, value], axis=2)
            self.key_values = (new_keys, new_values)
            start_offset = self.offset
            self.offset += S
            return new_keys, new_values, start_offset


class TinyKvRotatingCache(TinyKvCache):
    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.key_values = None
        self.head = 0
        self.head_offset = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array, offset: int
    ) -> tuple[mx.array, mx.array]:
        if self.key_values is None:
            assert offset == 0
            B, H, L, D = key.shape
            assert L <= self.max_seq_len
            keys = mx.zeros((B, H, self.max_seq_len, D))
            values = mx.zeros((B, H, self.max_seq_len, D))
            keys[:, :, :L, :] = key
            values[:, :, :L, :] = value
            self.key_values = (keys, values)
            self.head = L
            self.head_offset = L
            return keys[:, :, :L, :], values[:, :, :L, :]
        else:
            B, H, L, D = key.shape
            assert key.shape == value.shape
            assert offset == self.head_offset
            assert L <= self.max_seq_len
            keys, values = self.key_values
            if self.head + L <= self.max_seq_len:
                keys[:, :, self.head : self.head + L, :] = key
                values[:, :, self.head : self.head + L, :] = value
                self.head += L
                self.head_offset += L
            else:
                fill_size = self.max_seq_len - self.head
                keys[:, :, self.head : self.max_seq_len, :] = key[:, :, :fill_size, :]
                values[:, :, self.head : self.max_seq_len, :] = value[
                    :, :, :fill_size, :
                ]
                remaining_size = L - fill_size
                keys[:, :, :remaining_size, :] = key[:, :, fill_size:, :]
                values[:, :, :remaining_size, :] = value[:, :, fill_size:, :]
                self.head = remaining_size
                self.head_offset += L
            self.key_values = (keys, values)
            if self.head_offset < self.max_seq_len:
                return keys[:, :, : self.head_offset, :], values[
                    :, :, : self.head_offset, :
                ]
            else:
                before_keys = keys[:, :, self.head_offset :, :]
                before_values = values[:, :, self.head_offset :, :]
                after_keys = keys[:, :, : self.head_offset, :]
                after_values = values[:, :, : self.head_offset, :]
                keys = mx.concat([after_keys, before_keys], axis=2)
                values = mx.concat([after_values, before_values], axis=2)
                return keys, values
