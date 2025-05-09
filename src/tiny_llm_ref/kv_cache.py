from typing import Optional

import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.key_values = None
        self.head_offsets = mx.array([0] * max_active_requests)
        self.head = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        B, H, L, D = key.shape
        assert key.shape == value.shape
        assert L <= self.max_seq_len
        keys, values = self.key_values
        if self.head + L <= self.max_seq_len:
            keys[:, :, self.head : self.head + L, :] = key
            values[:, :, self.head : self.head + L, :] = value
            self.head += L
            self.head_offsets += L
        else:
            fill_size = self.max_seq_len - self.head
            keys[:, :, self.head : self.max_seq_len, :] = key[:, :, :fill_size, :]
            values[:, :, self.head : self.max_seq_len, :] = value[:, :, :fill_size, :]
            remaining_size = L - fill_size
            keys[:, :, :remaining_size, :] = key[:, :, fill_size:, :]
            values[:, :, :remaining_size, :] = value[:, :, fill_size:, :]
            self.head = remaining_size
            self.head_offsets += L
        self.key_values = (keys, values)

        before_keys = keys[:, :, self.head :, :]
        before_values = values[:, :, self.head :, :]
        after_keys = keys[:, :, : self.head, :]
        after_values = values[:, :, : self.head, :]
        keys = mx.concat([after_keys, before_keys], axis=2)
        values = mx.concat([after_values, before_values], axis=2)
        return keys, values, self.head_offsets

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        keys, values = prefilled.key_values
        B, H, L, D = keys.shape
        assert B == 1
        if self.key_values is None:
            self.key_values = (
                mx.zeros((self.max_active_requests, H, self.max_seq_len, D)),
                mx.zeros((self.max_active_requests, H, self.max_seq_len, D)),
            )
        cached_keys, cached_values = self.key_values
        # Firstly, fill the cache with zeros
        cached_keys[id, :, :, :] = 0
        cached_values[id, :, :, :] = 0
        # Then, fill the cache with the prefilled values up to self.head (may wrap)
        start_pos = (self.head - L + self.max_seq_len) % self.max_seq_len
        if start_pos + L <= self.max_seq_len:
            cached_keys[id, :, start_pos : start_pos + L, :] = keys[0, :, :, :]
            cached_values[id, :, start_pos : start_pos + L, :] = values[0, :, :, :]
        else:
            cached_keys[id, :, start_pos : self.max_seq_len, :] = keys[
                0, :, : self.max_seq_len - start_pos, :
            ]
            cached_values[id, :, start_pos : self.max_seq_len, :] = values[
                0, :, : self.max_seq_len - start_pos, :
            ]
            cached_keys[id, :, : L - (self.max_seq_len - start_pos), :] = keys[
                0, :, self.max_seq_len - start_pos :, :
            ]
            cached_values[id, :, : L - (self.max_seq_len - start_pos), :] = values[
                0, :, self.max_seq_len - start_pos :, :
            ]
        self.head_offsets[id] = L
        self.key_values = (cached_keys, cached_values)

    def remove_request(self, id: int):
        if self.key_values is None:
            raise ValueError(f"Request id {id} is not in the cache")
        cached_keys, cached_values = self.key_values
        cached_keys[id, :, :, :] = 0
        cached_values[id, :, :, :] = 0


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
