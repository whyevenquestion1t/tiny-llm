[[kernel]] void quantized_matmul_w4a16_g64(
    device const half* scales [[buffer(0)]],
    device const half* biases [[buffer(1)]],
    device const half* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device half* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
    const int group_size = 64;
    const int bits = 4;
    const int packs_per_item = 32 / bits;
    const int item_mask = (1 << bits) - 1;
    const int groups_per_row = N / group_size;
    // Each threadgroup processes an element in the output matrix
    const int64_t idx = group_id * threads_per_threadgroup + thread_id;
    const int64_t i = idx / K;
    const int64_t k = idx % K;
    float sum = 0;
    for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
        const int64_t scales_biases_loc = k * groups_per_row + group_idx;
        const float scale = scales[scales_biases_loc];
        const float bias = biases[scales_biases_loc];
        int64_t b_loc = (k * N + group_idx * group_size) / 8;
        int64_t a_loc = i * N + group_idx * group_size;
        for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
            const uint32_t b_val = b[b_loc];
            thread const uint32_t *b_val_ref = &b_val;
            thread const uint8_t *b_bytes = reinterpret_cast<thread const uint8_t *>(b_val_ref);
            for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                const uint8_t item_val = (b_bytes[pack_idx / 2] >> ((pack_idx % 2) * bits)) & item_mask;
                const float b_val = static_cast<float>(item_val) * scale + bias;
                const float a_val = a[a_loc];
                sum += a_val * b_val;
                a_loc += 1;
            }
            b_loc += 1;
        }
    }
    if (i < M && k < K) {
        out[i * K + k] = sum;
    }
}
