[[kernel]] void flash_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* scale [[buffer(3)]],
    device const int &num_kv_heads [[buffer(4)]],
    device const int &num_heads [[buffer(5)]],
    device const int &head_dim [[buffer(6)]],
    device const int &kv_seq_len [[buffer(7)]],
    device const int &seq_len [[buffer(8)]],
    device const int &batch_size [[buffer(9)]],
    device float* out [[buffer(10)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
}
