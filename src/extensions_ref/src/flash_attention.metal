#include <metal_stdlib>
using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    [[maybe_unused]] device const int64_t &N [[buffer(4)]],
    device const int64_t &S [[buffer(5)]],
    device const int64_t &L [[buffer(6)]],
    device const int64_t &E [[buffer(7)]],
    device const int64_t &num_kv_heads [[buffer(8)]],
    device const int64_t &num_heads [[buffer(9)]],
    device const float &scale [[buffer(10)]],
    device const int64_t &Br [[buffer(11)]],
    device const int64_t &Bc [[buffer(12)]],
    [[maybe_unused]] device const int64_t &Tr [[buffer(13)]],
    device const int64_t &Tc [[buffer(14)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    int n = group_id.x;
    int i = group_id.y; // loop over Tr
    int a = simd_gid; // max=Br
    int b = simd_lid; // max=Bc

    // We do not use the shared memory for the threadgroup in this course --
    // this is left as an exercise for the students. For example, you can allocate
    // 128*32*sizeof(float) bytes * number of arrays and use them as the threadgroup
    // shared memory.

    bool is_i_in_range = i * Br + a < S && a < Br;

    const int q_kv_ratio = num_heads / num_kv_heads;
    device const float *q_ptr = q + n * S * E + i * Br * E;
    device const float *k_ptr_base = k + (n / q_kv_ratio) * L * E;
    device const float *v_ptr_base = v + (n / q_kv_ratio) * L * E;
    threadgroup float o_i[32][128]; // Br x E, each simd group shares an o_i, only lane 0 writes to it

    if (simd_lid == 0) {
        for (int c = 0; c < E; c++) {
            o_i[a][c] = 0.0;
        }
    }

    // q_ptr: S * E
    // k_ptr: L * E
    // v_ptr: L * E
    // To access q[a, c]: use a * E + c
    // To access k/v[b, c]: use b * E + c

    float m_i = -1e9; // per thread; sync to threadgroup memory later
    float l_i = 0.0; // per thread; sync to threadgroup memory later

    for (int j = 0; j < Tc; j++) {
        bool is_j_in_range = j * Bc + b < L && b < Bc;

        device const float *k_ptr = k_ptr_base + j * Bc * E;
        device const float *v_ptr = v_ptr_base + j * Bc * E;

        // compute s_i = q_i @ k_j^T; store the result of each cell in thread local memory
        float s_a_b = 0.0;
        for (int c = 0; c < E; c++) {
            if (is_i_in_range && is_j_in_range) {
                s_a_b += q_ptr[a * E + c] * k_ptr[b * E + c];
            }
        }
        // for each cell, get the rowmax of the corresponding row, and compute m_i in each
        // of the cells
        float rowmax = simd_max(s_a_b);
        float new_max = max(m_i, rowmax);
        float m_i_diff = m_i - new_max;
        float m_i_diff_exp = exp(m_i_diff);
        m_i = new_max;

        // compute matrix p_j for each of the cell
        float p_a_b;
        if (is_i_in_range && is_j_in_range) {
            p_a_b = exp(s_a_b - m_i);
        } else {
            p_a_b = 0.0;
        }

        // compute l
        // get the rowsum of each row of p_j in all of the cells
        float rowsum = simd_sum(p_a_b);
        l_i = m_i_diff_exp * l_i + rowsum;

        // compute o_i, where O is Br x E; note that this does not align
        // with the threadgroup we dispatch, so we have to do threadgroup sync
        for (int c = 0; c < E; c++) {
            float v;
            if (is_i_in_range && is_j_in_range) {
                v = p_a_b * v_ptr[b * E + c];
            } else {
                v = 0.0;
            }
            float res = simd_sum(v); // res = sum(p_a_b * v_j) on each cell
            // only lane 0 will write to threadgroup memory
            if (simd_lid == 0) {
                o_i[a][c] = m_i_diff_exp * o_i[a][c] + res;
            }
        }
    }

    // write to output
    if (simd_lid == 0) {
        for (int c = 0; c < E; c++) {
            o_i[a][c] /= l_i;
        }
        for (int c = 0; c < E; c++) {
            if (is_i_in_range) {
                out[n * S * E + (i * Br + a) * E + c] = o_i[a][c];
            }
        }
    }
}
