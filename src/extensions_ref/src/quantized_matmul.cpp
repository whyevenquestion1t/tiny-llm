#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext_ref {

mx::array quantized_matmul(const mx::array &scales,         // Input array scales
                           const mx::array &biases,         // Input array biases
                           const int group_size,            // Group size
                           const int bits,                  // Number of bits
                           const mx::array &a,              // Input array a (not quantized)
                           const mx::array &b,              // Input array b (quantized)
                           const bool transpose_b,          // Whether to transpose b
                           mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
) {
    if (scales.dtype() != mx::float16 || biases.dtype() != mx::float16) {
        throw std::runtime_error("quantized_matmul: scales and biases must be float16");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.dtype() != mx::float16) {
        throw std::runtime_error("quantized_matmul: a must be float16");
    }
    if (a.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    auto out_shape = a.shape();
    if (out_shape.size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    out_shape[1] = b.shape()[0];
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }
    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ mx::float16,
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
        /* const std::vector<mx::array>& inputs = */ {scales, biases, a, b});
}

void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                           mx::array &out, int group_size, int bits, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    // Launch the CPU kernel
    encoder.dispatch([a_ptr = a.data<uint32_t>(), a_shape = a.shape(), a_strides = a.strides(),
                      b_ptr = b.data<float16_t>(), b_shape = b.shape(), b_strides = b.strides(),
                      out_ptr = out.data<float16_t>(), scales_ptr = scales.data<float16_t>(),
                      scales_shape = scales.shape(), scales_strides = scales.strides(),
                      biases_ptr = biases.data<float16_t>(), biases_shape = biases.shape(),
                      biases_strides = biases.strides(), group_size, bits]() {
        int M = a_shape[0];
        int N = a_shape[1];
        int K = b_shape[0];  // because we transposed b

    });
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    // TODO: dispatch to f32, f16, bf16
    quantized_matmul_impl(scales, biases, a, b, out, group_size_, bits_, stream());
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &out) {
    throw std::runtime_error("QuantizedMatmul has no GPU implementation.");
}

bool QuantizedMatmul::is_equivalent(const Primitive &other) const {
    const QuantizedMatmul &r_other = static_cast<const QuantizedMatmul &>(other);
    return group_size_ == r_other.group_size_ && bits_ == r_other.bits_;
}

}  // namespace tiny_llm_ext_ref
