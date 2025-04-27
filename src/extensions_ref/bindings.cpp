// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "axpby/axpby.h"
#include "tiny_llm_ext.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("axpby", &tiny_llm_ext_ref::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

    // m.def("quantized_linear", &tiny_llm_ext_ref::quantized_linear, "scales"_a, "biases"_a, "group_size"_a, "bits"_a,
    //       "x"_a, "w"_a, "bias"_a = nb::none(), nb::kw_only(), "stream"_a = nb::none(),
    //       R"(
    //     Quantized linear layer

    //     Follows numpy style broadcasting between ``x`` and ``w``
    //     Inputs are upcasted to floats if needed

    //     Args:
    //         scales (array): Scaling factors for ``x``.
    //         biases (array): Biases for ``x``.
    //         group_size (int): Group size for ``x``.
    //         bits (int): Number of bits for ``x``.
    //         x (array): Input array.
    //         w (array): Input array.
    //         bias (array): Input array.

    //     Returns:
    //         array: ``x * w + bias``
    //   )");
}
