/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <Python.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/complex.h>
#include <torch/headeronly/util/shim_utils.h>
#include <cuda_runtime.h>
#include <vector>

#include "selective_scan.h"

using torch::stable::Tensor;

// The current CUDA stream comes from the AOTInductor C shim rather than
// torch::stable::accelerator::Stream::nativeHandle(), which is only declared
// from torch 2.13 onwards. This is the same class of stable C entry point that
// torch/csrc/stable/accelerator.h is itself built on, and it keeps the ABI
// floor for these sources at 2.10. Same approach as vLLM's
// csrc/libtorch_stable/torch_utils.h.
static inline cudaStream_t get_cuda_stream(int32_t device_index) {
    void *stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream));
    return reinterpret_cast<cudaStream_t>(stream);
}

// torch::stable has no zeros_like. ATen implements zeros_like as
// empty_like(memory_format=preserve) followed by zero_, and both halves are
// available here, so compose them to keep the layout-preserving behaviour that
// new_zeros does not have.
static inline Tensor zeros_like_stable(
        const Tensor &self,
        std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
    Tensor result = torch::stable::empty_like(self);
    torch::stable::zero_(result);
    if (dtype.has_value() && dtype.value() != self.scalar_type()) {
        result = torch::stable::to(result, dtype.value());
    }
    return result;
}

#define CHECK_SHAPE(x, ...) STD_TORCH_CHECK(x.sizes().equals(torch::headeronly::IntHeaderOnlyArrayRef({__VA_ARGS__})), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == torch::headeronly::ScalarType::Half) {                              \
        using input_t = torch::headeronly::Half;                                     \
        __VA_ARGS__();                                                               \
    } else if (ITYPE == torch::headeronly::ScalarType::BFloat16) {                   \
        using input_t = torch::headeronly::BFloat16;                                 \
        __VA_ARGS__();                                                               \
    } else if (ITYPE == torch::headeronly::ScalarType::Float)  {                     \
        using input_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        STD_TORCH_CHECK(false, #NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == torch::headeronly::ScalarType::Half) {                               \
        using weight_t = torch::headeronly::Half;                                     \
        __VA_ARGS__();                                                                \
    } else if (WTYPE == torch::headeronly::ScalarType::BFloat16) {                    \
        using weight_t = torch::headeronly::BFloat16;                                 \
        __VA_ARGS__();                                                                \
    } else if (WTYPE == torch::headeronly::ScalarType::Float)  {                      \
        using weight_t = float;                                                       \
        __VA_ARGS__();                                                                \
    } else {                                                                          \
        STD_TORCH_CHECK(false, #NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == torch::headeronly::ScalarType::Float) {                              \
       using weight_t = float;                                                        \
        __VA_ARGS__();                                                                \
    } else if (WTYPE == torch::headeronly::ScalarType::ComplexFloat) {                \
        using weight_t = torch::headeronly::complex<float>;                           \
        __VA_ARGS__();                                                                \
    } else {                                                                          \
        STD_TORCH_CHECK(false, #NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const Tensor &u,
                        const Tensor &delta,
                        const Tensor &A,
                        const Tensor &B,
                        const Tensor &C,
                        const Tensor &out,
                        const Tensor &z,
                        const Tensor &out_z,
                        const void* D_ptr,
                        const void* delta_bias_ptr,
                        void* x_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides. out_ptr and x_ptr stay non-const: fwd writes
    // them and bwd reads them, and both share SSMParamsBase, so there is no one
    // constness correct for both. Everything else is read-only in both.
    // Set the pointers and strides.
    params.u_ptr = u.const_data_ptr();
    params.delta_ptr = delta.const_data_ptr();
    params.A_ptr = A.const_data_ptr();
    params.B_ptr = B.const_data_ptr();
    params.C_ptr = C.const_data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.mutable_data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.const_data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.mutable_data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        params.out_z_batch_stride = out_z.stride(0);
        params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

void set_ssm_params_bwd(SSMParamsBwd &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const Tensor &u,
                        const Tensor &delta,
                        const Tensor &A,
                        const Tensor &B,
                        const Tensor &C,
                        const Tensor &z,
                        const Tensor &out,
                        const Tensor &out_z,
                        const void* D_ptr,
                        const void* delta_bias_ptr,
                        void* x_ptr,
                        const Tensor &dout,
                        const Tensor &du,
                        const Tensor &ddelta,
                        const Tensor &dA,
                        const Tensor &dB,
                        const Tensor &dC,
                        const Tensor &dz,
                        void* dD_ptr,
                        void* ddelta_bias_ptr,
                        bool has_z,
                        bool delta_softplus,
                        bool recompute_out_z) {
    // Pass in "dout" instead of "out", we're not gonna use "out" unless we have z
    set_ssm_params_fwd(params, batch, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, has_z ? out : dout,
                       has_z ? z : dout,
                       // If not recompute_out_z, pass dout instead of out_z.
                       // This won't be used by the bwd kernel
                       recompute_out_z ? out_z : dout,
                       D_ptr, delta_bias_ptr, x_ptr, has_z, delta_softplus);
    if (!recompute_out_z) { params.out_z_ptr = nullptr; }

    // Set the pointers and strides.
    params.dout_ptr = dout.const_data_ptr();
    params.du_ptr = du.mutable_data_ptr();
    params.dA_ptr = dA.mutable_data_ptr();
    params.dB_ptr = dB.mutable_data_ptr();
    params.dC_ptr = dC.mutable_data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.mutable_data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;
    params.dz_ptr = has_z ? dz.mutable_data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.dA_d_stride = dA.stride(0);
    params.dA_dstate_stride = dA.stride(1);
    if (!is_variable_B) {
        params.dB_d_stride = dB.stride(0);
    } else {
        params.dB_batch_stride = dB.stride(0);
        params.dB_group_stride = dB.stride(1);
    }
    params.dB_dstate_stride = !is_variable_B ? dB.stride(1) : dB.stride(2);
    if (!is_variable_C) {
        params.dC_d_stride = dC.stride(0);
    } else {
        params.dC_batch_stride = dC.stride(0);
        params.dC_group_stride = dC.stride(1);
    }
    params.dC_dstate_stride = !is_variable_C ? dC.stride(1) : dC.stride(2);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_d_stride = ddelta.stride(1);
    if (has_z) {
        params.dz_batch_stride = dz.stride(0);
        params.dz_d_stride = dz.stride(1);
    }
}

std::vector<Tensor>
selective_scan_fwd(const Tensor &u, const Tensor &delta,
                  const Tensor &A, const Tensor &B, const Tensor &C,
                  const std::optional<Tensor> &D_,
                  const std::optional<Tensor> &z_,
                  const std::optional<Tensor> &delta_bias_,
                  bool delta_softplus) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    STD_TORCH_CHECK(input_type == torch::headeronly::ScalarType::Float || input_type == torch::headeronly::ScalarType::Half || input_type == torch::headeronly::ScalarType::BFloat16);
    STD_TORCH_CHECK(weight_type == torch::headeronly::ScalarType::Float || weight_type == torch::headeronly::ScalarType::ComplexFloat);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == torch::headeronly::ScalarType::ComplexFloat;

    STD_TORCH_CHECK(delta.scalar_type() == input_type);
    STD_TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    STD_TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    STD_TORCH_CHECK(u.is_cuda());
    STD_TORCH_CHECK(delta.is_cuda());
    STD_TORCH_CHECK(A.is_cuda());
    STD_TORCH_CHECK(B.is_cuda());
    STD_TORCH_CHECK(C.is_cuda());

    STD_TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    STD_TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    STD_TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        STD_TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        STD_TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }

    if (D_.has_value()) {
        auto D = D_.value();
        STD_TORCH_CHECK(D.scalar_type() == torch::headeronly::ScalarType::Float);
        STD_TORCH_CHECK(D.is_cuda());
        STD_TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        STD_TORCH_CHECK(delta_bias.scalar_type() == torch::headeronly::ScalarType::Float);
        STD_TORCH_CHECK(delta_bias.is_cuda());
        STD_TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        STD_TORCH_CHECK(z.scalar_type() == input_type);
        STD_TORCH_CHECK(z.is_cuda());
        STD_TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        out_z = torch::stable::empty_like(z);
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // Tensor out = torch::stable::empty_like(u);
    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    Tensor out = torch::stable::empty_like(delta);
    Tensor x = torch::stable::new_empty(u, {batch_size, dim, n_chunks, dstate * 2}, weight_type);

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_.has_value() ? D_.value().const_data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().const_data_ptr() : nullptr,
                       x.mutable_data_ptr(),
                       has_z,
                       delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    const torch::stable::accelerator::DeviceGuard device_guard(u.get_device_index());
    auto stream = get_cuda_stream(u.get_device_index());
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_fwd", [&] {
            selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<Tensor> result = {out, x};
    if (has_z) { result.push_back(out_z); }
    return result;
}

std::vector<Tensor>
selective_scan_bwd(const Tensor &u, const Tensor &delta,
                  const Tensor &A, const Tensor &B, const Tensor &C,
                  const std::optional<Tensor> &D_,
                  const std::optional<Tensor> &z_,
                  const std::optional<Tensor> &delta_bias_,
                  const Tensor &dout,
                  const std::optional<Tensor> &x_,
                  const std::optional<Tensor> &out_,
                  const std::optional<Tensor> &dz_,
                  bool delta_softplus,
                  bool recompute_out_z) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    STD_TORCH_CHECK(input_type == torch::headeronly::ScalarType::Float || input_type == torch::headeronly::ScalarType::Half || input_type == torch::headeronly::ScalarType::BFloat16);
    STD_TORCH_CHECK(weight_type == torch::headeronly::ScalarType::Float || weight_type == torch::headeronly::ScalarType::ComplexFloat);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == torch::headeronly::ScalarType::ComplexFloat;

    STD_TORCH_CHECK(delta.scalar_type() == input_type);
    STD_TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    STD_TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));
    STD_TORCH_CHECK(dout.scalar_type() == input_type);

    STD_TORCH_CHECK(u.is_cuda());
    STD_TORCH_CHECK(delta.is_cuda());
    STD_TORCH_CHECK(A.is_cuda());
    STD_TORCH_CHECK(B.is_cuda());
    STD_TORCH_CHECK(C.is_cuda());
    STD_TORCH_CHECK(dout.is_cuda());

    STD_TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    STD_TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);
    STD_TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    STD_TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        STD_TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        STD_TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    if (D_.has_value()) {
        auto D = D_.value();
        STD_TORCH_CHECK(D.scalar_type() == torch::headeronly::ScalarType::Float);
        STD_TORCH_CHECK(D.is_cuda());
        STD_TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        STD_TORCH_CHECK(delta_bias.scalar_type() == torch::headeronly::ScalarType::Float);
        STD_TORCH_CHECK(delta_bias.is_cuda());
        STD_TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    Tensor z, out, dz, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        STD_TORCH_CHECK(z.scalar_type() == input_type);
        STD_TORCH_CHECK(z.is_cuda());
        STD_TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);

        STD_TORCH_CHECK(out_.has_value());
        out = out_.value();
        STD_TORCH_CHECK(out.scalar_type() == input_type);
        STD_TORCH_CHECK(out.is_cuda());
        STD_TORCH_CHECK(out.stride(-1) == 1 || out.size(-1) == 1);
        CHECK_SHAPE(out, batch_size, dim, seqlen);

        if (dz_.has_value()) {
            dz = dz_.value();
            STD_TORCH_CHECK(dz.scalar_type() == input_type);
            STD_TORCH_CHECK(dz.is_cuda());
            STD_TORCH_CHECK(dz.stride(-1) == 1 || dz.size(-1) == 1);
            CHECK_SHAPE(dz, batch_size, dim, seqlen);
        } else {
            dz = torch::stable::empty_like(z);
        }
        if (recompute_out_z) {
            out_z = torch::stable::empty_like(out);
        }
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    if (n_chunks > 1) { STD_TORCH_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        auto x = x_.value();
        STD_TORCH_CHECK(x.scalar_type() == weight_type);
        STD_TORCH_CHECK(x.is_cuda());
        STD_TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, n_chunks, 2 * dstate);
    }

    Tensor du = torch::stable::empty_like(u);
    Tensor ddelta = torch::stable::empty_like(delta);
    Tensor dA = zeros_like_stable(A);
    Tensor dB = !is_variable_B ? zeros_like_stable(B) : zeros_like_stable(B, torch::headeronly::ScalarType::Float);
    Tensor dC = !is_variable_C ? zeros_like_stable(C) : zeros_like_stable(C, torch::headeronly::ScalarType::Float);
    Tensor dD;
    if (D_.has_value()) { dD = zeros_like_stable(D_.value()); }
    Tensor ddelta_bias;
    if (delta_bias_.has_value()) { ddelta_bias = zeros_like_stable(delta_bias_.value()); }

    SSMParamsBwd params;
    set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, z, out, out_z,
                       D_.has_value() ? D_.value().const_data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().const_data_ptr() : nullptr,
                       x_.has_value() ? x_.value().mutable_data_ptr() : nullptr,
                       dout, du, ddelta, dA, dB, dC, dz,
                       D_.has_value() ? dD.mutable_data_ptr() : nullptr,
                       delta_bias_.has_value() ? ddelta_bias.mutable_data_ptr() : nullptr,
                       has_z, delta_softplus, recompute_out_z);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    const torch::stable::accelerator::DeviceGuard device_guard(u.get_device_index());
    auto stream = get_cuda_stream(u.get_device_index());
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_bwd", [&] {
            selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<Tensor> result = {du, ddelta, dA, torch::stable::to(dB, B.scalar_type()), torch::stable::to(dC, C.scalar_type()), dD, ddelta_bias};
    if (has_z) { result.push_back(dz); }
    if (recompute_out_z) { result.push_back(out_z); }
    return result;
}

STABLE_TORCH_LIBRARY(selective_scan, m) {
    m.def("fwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, "
          "Tensor? D, Tensor? z, Tensor? delta_bias, bool delta_softplus) -> Tensor[]");
    // dz is an in/out buffer: when the caller supplies one the bwd kernel writes
    // through it (store_output in selective_scan_bwd_kernel.cuh), so it is
    // annotated as mutated. It stays optional -- selective_scan_fn passes None
    // and lets this function allocate it.
    m.def("bwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, "
          "Tensor? D, Tensor? z, Tensor? delta_bias, Tensor dout, "
          "Tensor? x, Tensor? out, Tensor(dz!)? dz, "
          "bool delta_softplus, bool recompute_out_z) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(selective_scan, CUDA, m) {
    m.impl("fwd", TORCH_BOX(&selective_scan_fwd));
    m.impl("bwd", TORCH_BOX(&selective_scan_bwd));
}

static PyMethodDef _methods[] = {{NULL, NULL, 0, NULL}};
static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "selective_scan_cuda", NULL, -1, _methods
};
extern "C" PyObject* PyInit_selective_scan_cuda(void) {
    return PyModule_Create(&_module);
}
