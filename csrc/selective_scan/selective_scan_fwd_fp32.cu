/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_fwd_kernel_minimal.cuh"

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);

template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, complex_t>(SSMParamsBase &params, cudaStream_t stream);

template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, complex_t>(SSMParamsBase &params, cudaStream_t stream);