/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "../selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<2, float, complex_t>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<2, at::Half, complex_t>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<2, at::BFloat16, complex_t>(SSMParamsBwd &params, cudaStream_t stream);
