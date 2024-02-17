/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "../selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<1, float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::Half, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::BFloat16, float>(SSMParamsBwd &params, cudaStream_t stream);
