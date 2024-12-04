#pragma once

#include "common.h"
#include "Tensor.h"
#include <cuda_bf16.h>

Tensor gemv_awq(
    Tensor _in_feats,
    Tensor _kernel,
    Tensor _scaling_factors,
    Tensor _zeros,
    int m,
    int n,
    int k,
    int group_size);

typedef __nv_bfloat16 half_t;

void gemv_awq_ops(
    half_t* in_feats,
    uint32_t* kernel,
    half_t* scaling_factors,
    half_t* zeros,
    int m,
    int n,
    int k,
    int group_size,
    half_t * out_feats);