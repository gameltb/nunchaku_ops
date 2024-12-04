#pragma once

#include <cstdint>
#include <cuda_bf16.h>

typedef __nv_bfloat16 half_t;

void gemv_awq_ops(half_t *in_feats, uint32_t *kernel, half_t *scaling_factors,
                  half_t *zeros, int m, int n, int k, int group_size,
                  half_t *out_feats);