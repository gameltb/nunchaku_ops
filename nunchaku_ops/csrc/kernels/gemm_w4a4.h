#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <vector>

typedef __nv_bfloat16 half_t;

void quantize_w4a4_act_fuse_lora_ops(half_t *input, uint8_t *output,
                                     half_t *oscales, half_t *lora_down,
                                     float *lora_act_out, half_t *smooth, int M,
                                     int N, int rank);
void gemm_w4a4_ops(
    uint8_t *act,          // packed act [M, K / 2]
    uint8_t *wgt,          // packed act [N, K / 2]
    half_t *out,           // linear     [M, N]
    uint8_t *qout,         // packed act [M, N / 2]
    half_t *ascales,       // packed as  [K / 64, M]
    half_t *wscales,       // packed ws  [K / 64, N]
    half_t *oscales,       // packed as  [N / 64, M]
    void *poolout,         // linear     [M / PoolSize, N]
    float *lora_act_in,    // packed lora_act [M, R]
    half_t *lora_up,       // packed lora_wgt [N, R]
    half_t *lora_down,     // packed lora_wgt [N, R]
    float *lora_act_out,   // packed lora_act [M, R]
    half_t *norm_q,        // linear     [HEAD_DIM]
    half_t *norm_k,        // linear     [HEAD_DIM]
    float *rotary_emb,     // linear     [M, HEAD_DIM / 2, 2, 2]
    half_t *bias,          // packed ws  [N]
    half_t *smooth_factor, // packed ws  [N], for quantization of the next layer
    bool act_unsigned,
    std::vector<float> lora_scales, // [R / 16]
    int M, int N, int K, int lora_rank);