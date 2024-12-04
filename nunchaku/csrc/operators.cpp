#include <spdlog/spdlog.h>
#include <torch/extension.h>

#include "kernels/awq/gemv_awq.h"
#include "kernels/gemm_w4a4.h"

typedef at::BFloat16 bf16;

namespace nunchaku {

void gemv_awq(torch::Tensor &in_feats, torch::Tensor &kernel,
              torch::Tensor &scaling_factors, torch::Tensor &zeros, int64_t m,
              int64_t n, int64_t k, int64_t group_size,
              torch::Tensor &out_feats) {
  gemv_awq_ops(reinterpret_cast<half_t *>(in_feats.data_ptr<bf16>()),
               reinterpret_cast<uint32_t *>(kernel.data_ptr<int32_t>()),
               reinterpret_cast<half_t *>(scaling_factors.data_ptr<bf16>()),
               reinterpret_cast<half_t *>(zeros.data_ptr<bf16>()), m, n, k,
               group_size,
               reinterpret_cast<half_t *>(out_feats.data_ptr<bf16>()));
}

void quantize_w4a4_act_fuse_lora(torch::Tensor &input, torch::Tensor &output,
                                 torch::Tensor &oscales,
                                 torch::Tensor &lora_down,
                                 torch::Tensor &lora_act_out,
                                 c10::optional<torch::Tensor> smooth) {

  int M = input.numel() / input.sizes().back();
  int N = input.sizes().back();

  // assert(output.dtype() == Tensor::INT8);
  assert(output.numel() / output.sizes().back() == M);
  assert(output.sizes().back() == N / 2);

  // assert(oscales.dtype() == Tensor::FP16);
  // assert(isTypeMatch<GEMM::half_t>(oscales.dtype()));
  assert(oscales.numel() == M * N / 64);

  const int rank = lora_down.sizes()[1];

  assert(lora_down.sizes()[0] == N);
  // assert(lora_down.shape[1] == Lora::LORA_RANK);
  assert(lora_act_out.sizes()[0] == M);
  assert(lora_act_out.sizes()[1] == rank);

  lora_act_out.zero_();

  quantize_w4a4_act_fuse_lora_ops(
      reinterpret_cast<half_t *>(input.data_ptr<bf16>()),
      reinterpret_cast<uint8_t *>(output.data_ptr<int8_t>()),
      reinterpret_cast<half_t *>(oscales.data_ptr<bf16>()),
      reinterpret_cast<half_t *>(lora_down.data_ptr<bf16>()),
      reinterpret_cast<float *>(lora_act_out.data_ptr<float>()),
      smooth.has_value()
          ? reinterpret_cast<half_t *>(smooth.value().data_ptr<bf16>())
          : nullptr,
      M, N, rank);
}

void gemm_w4a4(
    torch::Tensor &act,                        // packed act [M, K / 2]
    torch::Tensor &wgt,                        // packed act [N, K / 2]
    c10::optional<torch::Tensor> out,          // linear     [M, N]
    c10::optional<torch::Tensor> qout,         // packed act [M, N / 2]
    torch::Tensor &ascales,                    // packed as  [K / 64, M]
    torch::Tensor &wscales,                    // packed ws  [K / 64, N]
    c10::optional<torch::Tensor> oscales,      // packed as  [N / 64, M]
    c10::optional<torch::Tensor> poolout,      // linear     [M / PoolSize, N]
    torch::Tensor &lora_act_in,                // packed lora_act [M, R]
    torch::Tensor &lora_up,                    // packed lora_wgt [N, R]
    c10::optional<torch::Tensor> lora_down,    // packed lora_wgt [N, R]
    c10::optional<torch::Tensor> lora_act_out, // packed lora_act [M, R]
    c10::optional<torch::Tensor> norm_q,       // linear     [HEAD_DIM]
    c10::optional<torch::Tensor> norm_k,       // linear     [HEAD_DIM]
    c10::optional<torch::Tensor>
        rotary_emb,      // linear     [M, HEAD_DIM / 2, 2, 2]
    torch::Tensor &bias, // packed ws  [N]
    c10::optional<torch::Tensor>
        smooth_factor, // packed ws  [N], for quantization of the next layer
    bool act_unsigned, std::vector<double> lora_scales) {
  std::vector<float> float_lora_scales(lora_scales.begin(), lora_scales.end());

  // spdlog::warn("float_lora_scales={}", float_lora_scales.at(0));
  // spdlog::warn("float_lora_scales={}", float_lora_scales.at(1));

  int M = act.numel() / act.sizes().back();
  int N = wgt.sizes()[0];
  int K = act.sizes().back() * 2;
  assert(K == wgt.sizes()[1] * 2);

  assert(bias.numel() == N);

  assert(lora_down.has_value() == lora_act_out.has_value());

  const int rank_up = lora_up.sizes()[1];

  assert(lora_up.sizes()[0] == N);
  // assert(lora_up.shape[1] == Lora::LORA_RANK);
  assert(lora_act_in.sizes()[0] == M);
  assert(lora_act_in.sizes()[1] == rank_up);

  if (lora_down.has_value()) {
    const int rank_down = lora_down.value().sizes()[1];

    assert(rank_down == rank_up);

    assert(lora_down.value().sizes()[0] == N);
    // assert(lora_down.shape[1] == Lora::LORA_RANK);
    assert(lora_act_out.value().sizes()[0] == M);
    assert(lora_act_out.value().sizes()[1] == rank_down);

    lora_act_out.value().zero_();
  }

  if (qout.has_value() && oscales.has_value()) {

  } else if (rotary_emb.has_value()) {
    assert(norm_q.has_value());
    assert(norm_k.has_value());
    // assert(isTypeMatch<GEMM::half_t>(rotary_emb.scalar_type()));
    // assert(rotary_emb.scalar_type() == Tensor::FP32);
    assert(rotary_emb.value().numel() == M * 128 / 2 * 2);
  }

  gemm_w4a4_ops(
      reinterpret_cast<uint8_t *>(act.data_ptr<int8_t>()),
      reinterpret_cast<uint8_t *>(wgt.data_ptr<int8_t>()),
      out.has_value() ? reinterpret_cast<half_t *>(out.value().data_ptr<bf16>())
                      : nullptr,
      qout.has_value()
          ? reinterpret_cast<uint8_t *>(qout.value().data_ptr<int8_t>())
          : nullptr,
      reinterpret_cast<half_t *>(ascales.data_ptr<bf16>()),
      reinterpret_cast<half_t *>(wscales.data_ptr<bf16>()),
      oscales.has_value()
          ? reinterpret_cast<half_t *>(oscales.value().data_ptr<bf16>())
          : nullptr,
      nullptr, // reinterpret_cast<half_t *>(poolout.data_ptr<bf16>()),
      reinterpret_cast<float *>(lora_act_in.data_ptr<float>()),
      reinterpret_cast<half_t *>(lora_up.data_ptr<bf16>()),
      lora_down.has_value()
          ? reinterpret_cast<half_t *>(lora_down.value().data_ptr<bf16>())
          : nullptr,
      lora_act_out.has_value()
          ? reinterpret_cast<float *>(lora_act_out.value().data_ptr<float>())
          : nullptr,
      norm_q.has_value()
          ? reinterpret_cast<half_t *>(norm_q.value().data_ptr<bf16>())
          : nullptr,
      norm_k.has_value()
          ? reinterpret_cast<half_t *>(norm_k.value().data_ptr<bf16>())
          : nullptr,
      rotary_emb.has_value()
          ? reinterpret_cast<float *>(rotary_emb.value().data_ptr<float>())
          : nullptr,
      reinterpret_cast<half_t *>(bias.data_ptr<bf16>()),
      smooth_factor.has_value()
          ? reinterpret_cast<half_t *>(smooth_factor.value().data_ptr<bf16>())
          : nullptr,
      act_unsigned, float_lora_scales, M, N, K, rank_up);
}
} // namespace nunchaku

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(nunchaku, m) {
  m.def("gemv_awq", nunchaku::gemv_awq);
  m.def("quantize_w4a4_act_fuse_lora", nunchaku::quantize_w4a4_act_fuse_lora);
  m.def("gemm_w4a4", nunchaku::gemm_w4a4);
}