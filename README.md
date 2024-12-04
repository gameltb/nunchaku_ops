# [Nunchaku](https://github.com/mit-han-lab/nunchaku) ops for pytorch

Nunchaku is an inference engine designed for 4-bit diffusion models, as demonstrated in our paper [SVDQuant](http://arxiv.org/abs/2411.05007). Please check [DeepCompressor](https://github.com/mit-han-lab/deepcompressor) for the quantization library.

This repo allows the use of `gemv_awq`, `quantize_w4a4_act_fuse_lora` and `gemm_w4a4` Nunchaku cuda kernels in pytorch.