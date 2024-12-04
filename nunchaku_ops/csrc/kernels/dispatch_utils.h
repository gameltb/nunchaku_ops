#pragma once

#include <cuda_fp16.h>

template <typename F, int... N>
inline auto dispatchVal(int val, std::integer_sequence<int, N...>, F &&func) {
  auto call = [&]<int i>() {
    if (val == i) {
      func.template operator()<i>();
    }
  };
  (call.template operator()<N>(), ...);
}

template <typename F>
inline auto dispatchBool(bool val, F &&func) {
  if (val) {
    func.template operator()<true>();
  } else {
    func.template operator()<false>();
  }
}
