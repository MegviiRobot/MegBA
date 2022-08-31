/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/implicit_schur_linear_system.h"

namespace MegBA {
template <typename T>
void ImplicitSchurLinearSystem<T>::freeCUDA() {
  for (int i = 0; i < this->problemOption.deviceUsed.size(); ++i) {
    cudaSetDevice(i);
    const auto &container = implicitEquationContainers[i];
    for (auto p : container.csrVal) {
      cudaFree(p);
    }
  }
}

SPECIALIZE_STRUCT(ImplicitSchurLinearSystem);
}  // namespace MegBA
