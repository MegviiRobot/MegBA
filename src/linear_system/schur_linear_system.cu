/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/schur_linear_system.h"

namespace MegBA {
template <typename T>
void SchurLinearSystem<T>::freeCUDA() {
  for (int i = 0; i < this->problemOption.deviceUsed.size(); ++i) {
    cudaSetDevice(i);
    const auto &container = equationContainers[i];
    for (auto p : container.csrRowPtr) {
      cudaFree(p);
    }
    for (auto p : container.csrVal) {
      cudaFree(p);
    }
    for (auto p : container.csrColInd) {
      cudaFree(p);
    }
  }
}

template class SchurLinearSystem<double>;
template class SchurLinearSystem<float>;
}  // namespace MegBA
