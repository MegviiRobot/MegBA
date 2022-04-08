/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/base_linear_system.h"

namespace MegBA {
template <typename T>
void BaseLinearSystem<T>::freeCUDA() {
  for (int i = 0; i < problemOption.deviceUsed.size(); ++i) {
    cudaSetDevice(i);
    cudaFree(deltaXPtr[i]);
    cudaFree(g[i]);
  }
}

template class BaseLinearSystem<double>;
template class BaseLinearSystem<float>;
}  // namespace MegBA
