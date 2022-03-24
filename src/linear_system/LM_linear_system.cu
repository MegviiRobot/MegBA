/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/LM_linear_system.h"

namespace MegBA {
template <typename T>
void LMLinearSystem<T>::freeCUDA() {
  for (int i = 0; i < this->problemOption.deviceUsed.size(); ++i) {
    cudaSetDevice(i);
    cudaFree(deltaXPtrBackup[i]);
    cudaFree(gBackup[i]);
    for (auto p : extractedDiag[i]) {
      cudaFree(p);
    }
  }
}

SPECIALIZE_CLASS(LMLinearSystem);
}  // namespace MegBA
