/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "base_linear_system.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T>
struct LMLinearSystem : public BaseLinearSystem<T> {
  std::vector<T *> deltaXPtrBackup;

  std::vector<T *> gBackup;

  std::vector<std::array<T *, 2>> extractedDiag;

  virtual void processDiag(
      const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const = 0;

  virtual void backup() const = 0;

  virtual void rollback() const = 0;

 protected:
  explicit LMLinearSystem(const ProblemOption &option)
      : BaseLinearSystem<T>(option),
        deltaXPtrBackup{option.deviceUsed.size()},
        gBackup{option.deviceUsed.size()},
        extractedDiag{option.deviceUsed.size()} {}
};
}