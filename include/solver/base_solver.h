/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "common.h"

namespace MegBA {
template <typename T>
struct BaseSolver {
  virtual void solve(const BaseLinearSystem<T>& baseLinearSystem) = 0;
};
}  // namespace MegBA
