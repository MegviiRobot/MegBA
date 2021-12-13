/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "problem/base_problem.h"

namespace MegBA {
template <typename T>
class BaseAlgo {
    BaseProblem<T> *problem;
   public:
    explicit BaseAlgo(BaseProblem<T> *problem) : problem(problem) {};

  };
}
