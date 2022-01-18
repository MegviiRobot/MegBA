/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <memory>

#include "base_linear_system.h"
#include "problem/base_problem.h"
#include "schurLM_linear_system.h"

namespace MegBA {
template <typename T>
std::unique_ptr<BaseLinearSystem<T>> dispatchLinearSystem(const BaseProblem<T> &problem) {
 const ProblemOption &option = problem.getProblemOption();
 if (option.useSchur) {
   switch (option.algoKind) {
     case LM:
       return std::unique_ptr<BaseLinearSystem<T>>{new SchurLMLinearSystem<T>{option}};
   }
 } else {
   throw std::runtime_error("Not implemented");
 }
}
}  // namespace MegBA
