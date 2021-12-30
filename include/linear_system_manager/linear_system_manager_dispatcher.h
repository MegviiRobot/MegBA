/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <memory>

#include "base_linear_system_manager.h"
#include "problem/base_problem.h"
#include "schurLM_linear_system_manager.h"

namespace MegBA {
template <typename T>
std::unique_ptr<BaseLinearSystemManager<T>> dispatchLinearSystemManager(const BaseProblem<T> &problem) {
 const ProblemOption &option = problem.getProblemOption();
 if (option.useSchur) {
   switch (option.algoKind) {
     case LM:
       return std::unique_ptr<BaseLinearSystemManager<T>>{new SchurLMLinearSystemManager<T>{}};
   }
 } else {
   throw std::runtime_error("Not implemented");
 }
}
}  // namespace MegBA
