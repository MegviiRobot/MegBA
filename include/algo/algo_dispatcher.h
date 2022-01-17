/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <memory>

#include "base_algo.h"
#include "problem/base_problem.h"
#include "lm_algo.h"

namespace MegBA {
template <typename T>
std::unique_ptr<BaseAlgo<T>> dispatchAlgo(const BaseProblem<T> &problem) {
 const ProblemOption &option = problem.getProblemOption();
 if (option.useSchur) {
   switch (option.algoKind) {
     case LM:
       return std::unique_ptr<BaseAlgo<T>>{new LMAlgo<T>{option.algoOption}};
   }
 } else {
   throw std::runtime_error("Not implemented");
 }
}
}  // namespace MegBA
