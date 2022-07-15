/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system/implicit_schur_LM_linear_system.h"

#include <omp.h>

#include <thread>

#include "problem/base_problem.h"
#include "resource/memory_pool.h"
#include "solver/base_solver.h"

namespace MegBA {
template <typename T>
void ImplicitSchurLMLinearSystem<T>::buildIndex(const BaseProblem<T> &problem) {
 const auto &hessianEntrance = problem.getHessianEntrance();
 const auto worldSize = MemoryPool::getWorldSize();

 for (int i = 0; i < worldSize; ++i) {
   this->implicitEquationContainers[i].nnz[0] =
       this->num[0] * this->dim[0] * this->dim[0];
   this->implicitEquationContainers[i].nnz[1] =
       this->num[1] * this->dim[1] * this->dim[1];
 }
 allocateResourceCUDA();
}

template <typename T>
ImplicitSchurLMLinearSystem<T>::ImplicitSchurLMLinearSystem(
   const ProblemOption &option, std::unique_ptr<BaseSolver<T>> solver)
   : ImplicitSchurLinearSystem<T>{option, nullptr},
     LMLinearSystem<T>{option, nullptr},
     BaseLinearSystem<T>{option, std::move(solver)} {}

SPECIALIZE_STRUCT(ImplicitSchurLMLinearSystem);
}  // namespace MegBA
