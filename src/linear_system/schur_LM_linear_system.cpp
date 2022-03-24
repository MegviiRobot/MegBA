/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <omp.h>

#include <thread>

#include "linear_system/schur_LM_linear_system.h"
#include "problem/base_problem.h"
#include "resource/memory_pool.h"
#include "solver/base_solver.h"

namespace MegBA {
namespace {
template <typename T>
void internalBuildSpIndex(int *csrRowPtr, int *csrColInd, std::size_t *nnz,
                          int i, const HessianEntrance<T> *hessianEntrance) {
  const auto &hEntranceBlockMatrix = hessianEntrance->ra[i];
  const auto dimOther = hessianEntrance->dim[1 ^ i];
  csrRowPtr[0] = 0;
  std::size_t rowCounter{0};
  std::size_t nnzCounter{0};
  // row
  for (const auto &row : hEntranceBlockMatrix) {
    const auto rowSize = row.size();
    // col
    omp_set_num_threads(8);
#pragma omp parallel for
    for (int col = 0; col < rowSize; ++col) {
      csrColInd[nnzCounter + col] = row[col]->absolutePosition;
    }
    nnzCounter += rowSize;
    for (int j = 0; j < hessianEntrance->dim[i]; ++j) {
      csrRowPtr[rowCounter + 1] = csrRowPtr[rowCounter] + rowSize * dimOther;
      ++rowCounter;
      if (j > 0) {
        memcpy(&csrColInd[nnzCounter], &csrColInd[nnzCounter - rowSize],
               rowSize * sizeof(int));
        nnzCounter += rowSize;
      }
    }
  }
  *nnz = csrRowPtr[rowCounter];
}
}  // namespace

template <typename T>
void SchurLMLinearSystem<T>::buildIndex(const BaseProblem<T> &problem) {
  const auto &hessianEntrance = problem.getHessianEntrance();
  const auto worldSize = MemoryPool::getWorldSize();
  std::vector<std::thread> threads;

  for (int i = 0; i < worldSize; ++i) {
    auto &containerLocal = this->equationContainers[i];
    const auto &entranceLocal = hessianEntrance[i];
    for (int j = 0; j < 2; ++j) {
      containerLocal.csrRowPtr[j] = (int *)malloc(
          (entranceLocal.nra[j].size() * this->dim[j] + 1) * sizeof(int));
      containerLocal.csrColInd[j] =
          (int *)malloc(entranceLocal.counter * this->dim[j] * sizeof(int));
      threads.emplace_back(
          std::thread{internalBuildSpIndex<T>, containerLocal.csrRowPtr[j],
                      containerLocal.csrColInd[j], &containerLocal.nnz[j], j,
                      &entranceLocal});
    }
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (int i = 0; i < worldSize; ++i) {
    this->equationContainers[i].nnz[2] =
        this->num[0] * this->dim[0] * this->dim[0];
    this->equationContainers[i].nnz[3] =
        this->num[1] * this->dim[1] * this->dim[1];
  }
  allocateResourceCUDA();
}

template <typename T>
SchurLMLinearSystem<T>::SchurLMLinearSystem(
    const ProblemOption &option, std::unique_ptr<BaseSolver<T>> solver)
    : SchurLinearSystem<T>{option, nullptr},
      LMLinearSystem<T>{option, nullptr},
      BaseLinearSystem<T>{option, std::move(solver)} {}

template struct SchurLMLinearSystem<double>;
template struct SchurLMLinearSystem<float>;
}  // namespace MegBA
