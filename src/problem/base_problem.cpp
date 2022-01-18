/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/base_problem.h"
#include <thread>
#include <iostream>
#include "algo/base_algo.h"
#include "linear_system/base_linear_system.h"
#include "linear_system/schurLM_linear_system.h"
#include "macro.h"

namespace MegBA {
namespace {
template <typename T>
void internalBuildRandomAccess(
    int i,
    std::array<int *, 2> &csrRowPtr,
    std::array<int *, 2> &csrColInd,
                               HessianEntrance<T> *schurHessianEntrance) {
  const auto &hEntranceBlockMatrix = schurHessianEntrance->nra[i];
  const auto dimOther = schurHessianEntrance->dim[1 ^ i];
  auto &csrRowPtrKind = csrRowPtr[i];
  auto &csrColIndKind = csrColInd[i];
  csrRowPtrKind = (int *)malloc((hEntranceBlockMatrix.size() * schurHessianEntrance->dim[i] + 1) * sizeof(int));
  csrColIndKind = (int *)malloc(schurHessianEntrance->counter * schurHessianEntrance->dim[i] * sizeof(int));
  csrRowPtrKind[0] = 0;
  std::size_t rowCounter{0};
  std::size_t nnzCounter{0};
  auto &HessianEntranceRaBlockMatrix = schurHessianEntrance->ra[i];
  HessianEntranceRaBlockMatrix.reserve(hEntranceBlockMatrix.size());
  HessianEntranceRaBlockMatrix.clear();
  // row
  for (const auto &rowIter : hEntranceBlockMatrix) {
    const auto &hEntranceBlockRow = rowIter.second;
    const auto rowSize = hEntranceBlockRow.size();
    typename std::decay_t<decltype(HessianEntranceRaBlockMatrix)>::value_type
        HessianEntranceRaBlockRow;
    HessianEntranceRaBlockRow.reserve(rowSize);
    // col
    for (const auto &col : hEntranceBlockRow) {
      HessianEntranceRaBlockRow.push_back(col);
      csrColIndKind[nnzCounter] = col->absolutePosition;
      nnzCounter++;
    }
    for (int j = 0; j < schurHessianEntrance->dim[i]; ++j) {
      csrRowPtrKind[rowCounter + 1] =
          csrRowPtrKind[rowCounter] + rowSize * dimOther;
      ++rowCounter;
      if (j > 0) {
        memcpy(&csrColIndKind[nnzCounter], &csrColIndKind[nnzCounter - rowSize],
               rowSize * sizeof(int));
        nnzCounter += rowSize;
      }
    }
    HessianEntranceRaBlockMatrix.push_back(std::move(HessianEntranceRaBlockRow));
  }
  schurHessianEntrance->nnzInE = csrRowPtrKind[rowCounter];
}
}  // namespace

template <typename T> void HessianEntrance<T>::buildRandomAccess(
    std::array<int *, 2> &csrRowPtr,
    std::array<int *, 2> &csrColInd) {
  // camera and point
  std::vector<std::thread> threads;
  threads.emplace_back(std::thread{internalBuildRandomAccess<T>, 0, std::ref(csrRowPtr), std::ref(csrColInd), this});
  threads.emplace_back(std::thread{internalBuildRandomAccess<T>, 1, std::ref(csrRowPtr), std::ref(csrColInd), this});
  for (auto &thread : threads)
    thread.join();
}

template <typename T>
BaseProblem<T>::BaseProblem(const ProblemOption& option)
    : option(option),
      algo(BaseAlgo<T>::dispatch(this)),
      linearSystem(BaseLinearSystem<T>::dispatch(this)) {
  if (option.N != -1 && option.nItem != -1)
    MemoryPool::resetPool(option.N, option.nItem, sizeof(T), option.deviceUsed.size());
  if (option.useSchur) {
    schurWorkSpace.splitSize = option.nItem / option.deviceUsed.size() + 1;
    schurWorkSpace.workingDevice = 0;
    schurWorkSpace.hessianEntrance.resize(option.deviceUsed.size());
    schurWorkSpace.hessianEntrance.shrink_to_fit();
  }
}

template <typename T>
void BaseProblem<T>::appendVertex(int ID, BaseVertex<T> *vertex) {
  vertices.insert(std::make_pair(ID, vertex));
}

template <typename T> void BaseProblem<T>::appendEdge(BaseEdge<T> *edge) {
  bool success = edges.tryPushBack(edge);
  if (!success) {
    edges.tryPushBack(edge);
  }
  for (int vertex_idx = edge->size() - 1; vertex_idx >= 0; --vertex_idx) {
    auto vertex = edge->operator[](vertex_idx);
    auto kind = vertex->kind();
    auto find = verticesSets.find(kind);
    if (find == verticesSets.end())
      verticesSets.emplace(vertex->kind(), std::set<BaseVertex<T> *>{vertex});
    else
      find->second.emplace(vertex);

    if (option.useSchur) {
      for (int i = 0; i < option.deviceUsed.size(); ++i) {
        auto &workingSchurHessianEntrance = schurWorkSpace.hessianEntrance[i];
        workingSchurHessianEntrance.dim[kind] = vertex->getGradShape();
        auto &connectionBlockMatrix = workingSchurHessianEntrance.nra[kind];
        auto connectionFind = connectionBlockMatrix.find(vertex);
        if (connectionFind == connectionBlockMatrix.end()) {
          connectionFind =
              connectionBlockMatrix
                  .emplace(vertex, typename HessianEntrance<T>::BlockRow{})
                  .first;
        }
        if (i == schurWorkSpace.workingDevice) {
          connectionFind->second.emplace(edge->operator[](1 ^ vertex_idx));
        }
      }
    } else {
      // TODO(Jie Ren): implement this
    }
  }
  if (option.useSchur) {
    auto &workingSchurHessianEntrance =
        schurWorkSpace.hessianEntrance[schurWorkSpace.workingDevice];
    workingSchurHessianEntrance.counter++;
    if (workingSchurHessianEntrance.counter >= schurWorkSpace.splitSize)
      schurWorkSpace.workingDevice++;
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T> BaseVertex<T> &BaseProblem<T>::getVertex(int ID) {
  auto vertex = vertices.find(ID);
  if (vertex == vertices.end())
    throw std::runtime_error("The ID " + std::to_string(ID) +
                             " does not exist in the current graph.");
  return *vertex->second;
}

template <typename T>
const BaseVertex<T> &BaseProblem<T>::getVertex(int ID) const {
  const auto vertex = vertices.find(ID);
  if (vertex == vertices.end())
    throw std::runtime_error("The ID " + std::to_string(ID) +
                             " does not exist in the current graph.");
  return *vertex->second;
}

template <typename T> void BaseProblem<T>::eraseVertex(int ID) {
  const auto vertex = vertices.find(ID);
  if (vertex == vertices.end())
    throw std::runtime_error("The ID " + std::to_string(ID) +
                             " does not exist in the current graph.");
  edges.eraseVertex(*vertex->second);
  vertices.erase(ID);

  for (auto &vertices_set : verticesSets) {
    vertices_set.second.erase(vertex->second);
  }
}

template <typename T> void BaseProblem<T>::deallocateResource() {
  edges.deallocateResource();
  switch (option.device) {
  case Device::CUDA:
    deallocateResourceCUDA();
    break;
  default:
    throw std::runtime_error("Not Implemented.");
  }
}

template <typename T> void BaseProblem<T>::allocateResource() {
  switch (option.device) {
  case Device::CUDA:
    allocateResourceCUDA();
    break;
  default:
    throw std::runtime_error("Not Implemented.");
  }
}

template <typename T> void BaseProblem<T>::buildIndex() {
  linearSystem->num[0] = verticesSets.find(CAMERA)->second.size();
  linearSystem->num[1] = verticesSets.find(POINT)->second.size();
  linearSystem->dim[0] = (*(verticesSets.find(CAMERA)->second.begin()))->getGradShape();
  linearSystem->dim[1] = (*(verticesSets.find(POINT)->second.begin()))->getGradShape();
  allocateResource();
  setAbsolutePosition();
  if (option.useSchur) {
    std::vector<std::thread> threads;
    for (int i = 0; i < schurWorkSpace.hessianEntrance.size(); ++i) {
      threads.emplace_back(
          std::thread{[&, i=i]() {
        schurWorkSpace.hessianEntrance[i].buildRandomAccess(
                dynamic_cast<SchurLMLinearSystem<T> *>(linearSystem.get())->equationContainers[i].csrRowPtr,
                dynamic_cast<SchurLMLinearSystem<T> *>(linearSystem.get())->equationContainers[i].csrColInd);
          }});
    }
    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    // TODO(Jie Ren): implement this
  }

  edges.buildPositionContainer(schurWorkSpace.hessianEntrance);
  ASSERT_CUDA_NO_ERROR();
  linearSystem->buildIndex(*this);
  ASSERT_CUDA_NO_ERROR();
  edges.allocateResource();
  ASSERT_CUDA_NO_ERROR();
  edges.fitDevice();
  ASSERT_CUDA_NO_ERROR();
}

template <typename T> void BaseProblem<T>::setAbsolutePosition() {
  const auto hessianShape = linearSystem->getHessianShape();
  T *hxPtr = new T[hessianShape];
  std::size_t entranceBias{0};
  for (auto &setPair : verticesSets) {
    auto &set = setPair.second;
    int absolutePositionCounter = 0;
    bool fixed = (*set.begin())->fixed;
    std::size_t nnzEachItem = (*set.begin())->getEstimation().rows() *
                              (*set.begin())->getEstimation().cols();
    for (auto &vertex : set) {
      vertex->absolutePosition = absolutePositionCounter;
      ++absolutePositionCounter;
      if (!fixed) {
        memcpy(&hxPtr[entranceBias], vertex->getEstimation().data(),
               nnzEachItem * sizeof(T));
        entranceBias += nnzEachItem;
      }
    }
  }
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(xPtr[i], hxPtr, hessianShape * sizeof(T),
                      cudaMemcpyHostToDevice);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
  delete[] hxPtr;
}

template <typename T> void BaseProblem<T>::writeBack() {
  T *hxPtr = new T[linearSystem->getHessianShape()];
  std::size_t entrance_bias{0};
  if (option.useSchur) {
    cudaSetDevice(0);
    cudaMemcpy(hxPtr, xPtr[0], linearSystem->getHessianShape() * sizeof(T),
               cudaMemcpyDeviceToHost);
  } else {
    // TODO(Jie Ren): implement this
  }
  for (auto &vertexSetPair : verticesSets) {
    auto &vertexSet = vertexSetPair.second;
    if ((*vertexSet.begin())->fixed)
      continue;
    const auto nnzEachItem = (*vertexSet.begin())->getEstimation().rows() *
                             (*vertexSet.begin())->getEstimation().cols();
    for (auto &vertex : vertexSet) {
      memcpy(vertex->getEstimation().data(), &hxPtr[entrance_bias],
             nnzEachItem * sizeof(T));
      entrance_bias += nnzEachItem;
    }
  }
  delete[] hxPtr;
}
template <typename T> void BaseProblem<T>::solve() {
  buildIndex();
  algo->solve(*linearSystem, edges, xPtr[0]);
}

template <typename T>
BaseProblem<T>::~BaseProblem() {

};

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
