/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <iostream>
#include <thread>

#include "algo/base_algo.h"
#include "linear_system/base_linear_system.h"
#include "macro.h"
#include "problem/base_problem.h"

namespace MegBA {
namespace {
template <typename T>
void internalBuildRandomAccess(int i, HessianEntrance<T> *hessianEntrance) {
  const auto &hEntranceBlockMatrix = hessianEntrance->nra[i];
  auto &HessianEntranceRaBlockMatrix = hessianEntrance->ra[i];
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
    for (const auto col : hEntranceBlockRow) {
      HessianEntranceRaBlockRow.push_back(col);
    }
    HessianEntranceRaBlockMatrix.push_back(
        std::move(HessianEntranceRaBlockRow));
  }
}
}  // namespace

template <typename T>
void HessianEntrance<T>::buildRandomAccess() {
  // camera and point
  std::vector<std::thread> threads;
  threads.emplace_back(std::thread{internalBuildRandomAccess<T>, 0, this});
  threads.emplace_back(std::thread{internalBuildRandomAccess<T>, 1, this});
  for (auto &thread : threads) thread.join();
}

template <typename T>
BaseProblem<T>::BaseProblem(const ProblemOption &problemOption,
                            std::unique_ptr<BaseAlgo<T>> algo,
                            std::unique_ptr<BaseLinearSystem<T>> linearSystem)
    : problemOption(problemOption),
      algo(std::move(algo)),
      linearSystem(std::move(linearSystem)) {
  if (problemOption.N != -1 && problemOption.nItem != -1)
    MemoryPool::resetPool(&problemOption, sizeof(T));
  if (problemOption.useSchur) {
    schurWorkSpace.splitSize =
        problemOption.nItem / problemOption.deviceUsed.size() + 1;
    schurWorkSpace.workingDevice = 0;
    schurWorkSpace.hessianEntrance.resize(problemOption.deviceUsed.size());
    schurWorkSpace.hessianEntrance.shrink_to_fit();
  }
  if (this->algo->algoKind() != problemOption.algoKind) {
    throw std::runtime_error("Wrong algo type");
  }
  if (this->linearSystem->algoKind() != problemOption.algoKind &&
      this->linearSystem->linearSystemKind() !=
          problemOption.linearSystemKind) {
    throw std::runtime_error("Wrong linear system type");
  }
}

template <typename T>
void BaseProblem<T>::appendVertex(int ID, BaseVertex<T> *vertex) {
  vertices.insert(std::make_pair(ID, vertex));
}

template <typename T>
void BaseProblem<T>::appendEdge(BaseEdge<T> &edge) {
  bool success = edges.tryPushBack(edge);
  if (!success) {
    edges.tryPushBack(edge);
  }
  for (int vertex_idx = edge.size() - 1; vertex_idx >= 0; --vertex_idx) {
    auto vertex = edge[vertex_idx];
    auto kind = vertex->kind();
    auto find = verticesSets.find(kind);
    if (find == verticesSets.end())
      verticesSets.emplace(vertex->kind(), std::set<BaseVertex<T> *>{vertex});
    else
      find->second.emplace(vertex);

    if (problemOption.useSchur) {
      for (int i = 0; i < problemOption.deviceUsed.size(); ++i) {
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
          connectionFind->second.emplace(edge[1 ^ vertex_idx]);
        }
      }
    } else {
      // TODO(Jie Ren): implement this
    }
  }
  if (problemOption.useSchur) {
    auto &workingSchurHessianEntrance =
        schurWorkSpace.hessianEntrance[schurWorkSpace.workingDevice];
    workingSchurHessianEntrance.counter++;
    if (workingSchurHessianEntrance.counter >= schurWorkSpace.splitSize)
      schurWorkSpace.workingDevice++;
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T>
BaseVertex<T> &BaseProblem<T>::getVertex(int ID) {
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

template <typename T>
void BaseProblem<T>::eraseVertex(int ID) {
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

template <typename T>
void BaseProblem<T>::deallocateResource() {
  edges.deallocateResource();
  switch (problemOption.device) {
    case Device::CUDA:
      deallocateResourceCUDA();
      break;
    default:
      throw std::runtime_error("Not Implemented.");
  }
}

template <typename T>
void BaseProblem<T>::allocateResource() {
  switch (problemOption.device) {
    case Device::CUDA:
      allocateResourceCUDA();
      break;
    default:
      throw std::runtime_error("Not Implemented.");
  }
}

template <typename T>
void BaseProblem<T>::buildIndex() {
  linearSystem->num[0] = verticesSets.find(CAMERA)->second.size();
  linearSystem->num[1] = verticesSets.find(POINT)->second.size();
  linearSystem->dim[0] =
      (*(verticesSets.find(CAMERA)->second.begin()))->getGradShape();
  linearSystem->dim[1] =
      (*(verticesSets.find(POINT)->second.begin()))->getGradShape();
  allocateResource();
  setAbsolutePosition();
  if (problemOption.useSchur) {
    std::vector<std::thread> threads;
    for (int i = 0; i < schurWorkSpace.hessianEntrance.size(); ++i) {
      threads.emplace_back(std::thread{[&, i = i]() {
        schurWorkSpace.hessianEntrance[i].buildRandomAccess();
      }});
    }
    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    // TODO(Jie Ren): implement this
  }

  linearSystem->buildIndex(*this);
  ASSERT_CUDA_NO_ERROR();
  edges.buildPositionContainer(schurWorkSpace.hessianEntrance);
  ASSERT_CUDA_NO_ERROR();
  edges.allocateResource();
  ASSERT_CUDA_NO_ERROR();
  edges.fitDevice();
  ASSERT_CUDA_NO_ERROR();
}

template <typename T>
void BaseProblem<T>::setAbsolutePosition() {
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
  if (problemOption.useSchur) {
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

template <typename T>
void BaseProblem<T>::writeBack() {
  T *hxPtr = new T[linearSystem->getHessianShape()];
  std::size_t entrance_bias{0};
  if (problemOption.useSchur) {
    cudaSetDevice(0);
    cudaMemcpy(hxPtr, xPtr[0], linearSystem->getHessianShape() * sizeof(T),
               cudaMemcpyDeviceToHost);
  } else {
    // TODO(Jie Ren): implement this
  }
  for (auto &vertexSetPair : verticesSets) {
    auto &vertexSet = vertexSetPair.second;
    if ((*vertexSet.begin())->fixed) continue;
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
template <typename T>
void BaseProblem<T>::solve() {
  buildIndex();
  algo->solve(*linearSystem, edges, xPtr[0]);
  writeBack();
}

template <typename T>
BaseProblem<T>::~BaseProblem() {
  auto problemOptionBackup = new ProblemOption{problemOption};
  deallocateResource();
  MemoryPool::destruct();
  HandleManager::destroyNCCLComm();
  HandleManager::destroyCUBLASHandle();
  HandleManager::destroyCUSPARSEHandle();
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
