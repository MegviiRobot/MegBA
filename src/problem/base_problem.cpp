/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/base_problem.h"
#include <thread>
#include <iostream>
#include "algo/algo_dispatcher.h"
#include "solver/solver_dispatcher.h"
#include "linear_system_manager/schurLM_linear_system_manager.h"
#include "linear_system_manager/linear_system_manager_dispatcher.h"

namespace MegBA {
namespace {
template <typename T>
void internalBuildRandomAccess(
    int i,
    std::array<int *, 2> &csrRowPtr,
    std::array<int *, 2> &csrColInd,
    SchurHessianEntrance<T> *schurHessianEntrance) {
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

template <typename T> void SchurHessianEntrance<T>::buildRandomAccess(
    std::array<int *, 2> &csrRowPtr,
    std::array<int *, 2> &csrColInd) {
  // camera and point
  std::vector<std::thread> threads;
  std::cout << "Here: " << ra[0].size() << std::endl;
  std::cout << "Here: " << ra[1].size() << std::endl;
//  getchar();
  threads.emplace_back(std::thread{internalBuildRandomAccess<T>, 0, std::ref(csrRowPtr), std::ref(csrColInd), this});
//  getchar();
  threads.emplace_back(std::thread{internalBuildRandomAccess<T>, 1, std::ref(csrRowPtr), std::ref(csrColInd), this});
  for (auto &thread : threads)
    thread.join();
}

template <typename T>
BaseProblem<T>::BaseProblem(const ProblemOption& option)
    : option(option),
      algo(dispatchAlgo(*this)),
      solver(dispatchSolver(*this)),
      linearSystemManager(dispatchLinearSystemManager(*this)) {
  if (option.N != -1 && option.nItem != -1)
    MemoryPool::resetPool(option.N, option.nItem, sizeof(T), option.deviceUsed.size());
  if (option.useSchur) {
    schurWS.splitSize = option.nItem / option.deviceUsed.size() + 1;
    schurWS.workingDevice = 0;
    schurWS.schurHessianEntrance.resize(option.deviceUsed.size());
    schurWS.schurHessianEntrance.shrink_to_fit();
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
        auto &workingSchurHessianEntrance = schurWS.schurHessianEntrance[i];
        workingSchurHessianEntrance.dim[kind] = vertex->getGradShape();
        auto &connectionBlockMatrix = workingSchurHessianEntrance.nra[kind];
        auto connectionFind = connectionBlockMatrix.find(vertex);
        if (connectionFind == connectionBlockMatrix.end()) {
          connectionFind =
              connectionBlockMatrix
                  .emplace(vertex, typename SchurHessianEntrance<T>::BlockRow{})
                  .first;
        }
        if (i == schurWS.workingDevice) {
          connectionFind->second.emplace(edge->operator[](1 ^ vertex_idx));
        }
      }
    } else {
      // TODO(Jie Ren): implement this
    }
  }
  if (option.useSchur) {
    auto &workingSchurHessianEntrance = schurWS.schurHessianEntrance[schurWS.workingDevice];
    workingSchurHessianEntrance.counter++;
    if (workingSchurHessianEntrance.counter >= schurWS.splitSize)
      schurWS.workingDevice++;
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

template <typename T> unsigned int BaseProblem<T>::getHessianShape() const {
  unsigned int gradShape = 0;
  for (const auto &vertexSetPair : verticesSets) {
    const auto &vertexSet = vertexSetPair.second;
    BaseVertex<T> const *vertexPtr = *vertexSet.begin();
    gradShape += vertexSet.size() * vertexPtr->getGradShape();
  }
  return gradShape;
}

template <typename T> void BaseProblem<T>::prepareUpdateData() {
  switch (option.device) {
  case Device::CUDA:
    prepareUpdateDataCUDA();
    break;
  default:
    throw std::runtime_error("Not Implemented.");
  }
}

template <typename T> void BaseProblem<T>::makeVertices() {
  hessianShape = getHessianShape();
  prepareUpdateData();
  setAbsolutePosition();
  std::cout << "Here: " << schurWS.schurHessianEntrance[0].ra[0].size() << std::endl;
  std::cout << "Here: " << schurWS.schurHessianEntrance[0].ra[1].size() << std::endl;
  if (option.useSchur) {
    std::vector<std::thread> threads;
    for (int i = 0; i < schurWS.schurHessianEntrance.size(); ++i) {
      auto &entrance = schurWS.schurHessianEntrance[i];
      auto &csrRowPtr = dynamic_cast<SchurLMLinearSystemManager<T> *>(linearSystemManager.get())->equationContainers[i].csrRowPtr;
      auto &csrColInd = dynamic_cast<SchurLMLinearSystemManager<T> *>(linearSystemManager.get())->equationContainers[i].csrColInd;
      threads.emplace_back(
          std::thread{[&, entrance_ptr = &entrance]() {
            (*entrance_ptr).buildRandomAccess(csrRowPtr, csrColInd);
          }});
    }
    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    // TODO(Jie Ren): implement this
  }

  linearSystemManager->buildIndex(*this);
  edges.verticesSetPtr = &verticesSets;
  edges.allocateResourcePre();
//  edges.makeVertices();
//  edges.allocateResourcePost();
  edges.fitDevice();
}

template <typename T> void BaseProblem<T>::setAbsolutePosition() {
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
  T *hxPtr = new T[hessianShape];
  std::size_t entrance_bias{0};
  if (option.useSchur) {
    cudaSetDevice(0);
    cudaMemcpy(hxPtr, xPtr[0], hessianShape * sizeof(T),
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
  makeVertices();
  algo->solve(*linearSystemManager, edges, xPtr[0]);
//  switch (option.algoKind) {
//  case LM:
//    solveLM();
//    break;
//  }
}
template <typename T>
BaseProblem<T>::~BaseProblem() = default;

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
