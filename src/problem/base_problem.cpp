/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/base_problem.h"
#include <thread>
#include <condition_variable>
#include "solver/solver_dispatcher.h"

namespace MegBA {
namespace {
template <typename SchurHessianEntrance>
void internalBuildRandomAccess(int i, SchurHessianEntrance *schurHessianEntrance) {
  const auto &hEntranceBlockMatrix = schurHessianEntrance->nra[i];
  const auto dimOther = schurHessianEntrance->dim[1 ^ i];
  auto &csrRowPtrKind = schurHessianEntrance->csrRowPtr[i];
  auto &csrColIndKind = schurHessianEntrance->csrColInd[i];
  csrRowPtrKind.reset(
      new int[hEntranceBlockMatrix.size() * schurHessianEntrance->dim[i] + 1]);
  csrColIndKind.reset(
      new int[schurHessianEntrance->counter * schurHessianEntrance->dim[i]]);
  csrRowPtrKind[0] = 0;
  std::size_t rowCounter{0};
  std::size_t nnzCounter{0};
  auto &HessianEntranceRaBlockMatrix = schurHessianEntrance->ra[i];
  HessianEntranceRaBlockMatrix.clear();
  HessianEntranceRaBlockMatrix.reserve(hEntranceBlockMatrix.size());
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

template <typename T> void SchurHessianEntrance<T>::buildRandomAccess() {
  // camera and point
  std::vector<std::thread> threads;
  threads.emplace_back(std::thread{internalBuildRandomAccess<SchurHessianEntrance<T>>,
                                   0, this});
  threads.emplace_back(std::thread{internalBuildRandomAccess<SchurHessianEntrance<T>>,
                                   1, this});
  for (auto &thread : threads)
    thread.join();
}

template <typename T>
BaseProblem<T>::BaseProblem(const ProblemOption& option) : option(option), solver(dispatchSolver(*this)) {
  if (option.N != -1 && option.nElm != -1)
    MemoryPool::resetPool(option.N, option.nElm, sizeof(T), option.deviceUsed.size());
  if (option.useSchur) {
    schurWS.splitSize = option.nElm / option.deviceUsed.size() + 1;
    schurWS.workingDevice = 0;
    schurWS.schurHessianEntrance.resize(option.deviceUsed.size());
    schurWS.schurHessianEntrance.shrink_to_fit();
  }
}

template <typename T> const Device &BaseProblem<T>::getDevice() const {
  return option.device;
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
  if (option.useSchur) {
    std::vector<std::thread> threads;
    for (auto &schurHessianEntrance : schurWS.schurHessianEntrance) {
      threads.emplace_back(
          std::thread{[&]() { schurHessianEntrance.buildRandomAccess(); }});
    }
    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    // TODO(Jie Ren): implement this
  }

  edges.verticesSetPtr = &verticesSets;
  edges.allocateResourcePre();
  edges.makeVertices();
  edges.allocateResourcePost();
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
  switch (option.algoKind) {
  case LM:
    solveLM();
    break;
  }
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
