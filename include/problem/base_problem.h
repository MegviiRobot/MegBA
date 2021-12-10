/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <cusparse_v2.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <set>
#include "common.h"
#include "edge/base_edge.h"
#include "vertex/base_vertex.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T> class BaseProblem {
  const ProblemOption option;

  std::size_t hessianShape{0};
  std::unordered_map<int, BaseVertex<T> *> vertices{};
  std::unordered_map<VertexKind, std::set<BaseVertex<T> *>> verticesSets{};
  struct {
    // first: working index, second: body
    std::size_t splitSize{0};
    int workingDevice{0};
    std::vector<SchurHessianEntrance<T>> schurHessianEntrance;
  } schurWS{};
  EdgeVector<T> edges{option, schurWS.schurHessianEntrance};

  std::vector<T *> schurXPtr{nullptr};
  std::vector<T *> schurDeltaXPtr{nullptr};
  std::vector<T *> schurDeltaXPtrBackup{nullptr};

  void deallocateResource();

  void deallocateResourceCUDA();

  unsigned int getHessianShape() const;

  void makeVertices();

  void setAbsolutePosition();

  bool solveLinear(double tol, double solverRefuseRatio,
                   std::size_t maxIter);

  bool solveLinearCUDA(double tol, double solverRefuseRatio,
                       std::size_t maxIter);

  void prepareUpdateData();

  void writeBack();

  void prepareUpdateDataCUDA();

  void backupLM();

  void rollbackLM();
 public:
  explicit BaseProblem(ProblemOption option = ProblemOption{});

  ~BaseProblem() = default;

  const Device &getDevice() const;

  void addVertex(int ID, BaseVertex<T> *vertex);

  void addEdge(BaseEdge<T> *edge);

  BaseVertex<T> &getVertex(int ID);

  const BaseVertex<T> &getVertex(int ID) const;

  void eraseVertex(int ID);

  void solveLM(int iter, double solverTol, double solverRefuseRatio,
               int solverMaxIter, double tau, double epsilon1,
               double epsilon2);
};
}  // namespace MegBA
