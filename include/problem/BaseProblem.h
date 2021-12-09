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
#include "Common.h"
#include "edge/BaseEdge.h"
#include "vertex/BaseVertex.h"
#include "problem/HEntrance.h"

namespace MegBA {
template <typename T> class BaseProblem {
  const ProblemOption option;

  std::size_t hessianShape{0};
  std::unordered_map<int, BaseVertex<T> *> vertices{};
  std::unordered_map<VertexKind, std::set<BaseVertex<T> *>> verticesSets{};
  struct SchurWorkingSpace_t {
    // first: working index, second: body
    std::size_t splitSize{0};
    int workingDevice{0};
    std::vector<SchurHEntrance<T>> schurHEntrance;
  } schurWS{};
  EdgeVector<T> edges{option, schurWS.schurHEntrance};

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

  const device_t &getDevice() const;

  void appendVertex(int ID, BaseVertex<T> *vertex);

  void appendEdge(BaseEdge<T> *edge);

  BaseVertex<T> &getVertex(int ID);

  const BaseVertex<T> &getVertex(int ID) const;

  void eraseVertex(int ID);

  void solveLM(int iter, double solverTol, double solverRefuseRatio,
               int solverMaxIter, double tau, double epsilon1,
               double epsilon2);
};
}  // namespace MegBA
