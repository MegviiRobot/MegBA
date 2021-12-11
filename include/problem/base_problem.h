/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <cusparse_v2.h>

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "edge/base_edge.h"
#include "problem/hessian_entrance.h"
#include "solver/base_solver.h"
#include "vertex/base_vertex.h"

namespace MegBA {
template <typename T>
std::unique_ptr<BaseSolver<T>> dispatchSolver(const BaseProblem<T> &problem);

template <typename T>
class BaseProblem {
  friend std::unique_ptr<BaseSolver<T>> dispatchSolver<T>(
      const BaseProblem<T> &problem);
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

  std::vector<T *> xPtr{nullptr};
  std::vector<T *> deltaXPtr{nullptr};
  std::vector<T *> deltaXPtrBackup{nullptr};

  std::unique_ptr<BaseSolver<T>> solver;

  void deallocateResource();

  void deallocateResourceCUDA();

  unsigned int getHessianShape() const;

  void makeVertices();

  void setAbsolutePosition();

  void prepareUpdateData();

  void writeBack();

  void prepareUpdateDataCUDA();

  void backupLM();

  void rollbackLM();

 public:
  explicit BaseProblem(const ProblemOption &option = ProblemOption{});

  ~BaseProblem() = default;

  const Device &getDevice() const;

  void addVertex(int ID, BaseVertex<T> *vertex);

  void addEdge(BaseEdge<T> *edge);

  BaseVertex<T> &getVertex(int ID);

  const BaseVertex<T> &getVertex(int ID) const;

  void eraseVertex(int ID);

  void solve();

  void solveLM();
};
}  // namespace MegBA
