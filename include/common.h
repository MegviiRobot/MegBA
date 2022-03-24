/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#define SPECIALIZE_CLASS(className) \
  template class className<double>; \
  template class className<float>

#include <cstdint>
#include <vector>

namespace MegBA {
enum Device { CPU, CUDA };

enum AlgoKind { BASE_ALGO, LM };

enum LinearSystemKind { BASE_LINEAR_SYSTEM, SCHUR };

enum SolverKind { BASE_SOLVER, PCG };

struct SolverOption {
  struct SolverOptionPCG {
    int maxIter{100};
    double tol{1e-1};
    double refuseRatio{1e0};
  } solverOptionPCG;
};

struct AlgoOption {
  struct AlgoOptionLM {
    int maxIter{20};
    double initialRegion{1e3};
    double epsilon1{1};
    double epsilon2{1e-10};
  } algoOptionLM;
};

struct ProblemOption {
  bool useSchur{true};
  Device device{Device::CUDA};
  std::vector<int> deviceUsed{};
  int N{-1};
  int64_t nItem{-1};
  AlgoKind algoKind{LM};
  LinearSystemKind linearSystemKind{SCHUR};
  SolverKind solverKind{PCG};
};

struct AlgoStatus {
  struct AlgoStatusLM {
    double region;
    bool recoverDiag{false};
  } algoStatusLM;
};

template <typename T>
class JetVector;

template <typename T>
class BaseProblem;

template <typename T>
class BaseVertex;

template <typename T>
class BaseEdge;

template <typename T>
class EdgeVector;

template <typename T>
class BaseAlgo;

template <typename T>
class BaseSolver;

template <typename T>
class BaseLinearSystem;
}  // namespace MegBA
