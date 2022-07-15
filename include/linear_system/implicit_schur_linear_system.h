/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "LM_linear_system.h"
#include "vertex/base_vertex.h"

namespace MegBA {
template <typename T>
struct ImplicitSchurLinearSystem : virtual public BaseLinearSystem<T> {
 LinearSystemKind linearSystemKind() const override { return SCHUR; }

 ComputeKind computeKind() const { return IMPLICIT; }

 ImplicitSchurLinearSystem() = delete;

 virtual ~ImplicitSchurLinearSystem();

 struct EquationContainer {
   std::array<T *, 2> csrVal{nullptr, nullptr};
   std::array<std::size_t, 2> nnz{0, 0};
 };

 std::vector<EquationContainer> implicitEquationContainers;

 void solve(const EdgeVector<T> &edges, const JVD<T> &jetEstimation) const;

protected:
 explicit ImplicitSchurLinearSystem(const ProblemOption &option,
                            std::unique_ptr<BaseSolver<T>> solver);

private:
 void freeCPU();

 void freeCUDA();
};
}  // namespace MegBA
