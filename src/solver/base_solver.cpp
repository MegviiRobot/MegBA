/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "solver/base_solver.h"

namespace MegBA {
template <typename T>
BaseSolver<T>::BaseSolver(const ProblemOption &problemOption,
                         const SolverOption &solverOption)
   : problemOption{problemOption}, solverOption{solverOption} {}

SPECIALIZE_CLASS(BaseSolver);
}  // namespace MegBA
