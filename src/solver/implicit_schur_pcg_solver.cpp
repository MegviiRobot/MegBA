/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "solver/implicit_schur_pcg_solver.h"

namespace MegBA {
template <typename T>
ImplicitSchurPCGSolver<T>::ImplicitSchurPCGSolver(const ProblemOption &problemOption,
                                 const SolverOption &solverOption)
   : PCGSolver<T>{problemOption, solverOption},
     SchurSolver<T>{problemOption, solverOption},
     BaseSolver<T>{problemOption, solverOption} {}

SPECIALIZE_STRUCT(ImplicitSchurPCGSolver);
}  // namespace MegBA
