/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "algo/LM_algo.h"

namespace MegBA {
template <typename T>
LMAlgo<T>::LMAlgo(const BaseProblem<T> &problem) : BaseAlgo<T>(problem) {}
}
