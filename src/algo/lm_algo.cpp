/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "algo/lm_algo.h"

namespace MegBA {
template <typename T>
LMAlgo<T>::LMAlgo(const ProblemOption &problemOption,
                  const AlgoOption &algoOption)
    : BaseAlgo<T>{problemOption, algoOption} {
  this->algoStatus.algoStatusLM.region = algoOption.algoOptionLM.initialRegion;
}

SPECIALIZE_CLASS(LMAlgo);
}  // namespace MegBA
