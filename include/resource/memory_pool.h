/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <cassert>
#include <vector>

#include "common.h"

namespace MegBA {
class MemoryPool {
  static const ProblemOption *_problemOption;
  static std::vector<std::vector<void *>> _ptr;
  static std::vector<std::size_t> _poolSize;
  static std::vector<void *> _headPtr;
  static std::uint8_t _sizeofType;
  static std::size_t _ptrInUseCounter;

 public:
  static void resetPool(const ProblemOption *problemOption,
                        std::int8_t sizeofType);

  static void allocateJetVector(std::vector<void *> &valueDevicePtr,
                                std::vector<void *> &gradDevicePtr,
                                std::size_t N, std::size_t nItem,
                                std::int8_t sizeofType);

  static void deallocateJetVector(std::vector<void *> &ptr);

  static void allocateNormal(void **ptr, size_t size, int rank = 0);

  static void deallocateNormal(void *ptr, int rank = 0);

  static std::size_t getWorldSize() {
    return _problemOption->deviceUsed.size();
  }

  static const std::vector<int> &getWorld() {
    return _problemOption->deviceUsed;
  }

  static void redistribute();

  static std::size_t getItemNum(int rank) {
    const auto worldSize = getWorldSize();
    if (rank == worldSize - 1)
      return _problemOption->nItem -
             (_problemOption->nItem / worldSize + 1) * (worldSize - 1);
    else
      return _problemOption->nItem / worldSize + 1;
  }

  static std::size_t getItemNum(int rank, std::size_t nItem) {
    const auto worldSize = getWorldSize();
    if (rank == worldSize - 1)
      return nItem - (nItem / worldSize + 1) * (worldSize - 1);
    else
      return nItem / worldSize + 1;
  }
  static void destruct();
};
}  // namespace MegBA
