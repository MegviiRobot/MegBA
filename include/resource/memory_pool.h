/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <cassert>
#include <mutex>
#include <vector>

namespace MegBA {
class MemoryPool {
  static std::vector<std::vector<void *>> _ptr;
  static std::mutex _mutex;
  static std::vector<std::size_t> _poolSize;
  static std::vector<void *> _headPtr;
  static int _N;
  static std::size_t _nElm;
  static std::uint8_t _sizeofType;
  static int _worldSize;
  static std::size_t _ptrInUseCounter;

 public:
  static void resetPool(int N, std::size_t nElm, std::int8_t sizeofType,
                        int worldSize);

  static void allocateJetVector(std::vector<void *> *daPtr,
                                std::vector<void *> *dvPtr, std::size_t N,
                                std::size_t nElm, std::int8_t sizeofType);

  static void deallocateJetVector(std::vector<void *> *ptr);

  static void allocateNormal(void **ptr, size_t size, int rank = 0);

  static void deallocateNormal(void *ptr, int rank = 0);

  static int getWorldSize() { return _worldSize; }

  static void redistribute();

  static std::size_t getElmNum(int rank) {
    if (rank == _worldSize - 1)
      return _nElm - (_nElm / _worldSize + 1) * (_worldSize - 1);
    else
      return _nElm / _worldSize + 1;
  }

  static std::size_t getElmNum(int rank, std::size_t nElm) {
    if (rank == _worldSize - 1)
      return nElm - (nElm / _worldSize + 1) * (_worldSize - 1);
    else
      return nElm / _worldSize + 1;
  }
};
}  // namespace MegBA
