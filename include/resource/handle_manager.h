/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <mutex>
#include <vector>
#include <map>

namespace MegBA {
class HandleManager {
  static std::vector<ncclComm_t> _comms;
  static std::vector<cublasHandle_t> _cublasHandle;
  static std::vector<cusparseHandle_t> _cusparseHandle;
  static std::mutex _mutex;

 public:
  static void createNcclComm();

  static const std::vector<ncclComm_t> &getNcclComm();

  static void destroyNcclComm();

  static void createCublasHandle();

  static const std::vector<cublasHandle_t> &getCublasHandle();

  static void destroyCublasHandle();

  static void createCusparseHandle();

  static const std::vector<cusparseHandle_t> &getCusparseHandle();

  static void destroyCusparseHandle();
};
}  // namespace MegBA
