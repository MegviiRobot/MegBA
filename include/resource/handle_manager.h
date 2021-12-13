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
  static std::vector<ncclComm_t> comms;
  static std::vector<cublasHandle_t> cublasHandle;
  static std::vector<cusparseHandle_t> cusparseHandle;
  static std::mutex mutex;

 public:
  static void createNcclComm();

  static const std::vector<ncclComm_t> &getNcclComm();

  static void destroyNcclComm();

  static void createCUBLASHandle();

  static const std::vector<cublasHandle_t> &getCUBLASHandle();

  static void destroyCUBLASHandle();

  static void createCUSPARSEHandle();

  static const std::vector<cusparseHandle_t> &getCUSPARSEHandle();

  static void destroyCUSPARSEHandle();
};
}  // namespace MegBA
