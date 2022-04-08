/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <nccl.h>

#include <vector>

namespace MegBA {
class HandleManager {
  static std::vector<ncclComm_t> comms;
  static std::vector<cublasHandle_t> cublasHandle;
  static std::vector<cusparseHandle_t> cusparseHandle;

 public:
  static void createNCCLComm();

  static const std::vector<ncclComm_t> &getNCCLComm();

  static void destroyNCCLComm();

  static void createCUBLASHandle();

  static const std::vector<cublasHandle_t> &getCUBLASHandle();

  static void destroyCUBLASHandle();

  static void createCUSPARSEHandle();

  static const std::vector<cusparseHandle_t> &getCUSPARSEHandle();

  static void destroyCUSPARSEHandle();
};
}  // namespace MegBA
