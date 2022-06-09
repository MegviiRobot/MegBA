/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <cublas_v2.h>
#include <cusparse_v2.h>
#ifdef MEGBA_ENABLE_NCCL
#include <nccl.h>
#endif

#include <vector>

namespace MegBA {
class HandleManager {
#ifdef MEGBA_ENABLE_NCCL
  static std::vector<ncclComm_t> comms;
#endif
  static std::vector<cublasHandle_t> cublasHandle;
  static std::vector<cusparseHandle_t> cusparseHandle;

 public:
#ifdef MEGBA_ENABLE_NCCL
  static void createNCCLComm();
#endif

#ifdef MEGBA_ENABLE_NCCL
  static const std::vector<ncclComm_t> &getNCCLComm();
#endif

#ifdef MEGBA_ENABLE_NCCL
  static void destroyNCCLComm();
#endif

  static void createCUBLASHandle();

  static const std::vector<cublasHandle_t> &getCUBLASHandle();

  static void destroyCUBLASHandle();

  static void createCUSPARSEHandle();

  static const std::vector<cusparseHandle_t> &getCUSPARSEHandle();

  static void destroyCUSPARSEHandle();
};
}  // namespace MegBA
