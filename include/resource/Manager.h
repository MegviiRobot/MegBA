/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <mutex>
#include <map>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <nccl.h>

namespace MegBA {
    class HandleManager {
        static std::vector<ncclComm_t> comms_;
        static std::vector<cublasHandle_t> cublasHandle_;
        static std::vector<cusparseHandle_t> cusparseHandle_;
        static std::mutex mutex_;
    public:
        static void create_ncclComm();

        static const std::vector<ncclComm_t> &get_ncclComm();

        static void destroy_ncclComm();

        static void create_cublasHandle();

        static const std::vector<cublasHandle_t> &get_cublasHandle();

        static void destroy_cublasHandle();

        static void create_cusparseHandle();

        static const std::vector<cusparseHandle_t> &get_cusparseHandle();

        static void destroy_cusparseHandle();
    };
}
