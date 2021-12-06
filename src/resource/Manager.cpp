/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <cassert>
#include <resource/Memory_Pool.h>
#include <resource/Manager.h>

namespace MegBA {
    void HandleManager::create_ncclComm() {
        std::vector<int> devs;
        devs.resize(Memory_Pool::getWorldSize());
        comms_.resize(Memory_Pool::getWorldSize());
        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i)
            devs[i] = i;
        ncclCommInitAll(comms_.data(), Memory_Pool::getWorldSize(), devs.data());
    }

    const std::vector<ncclComm_t> &HandleManager::get_ncclComm() {
        return comms_;
    }

    void HandleManager::destroy_ncclComm() {
        for (auto comm : comms_) {
            ncclCommDestroy(comm);
        }
    }

    void HandleManager::create_cublasHandle() {
        std::unique_lock<std::mutex> lock{mutex_};
        assert(cublasHandle_.empty());
        cublasHandle_.resize(Memory_Pool::getWorldSize());
        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            cublasCreate_v2(&cublasHandle_[i]);
        }
    }

    const std::vector<cublasHandle_t> &HandleManager::get_cublasHandle() {
        std::unique_lock<std::mutex> lock{mutex_};
        assert(!cublasHandle_.empty());
        return cublasHandle_;
    }

    void HandleManager::destroy_cublasHandle() {
        std::unique_lock<std::mutex> lock{mutex_};
        for (int i = 0; i < cublasHandle_.size(); ++i) {
            cudaSetDevice(i);
            cublasDestroy_v2(cublasHandle_[i]);
        }
        cublasHandle_.clear();
    }

    void HandleManager::create_cusparseHandle() {
        std::unique_lock<std::mutex> lock{mutex_};
        assert(cusparseHandle_.empty());
        cusparseHandle_.resize(Memory_Pool::getWorldSize());
        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            cusparseCreate(&cusparseHandle_[i]);
        }
    }

    const std::vector<cusparseHandle_t> &HandleManager::get_cusparseHandle() {
        std::unique_lock<std::mutex> lock{mutex_};
        assert(!cusparseHandle_.empty());
        return cusparseHandle_;
    }

    void HandleManager::destroy_cusparseHandle() {
        std::unique_lock<std::mutex> lock{mutex_};
        for (int i = 0; i < cusparseHandle_.size(); ++i) {
            cudaSetDevice(i);
            cusparseDestroy(cusparseHandle_[i]);
        }
        cusparseHandle_.clear();
    }

    std::vector<ncclComm_t> HandleManager::comms_{};
    std::vector<cublasHandle_t> HandleManager::cublasHandle_{};
    std::vector<cusparseHandle_t> HandleManager::cusparseHandle_{};
    std::mutex HandleManager::mutex_{};
}
