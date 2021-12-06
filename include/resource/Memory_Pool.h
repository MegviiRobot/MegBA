/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <mutex>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

namespace MegBA {
    class Memory_Pool {
    private:
        static std::vector<std::vector<void *>> ptr_;
        static std::mutex mutex_;
//        static std::size_t block_size_;
        static std::vector<std::size_t> pool_size_;
        static std::vector<void *> head_ptr_;
        static int N_;
        static std::size_t nElm_;
        static std::uint8_t sizeof_type_;
        static int world_size_;
        static std::size_t ptr_in_use_counter_;

    public:
        static void reset_Pool(int N, std::size_t nElm, std::int8_t sizeof_type, int world_size);

        static void allocate_JetVector(std::vector<void *> &da_ptr, std::vector<void *> &dv_ptr,
                                        std::size_t N, std::size_t nElm, std::int8_t sizeof_type);

        static void deallocate_JetVector(std::vector<void *> &ptr);

        static void allocate_normal(void **ptr, size_t size, int rank=0);

        static void deallocate_normal(void *ptr, int rank=0);

        static int getWorldSize() { return world_size_; };

        static void redistribute();

        static std::size_t getElmNum(int rank) {
            if (rank == world_size_ - 1)
                return  nElm_ - (nElm_ / world_size_ + 1) * (world_size_ - 1);
            else
                return nElm_ / world_size_ + 1;
        }

        static std::size_t getElmNum(int rank, std::size_t nElm) {
            if (rank == world_size_ - 1)
                return  nElm - (nElm / world_size_ + 1) * (world_size_ - 1);
            else
                return nElm / world_size_ + 1;
        }
    };
}
