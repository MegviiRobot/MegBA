/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <unordered_map>
#include <resource/Memory_Pool.h>
#include <resource/Manager.h>
#include <stack>

namespace MegBA {
    namespace {
        union Ptr_t {
            explicit Ptr_t(void *address) : address_(address) {};
            void *address_;
#if __SIZEOF_POINTER__ == 8
            std::uint64_t number_;
#elif __SIZEOF_POINTER__ == 4
            std::uint32_t number_;
#elif __SIZEOF_POINTER__ == 2
            std::uint16_t number;
#endif
        };

        std::vector<std::stack<std::pair<void *, std::size_t>>> ptr_recorder{};
        std::vector<std::stack<std::pair<void *, std::size_t>>> overflowed_ptr_recorder{};

        std::vector<std::size_t> mem_offset_counter{};

        std::vector<std::size_t> mem_overflowed_counter{};

        std::vector<std::size_t> mem_overflowed_peak{};

    }

    void Memory_Pool::reset_Pool(int N, std::size_t nElm, std::int8_t sizeof_type, int world_size) {
        // TODO: maybe destroy only once
        std::unique_lock<std::mutex> lock{mutex_};
        N_ = N;
        nElm_ = nElm;
        sizeof_type_ = sizeof_type;
        world_size_ = world_size;
        HandleManager::destroy_ncclComm();
        HandleManager::create_ncclComm();
        HandleManager::destroy_cublasHandle();
        HandleManager::destroy_cusparseHandle();
        HandleManager::create_cublasHandle();
        HandleManager::create_cusparseHandle();
    }

    void Memory_Pool::allocate_JetVector(std::vector<void *> &da_ptr, std::vector<void *> &dv_ptr,
                                          std::size_t N, std::size_t nElm, std::int8_t sizeof_type) {
        std::unique_lock<std::mutex> lock{mutex_};
        da_ptr.clear();
        da_ptr.resize(world_size_);
        dv_ptr.clear();
        dv_ptr.resize(world_size_);
        assert((N == N_ || N == 0) && nElm == nElm_ && sizeof_type == sizeof_type_);
        for (auto offset : mem_offset_counter)
            if (offset != 0)
                throw std::runtime_error("memory leak");
        if (ptr_.empty()) {
            for (int i = 0; i < world_size_; ++i) {
                const auto nElm = getElmNum(i);
                cudaSetDevice(i);
                Ptr_t ptr{nullptr};
                cudaMalloc(&ptr.address_, (N_ + 1) * nElm * sizeof_type_);
                dv_ptr[i] = ptr.address_;
                ptr.number_ += N_ * nElm * sizeof_type_;
                da_ptr[i] = ptr.address_;
            }
        } else {
            std::vector<void *> back = std::move(ptr_.back());
            ptr_.pop_back();
            for (int i = 0; i < world_size_; ++i) {
                const auto nElm = getElmNum(i);
                cudaSetDevice(i);
                Ptr_t ptr{back[i]};
                dv_ptr[i] = ptr.address_;
                ptr.number_ += N_ * nElm * sizeof_type_;
                da_ptr[i] = ptr.address_;
            }
        }
        ptr_in_use_counter_++;
    }

    void Memory_Pool::deallocate_JetVector(std::vector<void *> &ptr) {
        std::unique_lock<std::mutex> lock{mutex_};
        ptr_.push_back(std::move(ptr));
        ptr_in_use_counter_--;
    }

    void Memory_Pool::allocate_normal(void **ptr, std::size_t size, int rank) {
        size += size % 8;
        std::unique_lock<std::mutex> lock{mutex_};
        Ptr_t ptr_helper{nullptr};

        if (mem_offset_counter.empty()) {
            mem_offset_counter.resize(world_size_);
            ptr_recorder.resize(world_size_);
            std::fill(mem_offset_counter.begin(), mem_offset_counter.end(), 0);
        }

        bool use_overflowed_stack{pool_size_[rank] < (mem_offset_counter[rank] + size)};
        if (use_overflowed_stack) {
            if (overflowed_ptr_recorder.empty()) {
                overflowed_ptr_recorder.resize(world_size_);
                mem_overflowed_counter.resize(world_size_);
                mem_overflowed_peak.resize(world_size_);
                std::fill(mem_overflowed_counter.begin(), mem_overflowed_counter.end(), 0);
                std::fill(mem_overflowed_peak.begin(), mem_overflowed_peak.end(), 0);
            }

            mem_overflowed_peak[rank] = std::max(mem_overflowed_peak[rank], mem_offset_counter[rank] + size - pool_size_[rank]);
            cudaSetDevice(rank);
            cudaMalloc(&ptr_helper.address_, size);
            overflowed_ptr_recorder[rank].emplace(ptr_helper.address_, size);
            mem_overflowed_counter[rank] += size;
        } else {
            ptr_helper.address_ = head_ptr_[rank];
            ptr_helper.number_ += mem_offset_counter[rank];
            mem_offset_counter[rank] += size;
        }
        *ptr = ptr_helper.address_;
        if (!use_overflowed_stack) {
            ptr_recorder[rank].emplace(ptr_helper.address_, size);
        }
    }

    void Memory_Pool::deallocate_normal(void *ptr, int rank) {
        std::unique_lock<std::mutex> lock{mutex_};
        std::pair<void *, std::size_t> back;
        if (ptr_recorder[rank].top().first == ptr) {
            back = std::move(ptr_recorder[rank].top());
            ptr_recorder[rank].pop();
            mem_offset_counter[rank] -= back.second;
        } else {
            if (!overflowed_ptr_recorder[rank].empty() && overflowed_ptr_recorder[rank].top().first == ptr) {
                back = std::move(overflowed_ptr_recorder[rank].top());
                overflowed_ptr_recorder[rank].pop();
                cudaSetDevice(rank);
                cudaFree(back.first);
                mem_overflowed_counter[rank] -= back.second;
            } else {
                throw std::runtime_error("not using a stack style malloc-free");
            }
        }
    }

    void Memory_Pool::redistribute() {
        if (pool_size_.empty()) {
            pool_size_.resize(world_size_);
            head_ptr_.resize(world_size_);
            for (int i = 0; i < world_size_; ++i) {
                cudaSetDevice(i);
                const auto nElm = getElmNum(i);
                for (const auto &v: ptr_) {
                    cudaFree(v[i]);
                }
                pool_size_[i] = (N_ + 1) * nElm * sizeof_type_ * ptr_.size();
                cudaMalloc(&head_ptr_[i], pool_size_[i]);
                int64_t offset{0};
                for (auto &item: ptr_) {
                    Ptr_t ptr{head_ptr_[i]};
                    ptr.number_ += offset;
                    offset += (N_ + 1) * nElm * sizeof_type_;
                    item[i] = ptr.address_;
                }
            }
        } else {
            bool overflowed{false};
            for (auto peak : mem_overflowed_peak)
                overflowed |= peak != 0;
            if (overflowed) {
                for (int i = 0; i < world_size_; ++i) {
                    cudaSetDevice(i);
                    const auto nElm = getElmNum(i);
                    cudaFree(head_ptr_[i]);
                    pool_size_[i] += mem_overflowed_peak[i];
                    cudaMalloc(&head_ptr_[i], pool_size_[i]);
                    int64_t offset{0};
                    for (auto &item: ptr_) {
                        Ptr_t ptr{head_ptr_[i]};
                        ptr.number_ += offset;
                        offset += (N_ + 1) * nElm * sizeof_type_;
                        item[i] = ptr.address_;
                    }
                }
            }
        }
    }

    std::vector<std::vector<void*>> Memory_Pool::ptr_{};
    std::mutex Memory_Pool::mutex_{};
    std::vector<std::size_t> Memory_Pool::pool_size_{};
    std::vector<void*> Memory_Pool::head_ptr_{};
    int Memory_Pool::N_{0};
    std::size_t Memory_Pool::nElm_{0};
    std::uint8_t Memory_Pool::sizeof_type_{0};
    int Memory_Pool::world_size_{1};
    std::size_t Memory_Pool::ptr_in_use_counter_{0};
}