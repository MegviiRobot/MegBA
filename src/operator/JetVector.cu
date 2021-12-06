/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <Common.h>
#include <operator/JetVector.h>
#include <resource/Memory_Pool.h>
#include <Macro.h>
#include <memory>

namespace MegBA {
    template<typename T>
    void JetVector<T>::cudaInitAs(const JetVector<T> &f) {
        const auto world_size = Memory_Pool::getWorldSize();
        std::vector<void *> da_ptr, dv_ptr;
        Memory_Pool::allocate_JetVector(da_ptr, dv_ptr, N_, nElm_, sizeof(T));
        dv_ptr_.clear();
        da_ptr_.clear();
        dv_ptr_.resize(world_size);
        da_ptr_.resize(world_size);
        for (int i = 0; i < world_size; ++i) {
            dv_ptr_[i] = (T *) dv_ptr[i];
            da_ptr_[i] = (T *) da_ptr[i];
        }
    }

    template<typename T> JetVector<T>&JetVector<T>::CUDA() {
        if (!IsEmpty()) {
            auto N = N_;
            auto nElm = nElm_;
            switch (device_) {
                case CUDA_t: {
                    break;
                }
                case CPU_t: {
                    // save counter
                    cuCPU_to_CUDA(*this);
                    clear();
                    break;
                }
            }  // switch device_
            N_ = N;
            nElm_ = nElm;
        }
        device_ = CUDA_t;
        return *this;
    }

    template<typename T>
    void JetVector<T>::CUDA2CPU(const JetVector<T> &f) {
        const auto world_size = Memory_Pool::getWorldSize();
        ha_data.resize(nElm_);
        hv_data.resize(N_);
        for (auto &v : hv_data)
            v.resize(nElm_);

        std::size_t start_idx{0};
        for (int i = 0; i < world_size; ++i) {
            cudaSetDevice(i);
            std::size_t nElm{get_Elm_Num(i)};
            cudaMemcpyAsync(&ha_data[start_idx], f.da_ptr_[i], nElm * sizeof(T), cudaMemcpyDeviceToHost);
            if (grad_position_ == -1) {
                for (unsigned int j = 0; j < N_; ++j)
                    cudaMemcpyAsync(&hv_data[j][start_idx], &f.dv_ptr_[i][j * nElm], nElm * sizeof(T), cudaMemcpyDeviceToHost);
            }
            start_idx += nElm;
        }
    }

    template<typename T>
    void JetVector<T>::cuCPU_to_CUDA(const JetVector<T> &f) {
        const auto world_size = Memory_Pool::getWorldSize();
        // if da_ptr_ != nullptr if binded
        if (da_ptr_.empty()) {
            if (pure_scalar_flag_) {
                da_ptr_.resize(world_size);
                std::size_t start_idx{0};
                for (int i = 0; i < world_size; ++i) {
                    cudaSetDevice(i);
                    cudaMalloc(&da_ptr_[i], nElm_ * sizeof(T));
                    std::size_t nElm{get_Elm_Num(i)};
                    cudaMemcpyAsync(da_ptr_[i], &f.ha_data[start_idx], nElm * sizeof(T), cudaMemcpyHostToDevice);
                    start_idx += nElm;
                }
                return;
            }
            std::vector<void *> da_ptr{}, dv_ptr{};
            Memory_Pool::allocate_JetVector(da_ptr, dv_ptr, N_, nElm_, sizeof(T));
            // dv_ptr_ must be nullptr
            dv_ptr_.clear();
            dv_ptr_.reserve(world_size);
            for (int i = 0; i < world_size; ++i)
                dv_ptr_.push_back((T *)dv_ptr[i]);

            da_ptr_.clear();
            da_ptr_.reserve(world_size);
            for (int i = 0; i < world_size; ++i)
                da_ptr_.push_back((T *)da_ptr[i]);

            std::size_t start_idx{0};
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                std::size_t nElm{get_Elm_Num(i)};
                cudaMemcpyAsync(da_ptr_[i], &f.ha_data[start_idx], nElm * sizeof(T), cudaMemcpyHostToDevice);
                for (unsigned int j = 0; j < N_; ++j)
                    cudaMemcpyAsync(&dv_ptr_[i][j * nElm], &f.hv_data[j][start_idx], nElm * sizeof(T), cudaMemcpyHostToDevice);
                start_idx += nElm;
            }
        } else {
            std::size_t start_idx{0};
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                std::size_t nElm{get_Elm_Num(i)};
                cudaMemcpyAsync(da_ptr_[i], &f.ha_data[start_idx], nElm * sizeof(T), cudaMemcpyHostToDevice);
                start_idx += nElm;
            }
        }
    }

    template<typename T>
    void JetVector<T>::CUDA2CUDA(const JetVector<T> &f) {
        const auto world_size = Memory_Pool::getWorldSize();
        if (da_ptr_.empty()) {
            std::vector<void *> da_ptr{}, dv_ptr{};
            Memory_Pool::allocate_JetVector(da_ptr, dv_ptr, N_, nElm_, sizeof(T));
            dv_ptr_.clear();
            da_ptr_.clear();
            dv_ptr_.reserve(world_size);
            da_ptr_.reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                dv_ptr_.push_back((T *) dv_ptr[i]);
                da_ptr_.push_back((T *) da_ptr[i]);
            }
        }
        for (int i = 0; i < world_size; ++i) {
            cudaSetDevice(i);
            std::size_t nElm{get_Elm_Num(i)};
            cudaMemcpyAsync(da_ptr_[i], f.da_ptr_[i], nElm * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(dv_ptr_[i], f.dv_ptr_[i], N_ * nElm * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    template<typename T>
    std::ostream &cuOSTREAM(std::ostream &s, const JetVector<T> &z) {
        auto N = z.get_Grad_Shape();
        auto nElm = z.get_Elm_Num();
        std::unique_ptr<T[]> Res{new T[nElm]};
        std::vector<std::unique_ptr<T []>> Grad;
        Grad.reserve(N);
        for (int i = 0; i < N; ++i)
            Grad.emplace_back(new T[nElm]);
        std::size_t start_idx{0};
        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            std::size_t nElm = z.get_Elm_Num(i);
            cudaMemcpyAsync(&Res[start_idx], z.get_CUDA_Res_ptr()[i], nElm * sizeof(T), cudaMemcpyDeviceToHost);
            for (unsigned int j = 0; j < N; ++j)
                cudaMemcpyAsync(&Grad[j][start_idx], &z.get_CUDA_Grad_ptr()[i][j * nElm], nElm * sizeof(T), cudaMemcpyDeviceToHost);
            start_idx += nElm;
        }
        s << "[Res: " << "[ ";
        for (std::size_t i = 0; i < nElm; ++i)
            s << Res[i] << ", ";
        s << "]," << std::endl;
        for (unsigned int i = 0; i < N; ++i) {
            s << "Grad[" << i << "]: " << "[ ";
            for (std::size_t j = 0; j < nElm; ++j)
                s << Grad[i][j] << ", ";
            s << "]," << std::endl;
        }
        s << "device_: " << std::to_string(z.get_Device())  << "]";
        return s;
    }
    template std::ostream &cuOSTREAM<float>(std::ostream &s, const JetVector<float> &);
    template std::ostream &cuOSTREAM<double>(std::ostream &s, const JetVector<double> &);

    template class JetVector<float>;
    template class JetVector<double>;
}  // namespace MegBA