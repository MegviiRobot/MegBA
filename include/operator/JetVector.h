/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <cassert>
#include "Common.h"
#include "JetVector.inl"
#include "vector"
#include "Jet_Vector_math.inl"
#include <resource/Memory_Pool.h>
#include <functional>
#include <resource/Manager.h>

namespace MegBA {
    template <typename T>
    class JetVector {
        // cuda functions
        void cudaInitAs(const JetVector<T> &f);
        void cuCPU_to_CUDA(const JetVector<T> &f);
        void CUDA2CUDA(const JetVector<T> &f);
        void CUDA2CPU(const JetVector<T> &f);

        unsigned int N_ = 0;
        unsigned int nElm_ = 0;
        device_t device_ = CPU_t;
        std::vector<std::vector<T>> hv_data{};
        std::vector<T*> dv_ptr_{};
        std::vector<T> ha_data{};
        std::vector<T*> da_ptr_{};
        int grad_position_ = -1;
        bool pure_scalar_flag_ = false;
        T pure_scalar_ = 0;
    public:
      JetVector()= default;

      JetVector(T scalar) : pure_scalar_flag_(true), pure_scalar_(scalar) { };

      JetVector(const JetVector<T> &f) :
                N_(f.N_),
                nElm_(f.nElm_),
                device_(f.device_),
                ha_data(f.ha_data),
                hv_data(f.hv_data) {
            switch (device_) {
                case CPU_t: {
                    break;
                }
                case CUDA_t: {
                  CUDA2CUDA(f);
                    break;
                }
            }
        };

        JetVector(JetVector<T> &&f) noexcept :
                N_(std::move(f.N_)),
                nElm_(std::move(f.nElm_)),
                device_(std::move(f.device_)),
                hv_data(std::move(f.hv_data)),
                dv_ptr_(std::move(f.dv_ptr_)),
                ha_data(std::move(f.ha_data)),
                da_ptr_(std::move(f.da_ptr_)),
                grad_position_(std::move(f.grad_position_)),
                pure_scalar_flag_(std::move(f.pure_scalar_flag_)),
                pure_scalar_(std::move((f.pure_scalar_))) {
            f.N_ = 0;
            f.nElm_ = 0;
            f.grad_position_ = -1;
        };

        template<typename F>
        JetVector(const JetVector<T> &init_template, F &&math_func) {
          InitAs(init_template);
            math_func(*this);
        }

        ~JetVector() { clear(); };

        void append_Jet(T a, int n);
        void append_Jet(T a);
        void clear();

        static const JetVector<T> &get_Init_Template(const JetVector<T> &f, const JetVector<T> &g) {
            return f.N_ > g.N_ ? f : g;
        };

        void InitAs(const JetVector<T> &f);
        JetVector<T>& to(device_t device);
        JetVector<T>& CPU();
        JetVector<T>& CUDA();
        bool IsEmpty();
        void set_Grad_Shape(unsigned int N);
        void erase(std::size_t idx) {
            assert(device_ == CPU_t || grad_position_ != -1 || N_ == 0);
            ha_data.erase(ha_data.begin() + idx);
            nElm_--;
        };
        const unsigned int& getGradShape() const { return N_; };
        const unsigned int& get_Elm_Num() const { return nElm_; };
        std::size_t get_Elm_Num(int rank) const {
            return Memory_Pool::getElmNum(rank);
        };
        int get_Grad_Position() const { return grad_position_; };
        const device_t& get_Device() const { return device_; };

        const std::vector<std::vector<T>>& get_CPU_Grad() const { return hv_data; };
        std::vector<std::vector<T>>& get_CPU_Grad() { return hv_data; };

        const std::vector<T>& get_CPU_Res() const { return ha_data; };
        std::vector<T>& get_CPU_Res() { return ha_data; };

        // TODO: change to vector
        const std::vector<T *>& get_CUDA_Grad_ptr() const { return dv_ptr_; };
        const std::vector<T *>& get_CUDA_Res_ptr() const { return da_ptr_; };

        // TODO: input a array vector
        void bind_da_ptr(T* da_ptr) { da_ptr_.resize(Memory_Pool::getWorldSize()); da_ptr_[0] = da_ptr; };

        void bind_da_ptr(std::vector<T *> &&da_ptr) { da_ptr_ = std::move(da_ptr); };

        void set_Grad_Position(int grad_position);

        JetVector<T>& operator=(const JetVector<T> &f);

        JetVector<T>& operator=(JetVector<T> &&f) noexcept;

        JetVector<T> operator+(const JetVector<T> &g) const;

        JetVector<T> operator-(const JetVector<T> &g) const;

        JetVector<T> operator*(const JetVector<T> &g) const;

        JetVector<T> operator/(const JetVector<T> &g) const;

        JetVector<T> &operator+=(const JetVector<T> &g);

        JetVector<T> &operator-=(const JetVector<T> &g);

        JetVector<T> &operator*=(const JetVector<T> &g);

        JetVector<T> &operator/=(const JetVector<T> &g);

        JetVector<T> operator-() const;

        JetVector<T> operator+(T g) const;

        JetVector<T> operator-(T g) const;

        JetVector<T> operator*(T g) const;

        JetVector<T> operator/(T g) const;

        JetVector<T> &operator+=(T g);

        JetVector<T> &operator-=(T g);

        JetVector<T> &operator*=(T g);

        JetVector<T> &operator/=(T g);

        JetVector<T> Scalar_minus_this(T g) const;
        JetVector<T> Scalar_divides_this(T g) const;
    };

    template<typename T>
    std::ostream &cuOSTREAM(std::ostream &s, const JetVector<T> &z);

    template<typename T>
    inline std::ostream &operator<<(std::ostream &s, const JetVector<T> &z) {
        switch (z.get_Device()) {
            case CPU_t: {
                s << "[Res: " << "[ ";
                for (auto &data : z.get_CPU_Res())
                    s << data << ", ";
                s << "]," << std::endl;
                for (unsigned int i = 0; i < z.getGradShape(); ++i) {
                    s << "Grad[" << i << "]: " << "[ ";
                    for (auto &data : z.get_CPU_Grad()[i])
                        s << data << ", ";
                    s << "]," << std::endl;
                }
                s << "device_: " << std::to_string(z.get_Device())  << "]";
                break;
            }
            case CUDA_t: {
                return cuOSTREAM(s, z);
            }
        }
        return s;
    }
}
