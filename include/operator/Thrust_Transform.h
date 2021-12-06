/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include "cmath"

namespace MegBA {
    namespace TT {
        template<typename T>
        struct JetVector_multiplies_JetVector_v {
            __host__ __device__
            T operator()(thrust::tuple<T, T, T, T> zip) {
                T fa, fv, ga, gv;
                thrust::tie(fa, fv, ga, gv) = zip;
                return fa * gv + fv * ga;
            }
        };

        template <typename T>
        struct inverse : public thrust::unary_function<T, T>
        {
            __host__ __device__
            T operator()(T x) { return T(1.) / x; }
        };

        template<typename T>
        struct JetVector_divides_JetVector_v {
            __host__ __device__
            T operator()(thrust::tuple<T, T, T, T> zip) {
                T fa, fv, inv_ga, gv;
                thrust::tie(fa, fv, inv_ga, gv) = zip;
                return (fv - fa * inv_ga * gv) * inv_ga;
            }
        };

        template<typename T>
        struct Scalar_Vector_divides_JetVector_v {
            __host__ __device__
            T operator()(thrust::tuple<T, T, T> zip) {
                T fa, inv_ga, gv;
                thrust::tie(fa, inv_ga, gv) = zip;
                return - fa * inv_ga * gv * inv_ga;
            }
        };

        template <typename T>
        struct scalar_minus_JetVector : public thrust::unary_function<T, T>
        {
            T scalar;
            explicit scalar_minus_JetVector(T scalar) : scalar(scalar) {}
            __host__ __device__
            T operator()(T x) { return scalar - x; }
        };

        template <typename T>
        struct scalar_divides_JetVector_a : public thrust::unary_function<T, T>
        {
            T scalar;
            explicit scalar_divides_JetVector_a(T scalar) : scalar(scalar) {}
            __host__ __device__
            T operator()(T x) { return scalar / x; }
        };

        template <typename T>
        struct scalar_divides_JetVector_v : public thrust::binary_function<T, T, T>
        {
            T scalar;
            explicit scalar_divides_JetVector_v(T scalar) : scalar(scalar) {}
            __host__ __device__
            T operator()(T a, T v) { return -v * scalar / (a * a); }
        };

        template <typename T>
        struct abs_mask : public thrust::unary_function<T, T>
        {
            __host__ __device__
            T operator()(T x) { return x > 0. ? T(1.) : T(-1.); }
        };

        template <typename T>
        struct sin : public thrust::unary_function<T, T>
        {
            __host__ __device__
            T operator()(T x) { return std::sin(x); }
        };

        template <typename T>
        struct negative_sin_multiplies : public thrust::binary_function<T, T, T>
        {
            __host__ __device__
            T operator()(T a, T v) { return -std::sin(a) * v; }
        };

        template <typename T>
        struct cos : public thrust::unary_function<T, T>
        {
            __host__ __device__
            T operator()(T x) { return std::cos(x); }
        };

        template <typename T>
        struct cos_multiplies : public thrust::binary_function<T, T, T>
        {
            __host__ __device__
            T operator()(T a, T v) { return std::cos(a) * v; }
        };

        template <typename T>
        struct sqrt : public thrust::unary_function<T, T>
        {
            __host__ __device__
            T operator()(T x) { return std::sqrt(x); }
        };

        template <typename T>
        struct sqrt_JetVector_v : public thrust::binary_function<T, T, T>
        {
            __host__ __device__
            T operator()(T sqrted_a, T v) { return T(0.5) * v / sqrted_a; }
        };

        template <typename T>
        struct weighted_plus : public thrust::binary_function<T, T, T> {
            T LR;
            explicit weighted_plus(T LR) : LR(LR) {}
            __host__ __device__
            T operator()(T weight, T grad) { return weight + LR * grad; }
        };
    }
}
