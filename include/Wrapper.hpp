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

__device__ float rsqrtf(float a);
__device__ double rsqrt(double a);

#define CUXXX_WRAPPER(name, Function_FP32, Function_FP64)                                           \
class name {                                                                                        \
    using features = function_args_type<decltype(Function_FP32), decltype(Function_FP64)>::features;\
    template<typename Common_Tuple_t>                                                               \
    struct wrapper;                                                                                 \
    template<typename ...Common_t>                                                                  \
    struct wrapper<std::tuple<Common_t...>> {                                                       \
        template<typename ...After_t>                                                               \
        static auto call(Common_t ...before, features::FP32_t here, After_t &&...after) {           \
            return Function_FP32(before..., here, std::forward<After_t>(after)...);                 \
        }                                                                                           \
        template<typename ...After_t>                                                               \
        static auto call(Common_t ...before, features::FP64_t here, After_t &&...after) {           \
            return Function_FP64(before..., here, std::forward<After_t>(after)...);                 \
        }                                                                                           \
    };                                                                                              \
    public:                                                                                         \
    template<typename ...Args>                                                                      \
    static auto call(Args&&... args) {                                                              \
        return wrapper<features::Common_Tuple_t>::call(std::forward<Args>(args)...);                \
    };                                                                                              \
};

namespace MegBA {
    namespace Wrapper {
        namespace {
            template<typename Before_Tuple_t,
                     typename FP32_After_Tuple_t, typename FP64_After_Tuple_t>
            struct common_args_helper;

            template<typename T, typename... Before_t>
            struct common_args_helper<std::tuple<Before_t...>,
                                      std::tuple<T>, std::tuple<T>> {
                typedef std::tuple<Before_t...> Common_Tuple_t;
                typedef T FP32_t;
                typedef T FP64_t;
            };

            template<typename T, typename... Before_t, typename... After_FP32_t, typename... After_FP64_t>
            struct common_args_helper<std::tuple<Before_t...>, std::tuple<T, After_FP32_t...>, std::tuple<T, After_FP64_t...>> {
                using Next_Helper = common_args_helper<std::tuple<Before_t..., T>, std::tuple<After_FP32_t...>, std::tuple<After_FP64_t...>>;
                typedef typename Next_Helper::Common_Tuple_t Common_Tuple_t;
                typedef typename Next_Helper::FP32_t FP32_t;
                typedef typename Next_Helper::FP64_t FP64_t;
            };

            template<typename FP32_t_, typename FP64_t_,
                     typename... Before_t,
                     typename... After_FP32_t, typename... After_FP64_t>
            struct common_args_helper<std::tuple<Before_t...>, std::tuple<FP32_t_, After_FP32_t...>, std::tuple<FP64_t_, After_FP64_t...>> {
                typedef std::tuple<Before_t...> Common_Tuple_t;
                typedef FP32_t_ FP32_t;
                typedef FP64_t_ FP64_t;
            };

            template<typename Tuple_FP32_t, typename Tuple_FP64_t>
            struct common_args {
                using Helper = common_args_helper<std::tuple<>, Tuple_FP32_t, Tuple_FP64_t>;
                typedef typename Helper::Common_Tuple_t Common_Tuple_t;
                typedef typename Helper::FP32_t FP32_t;
                typedef typename Helper::FP64_t FP64_t;
                static_assert(std::tuple_size<Tuple_FP32_t>::value == std::tuple_size<Tuple_FP64_t>::value, "different arguments num");
                static_assert(!std::is_same<Tuple_FP32_t, Tuple_FP64_t>::value, "same arguments type");
            };

            template<typename Function_FP32_t, typename Function_FP64_t>
            struct function_args_type;

            template<typename Return_FP32_t, typename... Args_FP32_t, typename Return_FP64_t, typename... Args_FP64_t>
            struct function_args_type<Return_FP32_t(Args_FP32_t...), Return_FP64_t(Args_FP64_t...)> {
                typedef common_args<std::tuple<Args_FP32_t...>, std::tuple<Args_FP64_t...>> features;
            };
        }

        CUXXX_WRAPPER(cublasGaxpy, cublasSaxpy_v2, cublasDaxpy_v2);

        CUXXX_WRAPPER(cublasGgeam, cublasSgeam, cublasDgeam);

        CUXXX_WRAPPER(cublasGdot, cublasSdot_v2, cublasDdot_v2);

        CUXXX_WRAPPER(cublasGcopy, cublasScopy_v2, cublasDcopy_v2);

        CUXXX_WRAPPER(cublasGscal, cublasSscal_v2, cublasDscal_v2);

        CUXXX_WRAPPER(cusparseGcsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt, cusparseDcsrgeam2_bufferSizeExt);

        CUXXX_WRAPPER(cusparseGcsrgeam2, cusparseScsrgeam2, cusparseDcsrgeam2);

        CUXXX_WRAPPER(cublasGgetrfBatched, cublasSgetrfBatched, cublasDgetrfBatched);

        CUXXX_WRAPPER(cublasGgetriBatched, cublasSgetriBatched, cublasDgetriBatched);

        CUXXX_WRAPPER(cublasGmatinvBatched, cublasSmatinvBatched, cublasDmatinvBatched);


        template<typename T>
        struct declared_cudaDatatype {
        };

        template<>
        struct declared_cudaDatatype<float> {
            constexpr static const ncclDataType_t nccl_dtype{ncclFloat32};
            constexpr static const cudaDataType_t cuda_dtype{CUDA_R_32F};
        };

        template<>
        struct declared_cudaDatatype<double> {
            constexpr static const ncclDataType_t nccl_dtype{ncclFloat64};
            constexpr static const cudaDataType_t cuda_dtype{CUDA_R_64F};
        };

        template<typename T>
        struct Shared_Memory {
        };

        template<>
        struct Shared_Memory<float> {
            static __device__ float *get() {
                extern __shared__ float s_float[];
                return s_float;
            }
        };

        template<>
        struct Shared_Memory<double> {
            static __device__ double *get() {
                extern __shared__ double s_double[];
                return s_double;
            }
        };

        template<>
        struct Shared_Memory<int> {
            static __device__ int *get() {
                extern __shared__ int s_int[];
                return s_int;
            }
        };

        template<typename T>
        struct sincosG {
        };

        template<>
        struct sincosG<float> {
            constexpr static void (*const call)(float, float *, float *) = sincosf;
        };

        template<>
        struct sincosG<double> {
            constexpr static void (*const call)(double, double *, double *) = sincos;
        };

        template<typename T>
        struct sqrtG {
        };

        template<>
        struct sqrtG<float> {
            constexpr static float (*const call)(float) = sqrtf;
        };

        template<>
        struct sqrtG<double> {
            constexpr static double (*const call)(double) = sqrt;
        };

        template<typename T>
        struct rsqrtG {
        };

        template<>
        struct rsqrtG<float> {
            constexpr static float (*const call)(float) = rsqrtf;
        };

        template<>
        struct rsqrtG<double> {
            constexpr static double (*const call)(double) = rsqrt;
        };
    }
}
#undef CUXXX_WRAPPER