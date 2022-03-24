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

#include <tuple>
#include <utility>

__device__ float rsqrtf(float a);
__device__ double rsqrt(double a);

#define CUXXX_WRAPPER(name, functionFP32, functionFP64)                      \
  class name {                                                               \
    using features = FunctionArgsType<decltype(functionFP32),                \
                                      decltype(functionFP64)>::features;     \
    template <typename CommonTuple>                                          \
    struct wrapper;                                                          \
    template <typename... Common>                                            \
    struct wrapper<std::tuple<Common...>> {                                  \
      template <typename... After>                                           \
      static auto call(Common... before, features::FP32 here,                \
                       After &&...after) {                                   \
        return functionFP32(before..., here, std::forward<After>(after)...); \
      }                                                                      \
      template <typename... After>                                           \
      static auto call(Common... before, features::FP64 here,                \
                       After &&...after) {                                   \
        return functionFP64(before..., here, std::forward<After>(after)...); \
      }                                                                      \
    };                                                                       \
                                                                             \
   public:                                                                   \
    template <typename... Args>                                              \
    static auto call(Args &&...args) {                                       \
      return wrapper<features::CommonTuple>::call(                           \
          std::forward<Args>(args)...);                                      \
    }                                                                        \
  };

namespace MegBA {
namespace Wrapper {
template <typename BeforeTuple, typename FP32AfterTuple,
          typename FP64AfterTuple>
struct CommonArgsHelper;

template <typename T, typename... Before>
struct CommonArgsHelper<std::tuple<Before...>, std::tuple<T>, std::tuple<T>> {
  typedef std::tuple<Before...> CommonTuple;
  typedef T FP32;
  typedef T FP64;
};

template <typename T, typename... Before, typename... AfterFP32,
          typename... AfterFP64>
struct CommonArgsHelper<std::tuple<Before...>, std::tuple<T, AfterFP32...>,
                        std::tuple<T, AfterFP64...>> {
  using Next_Helper =
      CommonArgsHelper<std::tuple<Before..., T>, std::tuple<AfterFP32...>,
                       std::tuple<AfterFP64...>>;
  typedef typename Next_Helper::CommonTuple CommonTuple;
  typedef typename Next_Helper::FP32 FP32;
  typedef typename Next_Helper::FP64 FP64;
};

template <typename FP32_, typename FP64_, typename... Before,
          typename... AfterFP32, typename... AfterFP64>
struct CommonArgsHelper<std::tuple<Before...>, std::tuple<FP32_, AfterFP32...>,
                        std::tuple<FP64_, AfterFP64...>> {
  typedef std::tuple<Before...> CommonTuple;
  typedef FP32_ FP32;
  typedef FP64_ FP64;
};

template <typename TupleFP32, typename TupleFP64>
struct CommonArgs {
  using Helper = CommonArgsHelper<std::tuple<>, TupleFP32, TupleFP64>;
  typedef typename Helper::CommonTuple CommonTuple;
  typedef typename Helper::FP32 FP32;
  typedef typename Helper::FP64 FP64;
  static_assert(std::tuple_size<TupleFP32>::value ==
                    std::tuple_size<TupleFP64>::value,
                "different arguments num");
  static_assert(!std::is_same<TupleFP32, TupleFP64>::value,
                "same arguments type");
};

template <typename FunctionFP32, typename FunctionFP64>
struct FunctionArgsType;

template <typename ReturnFP32, typename... ArgsFP32, typename ReturnFP64,
          typename... ArgsFP64>
struct FunctionArgsType<ReturnFP32(ArgsFP32...), ReturnFP64(ArgsFP64...)> {
  typedef CommonArgs<std::tuple<ArgsFP32...>, std::tuple<ArgsFP64...>> features;
};

CUXXX_WRAPPER(cublasGaxpy, cublasSaxpy_v2, cublasDaxpy_v2);

CUXXX_WRAPPER(cublasGgeam, cublasSgeam, cublasDgeam);

CUXXX_WRAPPER(cublasGdot, cublasSdot_v2, cublasDdot_v2);

CUXXX_WRAPPER(cublasGcopy, cublasScopy_v2, cublasDcopy_v2);

CUXXX_WRAPPER(cublasGscal, cublasSscal_v2, cublasDscal_v2);

CUXXX_WRAPPER(cusparseGcsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt,
              cusparseDcsrgeam2_bufferSizeExt);

CUXXX_WRAPPER(cusparseGcsrgeam2, cusparseScsrgeam2, cusparseDcsrgeam2);

CUXXX_WRAPPER(cublasGgetrfBatched, cublasSgetrfBatched, cublasDgetrfBatched);

CUXXX_WRAPPER(cublasGgetriBatched, cublasSgetriBatched, cublasDgetriBatched);

CUXXX_WRAPPER(cublasGmatinvBatched, cublasSmatinvBatched, cublasDmatinvBatched);

template <typename T>
struct declaredDtype {};

template <>
struct declaredDtype<float> {
  constexpr static const ncclDataType_t ncclDtype{ncclFloat32};
  constexpr static const cudaDataType cudaDtype{CUDA_R_32F};
};

template <>
struct declaredDtype<double> {
  constexpr static const ncclDataType_t ncclDtype{ncclFloat64};
  constexpr static const cudaDataType cudaDtype{CUDA_R_64F};
};

template <typename T>
struct SharedMemory {};

template <>
struct SharedMemory<float> {
  static __device__ float *get() {
    extern __shared__ float floatPtr[];
    return floatPtr;
  }
};

template <>
struct SharedMemory<double> {
  static __device__ double *get() {
    extern __shared__ double doublePtr[];
    return doublePtr;
  }
};

template <>
struct SharedMemory<int> {
  static __device__ int *get() {
    extern __shared__ int intPtr[];
    return intPtr;
  }
};

template <typename T>
struct sincosG {};

template <>
struct sincosG<float> {
  constexpr static void (*const call)(float, float *, float *) = sincosf;
};

template <>
struct sincosG<double> {
  constexpr static void (*const call)(double, double *, double *) = sincos;
};

template <typename T>
struct sqrtG {};

template <>
struct sqrtG<float> {
  constexpr static float (*const call)(float) = sqrtf;
};

template <>
struct sqrtG<double> {
  constexpr static double (*const call)(double) = sqrt;
};

template <typename T>
struct rsqrtG {};

template <>
struct rsqrtG<float> {
  constexpr static float (*const call)(float) = rsqrtf;
};

template <>
struct rsqrtG<double> {
  constexpr static double (*const call)(double) = rsqrt;
};  // namespace
}  // namespace Wrapper
}  // namespace MegBA
#undef CUXXX_WRAPPER
