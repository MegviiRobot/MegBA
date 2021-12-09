/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "operator/JetVector.h"
#include "operator/Thrust_Transform.h"
#include "operator/math_function_Jet_Vector_CUDA.cuh"

namespace MegBA {
    namespace math {
        namespace function {
            inline void fit_grid_and_block(unsigned int nElm, dim3& gridDim, dim3& blockDim){
                if (nElm < 256) {
                    blockDim = dim3(nElm);
                    gridDim = dim3(1);
                } else {
                    blockDim = dim3(256);
                    gridDim = dim3((nElm - 1) / blockDim.x + 1);
                }
            };

            template <typename T>
            __global__ void JetVector_add_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                    const T* f_res, const T* f_grad,
                                                                    const T* g_res, const T* g_grad,
                                                                    T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = g_grad[grid_thread_rank + i * nElm] + f_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void Jet_PVector_add_JetVector_Kernel(const unsigned int nElm,
                                                              const T* f_res, const int f_grad_position,
                                                              const T* g_res,
                                                              T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;

                out_grad[grid_thread_rank + f_grad_position * nElm] += 1;

                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void Jet_PVector_add_Jet_PVector_Kernel(const unsigned int nElm,
                                                               const T* f_res, const int f_grad_position,
                                                               const T* g_res, const int g_grad_position,
                                                               T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;

                out_grad[grid_thread_rank + f_grad_position * nElm] = 1;
                out_grad[grid_thread_rank + g_grad_position * nElm] += 1;

                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
            }

            template<typename T>
            void JetVector_add_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                                const MegBA::JetVector<T> &g,
                                                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        if (g.get_Grad_Position() != -1) {
                            // f is JPV, g is JPV
                            cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, f.getGradShape() * nElm * sizeof(T));
                            Jet_PVector_add_Jet_PVector_Kernel<T><<<gridDim, blockDim>>>(
                                    nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            // f is JPV, g is not JPV
                            cudaMemcpyAsync(out.get_CUDA_Grad_ptr()[i], g.get_CUDA_Grad_ptr()[i], out.getGradShape() * nElm * sizeof(T), cudaMemcpyDeviceToDevice);
                            Jet_PVector_add_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    } else {
                        // f is not JPV, g is JPV
                        if (g.get_Grad_Position() != -1) {
                            cudaMemcpyAsync(out.get_CUDA_Grad_ptr()[i], f.get_CUDA_Grad_ptr()[i], out.getGradShape() * nElm * sizeof(T), cudaMemcpyDeviceToDevice);
                            Jet_PVector_add_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    nElm,
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    f.get_CUDA_Res_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            JetVector_add_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    }
                }
            }

            template <typename T>
            __global__ void
            Jet_PVector_add_Scalar_Vector_Kernel(const unsigned int nElm,
                                                const T *f_res, const int f_grad_position,
                                                const T *g_res,
                                                T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_grad[grid_thread_rank + f_grad_position * nElm] = 1;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void
            JetVector_add_Scalar_Vector_Kernel(const unsigned int nElm,
                                                const T *f_res,
                                                const T *g_res,
                                                T *out_res) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
            }
            template<typename T>
            void JetVector_add_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                   const MegBA::JetVector<T> &g,
                                              MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    const auto nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        // f is JPV
                        cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                        Jet_PVector_add_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    } else {
                        // f is not JPV
                        cudaMemcpyAsync(out.get_CUDA_Grad_ptr()[i],
                                        f.get_CUDA_Grad_ptr()[i],
                                        f.getGradShape() * nElm * sizeof(T),
                                        cudaMemcpyDeviceToDevice);
                        JetVector_add_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i],
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i]);
                    }
                }
            }
            
            template <typename T>
            __global__ void
            Scalar_Vector_add_Scalar_Vector_Kernel(const unsigned int nElm,
                                                   const T *f_res,
                                                   const T *g_res,
                                                   T *out_res) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
            }
            template<typename T>
            void Scalar_Vector_add_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                   const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    const auto nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    Scalar_Vector_add_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                            nElm,
                            f.get_CUDA_Res_ptr()[i],
                            g.get_CUDA_Res_ptr()[i],
                            out.get_CUDA_Res_ptr()[i]);
                }
            }
            
            template<typename T>
            void Vector_add_Vector_CUDA(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> &out) {
                if (f.getGradShape() != 0)
                    if (g.getGradShape() != 0)
                        JetVector_add_JetVector_CUDA(f, g, out);
                    else
                        JetVector_add_Scalar_Vector_CUDA(f, g, out);
                else
                    if (g.getGradShape() != 0)
                        JetVector_add_Scalar_Vector_CUDA(g, f, out);
                    else
                        Scalar_Vector_add_Scalar_Vector_CUDA(f, g, out);
            }
            template void Vector_add_Vector_CUDA<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                                           MegBA::JetVector<double> &out);

            template void Vector_add_Vector_CUDA<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                                          MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void Jet_PVector_minus_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                const T* f_res, const int f_grad_position,
                                                                const T* g_res, const T* g_grad,
                                                                T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = - g_grad[grid_thread_rank + i * nElm];

                out_grad[grid_thread_rank + f_grad_position * nElm] += 1;

                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void Jet_PVector_minus_Jet_PVector_Kernel(const unsigned int nElm,
                                                                 const T* f_res, const int f_grad_position,
                                                                 const T* g_res, const int g_grad_position,
                                                                 T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_grad[grid_thread_rank + f_grad_position * nElm] = 1;
                out_grad[grid_thread_rank + g_grad_position * nElm] -= 1;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void JetVector_minus_Jet_PVector_Kernel(const unsigned int nElm,
                                                                const T* f_res,
                                                                const T* g_res, const int g_grad_position,
                                                                T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_grad[grid_thread_rank + g_grad_position * nElm] -= 1;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void JetVector_minus_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                               const T* f_res, const T* f_grad,
                                                               const T* g_res, const T* g_grad,
                                                               T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm] - g_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }

            template<typename T>
            void JetVector_minus_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                                           const MegBA::JetVector<T> &g,
                                                  MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        if (g.get_Grad_Position() != -1) {
                            // f is JPV, g is JPV
                            cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                            Jet_PVector_minus_Jet_PVector_Kernel<T><<< gridDim, blockDim >>>(
                                    nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            // f is JPV, g is not JPV
                            Jet_PVector_minus_JetVector_Kernel<T><<< gridDim, blockDim >>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    } else  {
                        if (g.get_Grad_Position() != -1) {
                            // f is not JPV, g is JPV
                            cudaMemcpyAsync(out.get_CUDA_Grad_ptr()[i],
                                            f.get_CUDA_Grad_ptr()[i],
                                            f.getGradShape() * nElm * sizeof(T),
                                            cudaMemcpyDeviceToDevice);
                            JetVector_minus_Jet_PVector_Kernel<T><<< gridDim, blockDim >>>(
                                    nElm,
                                    f.get_CUDA_Res_ptr()[i],
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            // f is not JPV, g is not JPV
                            JetVector_minus_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    }
                }
            }

            template <typename T>
            __global__ void
            Jet_PVector_minus_Scalar_Vector_Kernel(const unsigned int nElm,
                                                  const T *f_res, const int f_grad_position,
                                                  const T *g_res,
                                                  T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_grad[grid_thread_rank + f_grad_position * nElm] = 1;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void
            JetVector_minus_Scalar_Vector_Kernel(const unsigned int nElm,
                                                  const T *f_res,
                                                  const T *g_res,
                                                  T *out_res) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }
            template<typename T>
            void JetVector_minus_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                  const MegBA::JetVector<T> &g,
                                                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                        Jet_PVector_minus_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);

                    } else {
                        cudaMemcpyAsync(out.get_CUDA_Grad_ptr()[i],
                                        f.get_CUDA_Grad_ptr()[i],
                                        out.getGradShape() * nElm * sizeof(T),
                                        cudaMemcpyDeviceToDevice);
                        JetVector_minus_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i],
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i]);
                    }
                }
            }

            template <typename T>
            __global__ void
            Scalar_Vector_minus_PJetVector_Kernel(const unsigned int nElm,
                                                   const T *f_res,
                                                   const T *g_res, const int f_grad_position,
                                                   T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_grad[grid_thread_rank + f_grad_position * nElm] = -1;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }

            template <typename T>
            __global__ void
            Scalar_Vector_minus_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                  const T *f_res,
                                                  const T *g_res, const T *g_grad,
                                                  T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = -g_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }
            template<typename T>
            void Scalar_Vector_minus_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                                  const MegBA::JetVector<T> &g,
                                                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (g.get_Grad_Position() != -1) {
                        cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                        Scalar_Vector_minus_PJetVector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i],
                                g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    } else {
                        Scalar_Vector_minus_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                out.getGradShape(), nElm,
                                f.get_CUDA_Res_ptr()[i],
                                g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    }
                }
            }
            
            template <typename T>
            __global__ void
            Scalar_Vector_minus_Scalar_Vector_Kernel(const unsigned int nElm, 
                                                     const T *f_res,
                                                     const T *g_res,
                                                     T *out_res) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
            }
            template<typename T>
            void Scalar_Vector_minus_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                  const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    Scalar_Vector_minus_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                            nElm,
                            f.get_CUDA_Res_ptr()[i],
                            g.get_CUDA_Res_ptr()[i],
                            out.get_CUDA_Res_ptr()[i]);
                }
            }
            
            template<typename T>
            void Vector_minus_Vector_CUDA(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> &out) {
                if (f.getGradShape() != 0)
                    if (g.getGradShape() != 0)
                        JetVector_minus_JetVector_CUDA(f, g, out);
                    else
                        JetVector_minus_Scalar_Vector_CUDA(f, g, out);
                else
                    if (g.getGradShape() != 0)
                        Scalar_Vector_minus_JetVector_CUDA(f, g, out);
                    else
                        Scalar_Vector_minus_Scalar_Vector_CUDA(f, g, out);
            }
            template void Vector_minus_Vector_CUDA<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                                             MegBA::JetVector<double> &out);

            template void Vector_minus_Vector_CUDA<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                                            MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void JetVector_multiplies_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                    const T* f_res, const T* f_grad,
                                                                    const T* g_res, const T* g_grad,
                                                                    T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_res_local * g_grad[grid_thread_rank + i * nElm] + g_res_local * f_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res_local * g_res_local;
            }

            template <typename T>
            __global__ void Jet_PVector_multiplies_Jet_PVector_Kernel(const unsigned int nElm,
                                                                      const T* f_res, const int f_grad_position,
                                                                      const T* g_res, const int g_grad_position,
                                                                      T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                out_grad[grid_thread_rank + f_grad_position * nElm] = g_res_local;
                out_grad[grid_thread_rank + g_grad_position * nElm] += f_res_local;

                out_res[grid_thread_rank] = f_res_local * g_res_local;
            }

            template <typename T>
            __global__ void Jet_PVector_multiplies_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                     const T* f_res, const int f_grad_position,
                                                                     const T* g_res, const T* g_grad,
                                                                     T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_res_local * g_grad[grid_thread_rank + i * nElm];
                out_grad[grid_thread_rank + f_grad_position * nElm] += g_res_local;
                out_res[grid_thread_rank] = f_res_local * g_res_local;
            }

            template<typename T>
            void JetVector_multiplies_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                                                const MegBA::JetVector<T> &g,
                                                  MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        if (g.get_Grad_Position() != -1) {
                            cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                            Jet_PVector_multiplies_Jet_PVector_Kernel<T><<<gridDim, blockDim>>>(
                                    nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            Jet_PVector_multiplies_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    } else {
                        if (g.get_Grad_Position() != -1) {
                            Jet_PVector_multiplies_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            JetVector_multiplies_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    }
                }
            }

            template <typename T>
            __global__ void
            Jet_PVector_multiplies_Scalar_Vector_Kernel(const unsigned int nElm,
                                                        const T *f_res, const int f_grad_position,
                                                        const T *g_res,
                                                        T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                out_grad[grid_thread_rank + f_grad_position * nElm] = g_res_local;
                out_res[grid_thread_rank] = f_res_local * g_res_local;
            }

            template <typename T>
            __global__ void
            JetVector_multiplies_Scalar_Vector_Kernel(const unsigned int N, const unsigned int nElm,
                                                       const T *f_res, const T *f_grad,
                                                       const T *g_res,
                                                       T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = g_res_local * f_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res_local * g_res_local;
            }

            template<typename T>
            void JetVector_multiplies_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                          const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                        Jet_PVector_multiplies_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    } else {
                        JetVector_multiplies_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                out.getGradShape(), nElm,
                                f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    }
                }
            }
            
            template <typename T>
            __global__ void
            Scalar_Vector_multiplies_Scalar_Vector_Kernel(const unsigned int nElm,
                                                          const T *f_res,
                                                          const T *g_res,
                                                          T *out_res) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] * g_res[grid_thread_rank];
            }
            template<typename T>
            void Scalar_Vector_multiplies_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                       const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    Scalar_Vector_multiplies_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                            nElm,
                            f.get_CUDA_Res_ptr()[i],
                            g.get_CUDA_Res_ptr()[i],
                            out.get_CUDA_Res_ptr()[i]);
                }
            }
            
            template<typename T>
            void Vector_multiplies_Vector_CUDA(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                               MegBA::JetVector<T> &out) {
                if (f.getGradShape() != 0)
                    if (g.getGradShape() != 0)
                        JetVector_multiplies_JetVector_CUDA(f, g, out);
                    else
                        JetVector_multiplies_Scalar_Vector_CUDA(f, g, out);
                else
                    if (g.getGradShape() != 0)
                        JetVector_multiplies_Scalar_Vector_CUDA(g, f, out);
                    else
                        Scalar_Vector_multiplies_Scalar_Vector_CUDA(f, g, out);
            }
            template void Vector_multiplies_Vector_CUDA<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Vector_multiplies_Vector_CUDA<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void Jet_PVector_divides_Jet_PVector_Kernel(const unsigned int nElm,
                                                                   const T* f_res, const int f_grad_position,
                                                                   const T* g_res, const int g_grad_position,
                                                                   T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                T g_res_inv_local = T(1) / g_res_local;
                T f_res_div_g_res_local = f_res_local * g_res_inv_local;
                bool same_position = f_grad_position == g_grad_position;
                out_grad[grid_thread_rank + f_grad_position * nElm] = (1 - f_res_div_g_res_local * same_position) * g_res_inv_local;
                out_grad[grid_thread_rank + g_grad_position * nElm] += (same_position - f_res_div_g_res_local) * g_res_inv_local;
                out_res[grid_thread_rank] = f_res_div_g_res_local;
            }

            template <typename T>
            __global__ void Jet_PVector_divides_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                  const T* f_res, const int f_grad_position,
                                                                  const T* g_res, const T* g_grad,
                                                                  T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                T g_res_inv_local = T(1) / g_res_local;
                T f_res_div_g_res_local = f_res_local * g_res_inv_local;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = - f_res_div_g_res_local * g_grad[grid_thread_rank + i * nElm] * g_res_inv_local;
                out_grad[grid_thread_rank + f_grad_position * nElm] += g_res_inv_local;
                out_res[grid_thread_rank] = f_res_div_g_res_local;
            }

            template <typename T>
            __global__ void JetVector_divides_Jet_PVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                  const T* f_res, const T* f_grad,
                                                                  const T* g_res, const int g_grad_position,
                                                                  T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                T g_res_inv_local = T(1) / g_res_local;
                T f_res_div_g_res_local = f_res_local * g_res_inv_local;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm] * g_res_inv_local;
                out_grad[grid_thread_rank + g_grad_position * nElm] += - f_res_div_g_res_local * g_res_inv_local;
                out_res[grid_thread_rank] = f_res_div_g_res_local;
            }

            template <typename T>
            __global__ void JetVector_divides_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                    const T* f_res, const T* f_grad,
                                                                    const T* g_res, const T* g_grad,
                                                                    T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                T g_res_inv_local = T(1) / g_res_local;
                T f_res_div_g_res_local = f_res_local * g_res_inv_local;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = (f_grad[grid_thread_rank + i * nElm] - f_res_div_g_res_local * g_grad[grid_thread_rank + i * nElm]) * g_res_inv_local;
                out_res[grid_thread_rank] = f_res_div_g_res_local;
            }
            template<typename T>
            void JetVector_divides_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                                    const MegBA::JetVector<T> &g,
                                               MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != -1) {
                        if (g.get_Grad_Position() != -1) {
                            cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                            Jet_PVector_divides_Jet_PVector_Kernel<T><<<gridDim, blockDim>>>(
                                    nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            Jet_PVector_divides_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    } else {
                        if (g.get_Grad_Position() != -1) {
                            JetVector_divides_Jet_PVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                    g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        } else {
                            JetVector_divides_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                    out.getGradShape(), nElm,
                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                    g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                        }
                    }
                }
            }

            template <typename T>
            __global__ void
            Jet_PVector_divides_Scalar_Vector_Kernel(const unsigned int nElm,
                                                     const T *f_res, const int f_grad_position,
                                                     const T *g_res,
                                                     T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T g_res_inv_local = T(1) / g_res[grid_thread_rank];
                out_grad[grid_thread_rank + f_grad_position * nElm] = g_res_inv_local;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] * g_res_inv_local;
            }

            template <typename T>
            __global__ void
            JetVector_divides_Scalar_Vector_Kernel(const unsigned int N, const unsigned int nElm,
                                                    const T *f_res, const T *f_grad,
                                                    const T *g_res,
                                                    T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T g_res_inv_local = T(1) / g_res[grid_thread_rank];
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm] * g_res_inv_local;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] * g_res_inv_local;
            }

            template<typename T>
            void JetVector_divides_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                       const MegBA::JetVector<T> &g,
                                                  MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (f.get_Grad_Position() != 0) {
                        cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                        Jet_PVector_divides_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i], f.get_Grad_Position(),
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    } else {
                        JetVector_divides_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                                out.getGradShape(), nElm,
                                f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                g.get_CUDA_Res_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    }
                }
            }
            
            template <typename T>
            __global__ void
            Scalar_Vector_divides_Scalar_Vector_Kernel(const unsigned int nElm,
                                                       const T *f_res,
                                                       const T *g_res,
                                                       T *out_res) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] / g_res[grid_thread_rank];
            }
            template<typename T>
            void Scalar_Vector_divides_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                    const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    Scalar_Vector_divides_Scalar_Vector_Kernel<T><<<gridDim, blockDim>>>(
                            nElm,
                            f.get_CUDA_Res_ptr()[i],
                            g.get_CUDA_Res_ptr()[i],
                            out.get_CUDA_Res_ptr()[i]);
                }
            }

            template <typename T>
            __global__ void
            Scalar_Vector_divides_Jet_PVector_Kernel(const unsigned int nElm,
                                                     const T *f_res,
                                                     const T *g_res, const int g_grad_position,
                                                     T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                T g_res_inv_local = T(1) / g_res_local;
                T f_res_div_g_res_local = f_res_local * g_res_inv_local;
                out_grad[grid_thread_rank + g_grad_position * nElm] = - f_res_div_g_res_local * g_res_inv_local;
                out_res[grid_thread_rank] = f_res_div_g_res_local;
            }

            template <typename T>
            __global__ void
            Scalar_Vector_divides_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                    const T *f_res,
                                                    const T *g_res, const T *g_grad,
                                                    T *out_res, T *out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                T g_res_local = g_res[grid_thread_rank];
                T g_res_inv_local = T(1) / g_res_local;
                T f_res_div_g_res_local = f_res_local * g_res_inv_local;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = - f_res_div_g_res_local * g_grad[grid_thread_rank + i * nElm] * g_res_inv_local;
                out_res[grid_thread_rank] = f_res_div_g_res_local;
            }

            template<typename T>
            void Scalar_Vector_divides_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                                       const MegBA::JetVector<T> &g,
                                                  MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    if (g.get_Grad_Position() != 0) {
                        cudaMemsetAsync(out.get_CUDA_Grad_ptr()[i], 0, out.getGradShape() * nElm * sizeof(T));
                        Scalar_Vector_divides_Jet_PVector_Kernel<T><<<gridDim, blockDim>>>(
                                nElm,
                                f.get_CUDA_Res_ptr()[i],
                                g.get_CUDA_Res_ptr()[i], g.get_Grad_Position(),
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    } else {
                        Scalar_Vector_divides_JetVector_Kernel<T><<<gridDim, blockDim>>>(
                                out.getGradShape(), nElm,
                                f.get_CUDA_Res_ptr()[i],
                                g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                    }
                }
            }

            template<typename T>
            void Vector_divides_Vector_CUDA(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                            MegBA::JetVector<T> &out) {
                if (f.getGradShape() != 0)
                    if (g.getGradShape() != 0)
                        JetVector_divides_JetVector_CUDA(f, g, out);
                    else
                        JetVector_divides_Scalar_Vector_CUDA(f, g, out);
                else
                    if (g.getGradShape() != 0)
                        Scalar_Vector_divides_JetVector_CUDA(f, g, out);
                    else
                        Scalar_Vector_divides_Scalar_Vector_CUDA(f, g, out);
            }
            template void Vector_divides_Vector_CUDA<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Vector_divides_Vector_CUDA<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                                              MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void JetVector_add_Scalar_Kernel(const unsigned int N, const unsigned int nElm,
                                                                 const T* f_res, const T* f_grad,
                                                                 const T g,
                                                                 T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res[grid_thread_rank] + g;
            }

            template<typename T>
            void JetVector_add_Scalar_CUDA(const MegBA::JetVector<T> &f, T g,
                                            MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    JetVector_add_Scalar_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                           f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                           g,
                                                                           out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void JetVector_add_Scalar_CUDA<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_add_Scalar_CUDA<float>(
                    const MegBA::JetVector<float> &f, float g,
                                              MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void JetVector_minus_Scalar_Kernel(const unsigned int N, const unsigned int nElm,
                                                         const T* f_res, const T* f_grad,
                                                         const T g,
                                                         T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f_res[grid_thread_rank] - g;
            }
            template<typename T>
            void JetVector_minus_Scalar_CUDA(const MegBA::JetVector<T> &f, T g, MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    JetVector_minus_Scalar_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                             f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                             g,
                                                                             out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void JetVector_minus_Scalar_CUDA<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_minus_Scalar_CUDA<float>(
                    const MegBA::JetVector<float> &f, float g,
                MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void JetVector_multiplies_Scalar_Kernel(const unsigned int N, const unsigned int nElm,
                                                           const T* f_res, const T* f_grad,
                                                           const T g,
                                                           T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm] * g;
                out_res[grid_thread_rank] = f_res[grid_thread_rank] * g;
            }
            template<typename T>
            void JetVector_multiplies_Scalar_CUDA(const MegBA::JetVector<T> &f, T g, MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    JetVector_multiplies_Scalar_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                                  f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                                  g,
                                                                                  out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void JetVector_multiplies_Scalar_CUDA<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_multiplies_Scalar_CUDA<float>(
                    const MegBA::JetVector<float> &f, float g,
                MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void Scalar_minus_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                                const T f,
                                                                const T* g_res, const T* g_grad,
                                                                T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = -g_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = f - g_res[grid_thread_rank];
            }
            template<typename T>
            void Scalar_minus_JetVector_CUDA(T f, const JetVector<T> &g,
                                              JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 blockDim;
                    dim3 gridDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    Scalar_minus_JetVector_Kernel<T><<<gridDim, blockDim>>>(g.getGradShape(), nElm,
                                                                             f,
                                                                             g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                                                             out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void Scalar_minus_JetVector_CUDA<double>(
                    double f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Scalar_minus_JetVector_CUDA<float>(
                    float f, const MegBA::JetVector<float> &g,
                MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void Scalar_divides_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                             const T f,
                                                             const T* g_res, const T* g_grad,
                                                             T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T g_res_inv_local = T(1) / g_res[grid_thread_rank];
                T g_res_inv_times_f_local = f * g_res_inv_local;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = -g_grad[grid_thread_rank + i * nElm] * g_res_inv_local * g_res_inv_times_f_local;
                out_res[grid_thread_rank] = g_res_inv_times_f_local;
            }
            template<typename T>
            void Scalar_divides_JetVector_CUDA(T f, const JetVector<T> &g,
                                                JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 blockDim;
                    dim3 gridDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    Scalar_divides_JetVector_Kernel<T><<<gridDim, blockDim>>>(g.getGradShape(), nElm,
                                                                               f,
                                                                               g.get_CUDA_Res_ptr()[i], g.get_CUDA_Grad_ptr()[i],
                                                                               out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void Scalar_divides_JetVector_CUDA<double>(
                    double f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Scalar_divides_JetVector_CUDA<float>(
                    float f, const MegBA::JetVector<float> &g,
                MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void abs_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                  const T* f_res, const T* f_grad,
                                                  T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                int mask_local = (int)(f_res_local > 0) * 2 - 1;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = mask_local * f_grad[grid_thread_rank + i * nElm];
                out_res[grid_thread_rank] = mask_local * f_res_local;
            }
            template<typename T>
            void abs_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    abs_JetVector_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void abs_JetVector_CUDA<double>(
                    const MegBA::JetVector<double> &f,
                                        MegBA::JetVector<double> &out);

            template void abs_JetVector_CUDA<float>(
                    const MegBA::JetVector<float> &f,
                                       MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void cos_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                  const T* f_res, const T* f_grad,
                                                  T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = -f_grad[grid_thread_rank + i * nElm] * std::sin(f_res_local);
                out_res[grid_thread_rank] = std::cos(f_res_local);
            }
            template<typename T>
            void cos_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    cos_JetVector_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void cos_JetVector_CUDA<double>(
                    const MegBA::JetVector<double> &f,
                                        MegBA::JetVector<double> &out);

            template void cos_JetVector_CUDA<float>(
                    const MegBA::JetVector<float> &f,
                                       MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void sin_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                  const T* f_res, const T* f_grad,
                                                  T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_local = f_res[grid_thread_rank];
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm] * std::cos(f_res_local);
                out_res[grid_thread_rank] = std::sin(f_res_local);
            }
            template<typename T>
            void sin_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    sin_JetVector_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                    f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                    out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void sin_JetVector_CUDA<double>(
                    const MegBA::JetVector<double> &f,
                                        MegBA::JetVector<double> &out);

            template void sin_JetVector_CUDA<float>(
                    const MegBA::JetVector<float> &f,
                                       MegBA::JetVector<float> &out);

            template <typename T>
            __global__ void sqrt_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                                   const T* f_res, const T* f_grad,
                                                   T* out_res, T* out_grad) {
                /*
                 * 1D block and grid
                 */
                unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
                if (grid_thread_rank >= nElm)
                    return;
                T f_res_sqrt_local = std::sqrt(f_res[grid_thread_rank]);
                T f_res_sqrt_half_inv_local = T(0.5) / f_res_sqrt_local;
                for (unsigned int i = 0; i < N; ++i)
                    out_grad[grid_thread_rank + i * nElm] = f_grad[grid_thread_rank + i * nElm] * f_res_sqrt_half_inv_local;
                out_res[grid_thread_rank] = f_res_sqrt_local;
            }
            template<typename T>
            void sqrt_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                      MegBA::JetVector<T> &out) {
                for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                    cudaSetDevice(i);
                    unsigned int nElm = out.get_Elm_Num(i);
                    dim3 gridDim;
                    dim3 blockDim;
                    fit_grid_and_block(nElm, gridDim, blockDim);
                    sqrt_JetVector_Kernel<T><<<gridDim, blockDim>>>(f.getGradShape(), nElm,
                                                                     f.get_CUDA_Res_ptr()[i], f.get_CUDA_Grad_ptr()[i],
                                                                     out.get_CUDA_Res_ptr()[i], out.get_CUDA_Grad_ptr()[i]);
                }
            }
            template void sqrt_JetVector_CUDA<double>(
                    const MegBA::JetVector<double> &f,
                                         MegBA::JetVector<double> &out);

            template void sqrt_JetVector_CUDA<float>(
                    const MegBA::JetVector<float> &f,
                                        MegBA::JetVector<float> &out);

        }  // namespace function
    }  // namespace math
}  // namespace MegBA