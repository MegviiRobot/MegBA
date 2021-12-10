/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/
#include "operator/jet_vector_math_impl.cuh"
#include <array>
#include "operator/jet_vector.h"

namespace MegBA {
namespace math {
namespace impl {
inline std::array<dim3, 2> fitGridAndBlock(const unsigned int nElm) {
  std::array<dim3, 2> gridAndDim;
  if (nElm < 256) {
    gridAndDim[1] = dim3(nElm);
    gridAndDim[0] = dim3(1);
  } else {
    gridAndDim[1] = dim3(256);
    gridAndDim[0] = dim3((nElm - 1) / gridAndDim[1].x + 1);
  }
  return gridAndDim;
}

template <typename T>
__global__ void
JetVector_add_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                               const T *f_res, const T *f_grad, const T *g_res,
                               const T *g_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        g_grad[grid_thread_rank + i * nElm] +
        f_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
}

template <typename T>
__global__ void
Jet_PVector_add_JetVector_Kernel(const unsigned int nElm, const T *f_res,
                                 const int f_grad_position, const T *g_res,
                                 T *out_res, T *out_grad) {
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
__global__ void Jet_PVector_add_Jet_PVector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
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

template <typename T>
void JetVector_add_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                  const MegBA::JetVector<T> &g,
                                  MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        // f is JPV, g is JPV
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        f.getGradShape() * nElm * sizeof(T));
        Jet_PVector_add_Jet_PVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
            g.getCUDAResPtr()[i], g.getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        // f is JPV, g is not JPV
        cudaMemcpyAsync(out->getCUDAGradPtr()[i], g.getCUDAGradPtr()[i],
                        out->getGradShape() * nElm * sizeof(T),
                        cudaMemcpyDeviceToDevice);
        Jet_PVector_add_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
            g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      }
    } else {
      // f is not JPV, g is JPV
      if (g.getGradPosition() != -1) {
        cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                        out->getGradShape() * nElm * sizeof(T),
                        cudaMemcpyDeviceToDevice);
        Jet_PVector_add_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nElm, g.getCUDAResPtr()[i], g.getGradPosition(),
            f.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        JetVector_add_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nElm, f.getCUDAResPtr()[i],
            f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void
Jet_PVector_add_Scalar_Vector_Kernel(const unsigned int nElm, const T *f_res,
                                     const int f_grad_position, const T *g_res,
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
__global__ void JetVector_add_Scalar_Vector_Kernel(const unsigned int nElm,
                                                   const T *f_res,
                                                   const T *g_res, T *out_res) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
}
template <typename T>
void JetVector_add_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                      const MegBA::JetVector<T> &g,
                                      MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      // f is JPV
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nElm * sizeof(T));
      Jet_PVector_add_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          nElm, f.getCUDAResPtr()[i], f.getGradPosition(), g.getCUDAResPtr()[i],
          out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    } else {
      // f is not JPV
      cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                      f.getGradShape() * nElm * sizeof(T),
                      cudaMemcpyDeviceToDevice);
      JetVector_add_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
          out->getCUDAResPtr()[i]);
    }
  }
}

template <typename T>
__global__ void
Scalar_Vector_add_Scalar_Vector_Kernel(const unsigned int nElm, const T *f_res,
                                       const T *g_res, T *out_res) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] + g_res[grid_thread_rank];
}
template <typename T>
void Scalar_Vector_add_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                          const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    Scalar_Vector_add_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
        out->getCUDAResPtr()[i]);
  }
}

template <typename T>
void vectorAddVectorCUDA(const MegBA::JetVector<T> &f,
                         const MegBA::JetVector<T> &g,
                         MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      JetVector_add_JetVector_CUDA(f, g, out);
    } else {
      JetVector_add_Scalar_Vector_CUDA(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      JetVector_add_Scalar_Vector_CUDA(g, f, out);
    } else {
      Scalar_Vector_add_Scalar_Vector_CUDA(f, g, out);
    }
  }
}
template void vectorAddVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                          const MegBA::JetVector<double> &g,
                                          MegBA::JetVector<double> *out);

template void vectorAddVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                         const MegBA::JetVector<float> &g,
                                         MegBA::JetVector<float> *out);

template <typename T>
__global__ void
Jet_PVector_minus_JetVector_Kernel(const unsigned int N,
                                   const unsigned int nElm, const T *f_res,
                                   const int f_grad_position, const T *g_res,
                                   const T *g_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        -g_grad[grid_thread_rank + i * nElm];

  out_grad[grid_thread_rank + f_grad_position * nElm] += 1;

  out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
}

template <typename T>
__global__ void Jet_PVector_minus_Jet_PVector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
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
__global__ void
JetVector_minus_Jet_PVector_Kernel(const unsigned int nElm, const T *f_res,
                                   const T *g_res, const int g_grad_position,
                                   T *out_res, T *out_grad) {
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
__global__ void JetVector_minus_JetVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *f_grad, const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_grad[grid_thread_rank + i * nElm] -
        g_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
}

template <typename T>
void JetVector_minus_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                    const MegBA::JetVector<T> &g,
                                    MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        // f is JPV, g is JPV
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        out->getGradShape() * nElm * sizeof(T));
        Jet_PVector_minus_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
                g.getCUDAResPtr()[i], g.getGradPosition(),
                out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        // f is JPV, g is not JPV
        Jet_PVector_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nElm, f.getCUDAResPtr()[i],
            f.getGradPosition(), g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    } else {
      if (g.getGradPosition() != -1) {
        // f is not JPV, g is JPV
        cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                        f.getGradShape() * nElm * sizeof(T),
                        cudaMemcpyDeviceToDevice);
        JetVector_minus_Jet_PVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
            g.getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        // f is not JPV, g is not JPV
        JetVector_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nElm, f.getCUDAResPtr()[i],
            f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_minus_Scalar_Vector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, T *out_res, T *out_grad) {
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
JetVector_minus_Scalar_Vector_Kernel(const unsigned int nElm, const T *f_res,
                                     const T *g_res, T *out_res) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
}
template <typename T>
void JetVector_minus_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                        const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nElm * sizeof(T));
      Jet_PVector_minus_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
              g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);

    } else {
      cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                      out->getGradShape() * nElm * sizeof(T),
                      cudaMemcpyDeviceToDevice);
      JetVector_minus_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
          out->getCUDAResPtr()[i]);
    }
  }
}

template <typename T>
__global__ void
Scalar_Vector_minus_PJetVector_Kernel(const unsigned int nElm, const T *f_res,
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
__global__ void Scalar_Vector_minus_JetVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        -g_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
}
template <typename T>
void Scalar_Vector_minus_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                        const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (g.getGradPosition() != -1) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nElm * sizeof(T));
      Scalar_Vector_minus_PJetVector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
              g.getGradPosition(), out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      Scalar_Vector_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          out->getGradShape(), nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
          g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_minus_Scalar_Vector_Kernel(
    const unsigned int nElm, const T *f_res, const T *g_res, T *out_res) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] - g_res[grid_thread_rank];
}
template <typename T>
void Scalar_Vector_minus_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                            const MegBA::JetVector<T> &g,
                                            MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    Scalar_Vector_minus_Scalar_Vector_Kernel<T>
        <<<gridAndDim[0], gridAndDim[1]>>>(nElm, f.getCUDAResPtr()[i],
                                           g.getCUDAResPtr()[i],
                                           out->getCUDAResPtr()[i]);
  }
}

template <typename T>
void vectorMinusVectorCUDA(const MegBA::JetVector<T> &f,
                           const MegBA::JetVector<T> &g,
                           MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      JetVector_minus_JetVector_CUDA(f, g, out);
    } else {
      JetVector_minus_Scalar_Vector_CUDA(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      Scalar_Vector_minus_JetVector_CUDA(f, g, out);
    } else {
      Scalar_Vector_minus_Scalar_Vector_CUDA(f, g, out);
    }
  }
}
template void vectorMinusVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                            const MegBA::JetVector<double> &g,
                                            MegBA::JetVector<double> *out);

template void vectorMinusVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                           const MegBA::JetVector<float> &g,
                                           MegBA::JetVector<float> *out);

template <typename T>
__global__ void JetVector_multiplies_JetVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *f_grad, const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_local = f_res[grid_thread_rank];
  T g_res_local = g_res[grid_thread_rank];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_res_local * g_grad[grid_thread_rank + i * nElm] +
        g_res_local * f_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = f_res_local * g_res_local;
}

template <typename T>
__global__ void Jet_PVector_multiplies_Jet_PVector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
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
__global__ void Jet_PVector_multiplies_JetVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const int f_grad_position, const T *g_res, const T *g_grad, T *out_res,
    T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_local = f_res[grid_thread_rank];
  T g_res_local = g_res[grid_thread_rank];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_res_local * g_grad[grid_thread_rank + i * nElm];
  out_grad[grid_thread_rank + f_grad_position * nElm] += g_res_local;
  out_res[grid_thread_rank] = f_res_local * g_res_local;
}

template <typename T>
void JetVector_multiplies_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                         const MegBA::JetVector<T> &g,
                                         MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        out->getGradShape() * nElm * sizeof(T));
        Jet_PVector_multiplies_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
                g.getCUDAResPtr()[i], g.getGradPosition(),
                out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        Jet_PVector_multiplies_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nElm, f.getCUDAResPtr()[i],
                f.getGradPosition(), g.getCUDAResPtr()[i],
                g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      }
    } else {
      if (g.getGradPosition() != -1) {
        Jet_PVector_multiplies_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nElm, g.getCUDAResPtr()[i],
                g.getGradPosition(), f.getCUDAResPtr()[i],
                f.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      } else {
        JetVector_multiplies_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nElm, f.getCUDAResPtr()[i],
                f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
                g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_multiplies_Scalar_Vector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, T *out_res, T *out_grad) {
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
__global__ void JetVector_multiplies_Scalar_Vector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *f_grad, const T *g_res, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_local = f_res[grid_thread_rank];
  T g_res_local = g_res[grid_thread_rank];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        g_res_local * f_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = f_res_local * g_res_local;
}

template <typename T>
void JetVector_multiplies_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                             const MegBA::JetVector<T> &g,
                                             MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nElm * sizeof(T));
      Jet_PVector_multiplies_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
              g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      JetVector_multiplies_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              out->getGradShape(), nElm, f.getCUDAResPtr()[i],
              f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
              out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_multiplies_Scalar_Vector_Kernel(
    const unsigned int nElm, const T *f_res, const T *g_res, T *out_res) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] * g_res[grid_thread_rank];
}
template <typename T>
void Scalar_Vector_multiplies_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                 const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    Scalar_Vector_multiplies_Scalar_Vector_Kernel<T>
        <<<gridAndDim[0], gridAndDim[1]>>>(nElm, f.getCUDAResPtr()[i],
                                           g.getCUDAResPtr()[i],
                                           out->getCUDAResPtr()[i]);
  }
}

template <typename T>
void vectorMultipliesVectorCUDA(const MegBA::JetVector<T> &f,
                                const MegBA::JetVector<T> &g,
                                MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      JetVector_multiplies_JetVector_CUDA(f, g, out);
    } else {
      JetVector_multiplies_Scalar_Vector_CUDA(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      JetVector_multiplies_Scalar_Vector_CUDA(g, f, out);
    } else {
      Scalar_Vector_multiplies_Scalar_Vector_CUDA(f, g, out);
    }
  }
}
template void
vectorMultipliesVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                   const MegBA::JetVector<double> &g,
                                   MegBA::JetVector<double> *out);

template void
vectorMultipliesVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                  const MegBA::JetVector<float> &g,
                                  MegBA::JetVector<float> *out);

template <typename T>
__global__ void Jet_PVector_divides_Jet_PVector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
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
  out_grad[grid_thread_rank + f_grad_position * nElm] =
      (1 - f_res_div_g_res_local * same_position) * g_res_inv_local;
  out_grad[grid_thread_rank + g_grad_position * nElm] +=
      (same_position - f_res_div_g_res_local) * g_res_inv_local;
  out_res[grid_thread_rank] = f_res_div_g_res_local;
}

template <typename T>
__global__ void
Jet_PVector_divides_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nElm, const T *f_res,
                                     const int f_grad_position, const T *g_res,
                                     const T *g_grad, T *out_res, T *out_grad) {
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
    out_grad[grid_thread_rank + i * nElm] =
        -f_res_div_g_res_local * g_grad[grid_thread_rank + i * nElm] *
        g_res_inv_local;
  out_grad[grid_thread_rank + f_grad_position * nElm] += g_res_inv_local;
  out_res[grid_thread_rank] = f_res_div_g_res_local;
}

template <typename T>
__global__ void JetVector_divides_Jet_PVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *f_grad, const T *g_res, const int g_grad_position, T *out_res,
    T *out_grad) {
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
    out_grad[grid_thread_rank + i * nElm] =
        f_grad[grid_thread_rank + i * nElm] * g_res_inv_local;
  out_grad[grid_thread_rank + g_grad_position * nElm] +=
      -f_res_div_g_res_local * g_res_inv_local;
  out_res[grid_thread_rank] = f_res_div_g_res_local;
}

template <typename T>
__global__ void JetVector_divides_JetVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *f_grad, const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
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
    out_grad[grid_thread_rank + i * nElm] =
        (f_grad[grid_thread_rank + i * nElm] -
         f_res_div_g_res_local * g_grad[grid_thread_rank + i * nElm]) *
        g_res_inv_local;
  out_res[grid_thread_rank] = f_res_div_g_res_local;
}
template <typename T>
void JetVector_divides_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                      const MegBA::JetVector<T> &g,
                                      MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        out->getGradShape() * nElm * sizeof(T));
        Jet_PVector_divides_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
                g.getCUDAResPtr()[i], g.getGradPosition(),
                out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        Jet_PVector_divides_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nElm, f.getCUDAResPtr()[i],
                f.getGradPosition(), g.getCUDAResPtr()[i],
                g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      }
    } else {
      if (g.getGradPosition() != -1) {
        JetVector_divides_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nElm, f.getCUDAResPtr()[i],
                f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
                g.getGradPosition(), out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      } else {
        JetVector_divides_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nElm, f.getCUDAResPtr()[i],
            f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_divides_Scalar_Vector_Kernel(
    const unsigned int nElm, const T *f_res, const int f_grad_position,
    const T *g_res, T *out_res, T *out_grad) {
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
__global__ void JetVector_divides_Scalar_Vector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *f_grad, const T *g_res, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T g_res_inv_local = T(1) / g_res[grid_thread_rank];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_grad[grid_thread_rank + i * nElm] * g_res_inv_local;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] * g_res_inv_local;
}

template <typename T>
void JetVector_divides_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                          const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (f.getGradPosition() != 0) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nElm * sizeof(T));
      Jet_PVector_divides_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nElm, f.getCUDAResPtr()[i], f.getGradPosition(),
              g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      JetVector_divides_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              out->getGradShape(), nElm, f.getCUDAResPtr()[i],
              f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
              out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_divides_Scalar_Vector_Kernel(
    const unsigned int nElm, const T *f_res, const T *g_res, T *out_res) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] / g_res[grid_thread_rank];
}
template <typename T>
void Scalar_Vector_divides_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                              const MegBA::JetVector<T> &g,
                                              MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    Scalar_Vector_divides_Scalar_Vector_Kernel<T>
        <<<gridAndDim[0], gridAndDim[1]>>>(nElm, f.getCUDAResPtr()[i],
                                           g.getCUDAResPtr()[i],
                                           out->getCUDAResPtr()[i]);
  }
}

template <typename T>
__global__ void Scalar_Vector_divides_Jet_PVector_Kernel(
    const unsigned int nElm, const T *f_res, const T *g_res,
    const int g_grad_position, T *out_res, T *out_grad) {
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
  out_grad[grid_thread_rank + g_grad_position * nElm] =
      -f_res_div_g_res_local * g_res_inv_local;
  out_res[grid_thread_rank] = f_res_div_g_res_local;
}

template <typename T>
__global__ void Scalar_Vector_divides_JetVector_Kernel(
    const unsigned int N, const unsigned int nElm, const T *f_res,
    const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
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
    out_grad[grid_thread_rank + i * nElm] =
        -f_res_div_g_res_local * g_grad[grid_thread_rank + i * nElm] *
        g_res_inv_local;
  out_res[grid_thread_rank] = f_res_div_g_res_local;
}

template <typename T>
void Scalar_Vector_divides_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                          const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    if (g.getGradPosition() != 0) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nElm * sizeof(T));
      Scalar_Vector_divides_Jet_PVector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nElm, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
              g.getGradPosition(), out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      Scalar_Vector_divides_JetVector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              out->getGradShape(), nElm, f.getCUDAResPtr()[i],
              g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
              out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
void vectorDividesVectorCUDA(const MegBA::JetVector<T> &f,
                             const MegBA::JetVector<T> &g,
                             MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      JetVector_divides_JetVector_CUDA(f, g, out);
    } else {
      JetVector_divides_Scalar_Vector_CUDA(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      Scalar_Vector_divides_JetVector_CUDA(f, g, out);
    } else {
      Scalar_Vector_divides_Scalar_Vector_CUDA(f, g, out);
    }
  }
}
template void vectorDividesVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                              const MegBA::JetVector<double> &g,
                                              MegBA::JetVector<double> *out);

template void vectorDividesVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                             const MegBA::JetVector<float> &g,
                                             MegBA::JetVector<float> *out);

template <typename T>
__global__ void
JetVector_add_Scalar_Kernel(const unsigned int N, const unsigned int nElm,
                            const T *f_res, const T *f_grad, const T g,
                            T *out_res, T *out_grad) {
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

template <typename T>
void jetVectorAddScalarCUDA(const MegBA::JetVector<T> &f, T g,
                            MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    JetVector_add_Scalar_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i], g,
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void jetVectorAddScalarCUDA<double>(const MegBA::JetVector<double> &f,
                                             double g,
                                             MegBA::JetVector<double> *out);

template void jetVectorAddScalarCUDA<float>(const MegBA::JetVector<float> &f,
                                            float g,
                                            MegBA::JetVector<float> *out);

template <typename T>
__global__ void
JetVector_minus_Scalar_Kernel(const unsigned int N, const unsigned int nElm,
                              const T *f_res, const T *f_grad, const T g,
                              T *out_res, T *out_grad) {
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
template <typename T>
void jetVectorMinusScalarCUDA(const MegBA::JetVector<T> &f, T g,
                              MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    JetVector_minus_Scalar_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i], g,
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void
jetVectorMinusScalarCUDA<double>(const MegBA::JetVector<double> &f, double g,
                                 MegBA::JetVector<double> *out);

template void jetVectorMinusScalarCUDA<float>(const MegBA::JetVector<float> &f,
                                              float g,
                                              MegBA::JetVector<float> *out);

template <typename T>
__global__ void JetVector_multiplies_Scalar_Kernel(const unsigned int N,
                                                   const unsigned int nElm,
                                                   const T *f_res,
                                                   const T *f_grad, const T g,
                                                   T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_grad[grid_thread_rank + i * nElm] * g;
  out_res[grid_thread_rank] = f_res[grid_thread_rank] * g;
}
template <typename T>
void jetVectorMultipliesScalarCUDA(const MegBA::JetVector<T> &f, T g,
                                   MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    JetVector_multiplies_Scalar_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i], g,
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void
jetVectorMultipliesScalarCUDA<double>(const MegBA::JetVector<double> &f,
                                      double g, MegBA::JetVector<double> *out);

template void
jetVectorMultipliesScalarCUDA<float>(const MegBA::JetVector<float> &f, float g,
                                     MegBA::JetVector<float> *out);

template <typename T>
__global__ void
Scalar_minus_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                              const T f, const T *g_res, const T *g_grad,
                              T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        -g_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = f - g_res[grid_thread_rank];
}
template <typename T>
void scalarMinusJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);
    dim3 blockDim;
    dim3 gridDim;
    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    Scalar_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        g.getGradShape(), nElm, f, g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void
scalarMinusJetVectorCUDA<double>(double f, const MegBA::JetVector<double> &g,
                                 MegBA::JetVector<double> *out);

template void scalarMinusJetVectorCUDA<float>(float f,
                                              const MegBA::JetVector<float> &g,
                                              MegBA::JetVector<float> *out);

template <typename T>
__global__ void
Scalar_divides_JetVector_Kernel(const unsigned int N, const unsigned int nElm,
                                const T f, const T *g_res, const T *g_grad,
                                T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T g_res_inv_local = T(1) / g_res[grid_thread_rank];
  T g_res_inv_times_f_local = f * g_res_inv_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        -g_grad[grid_thread_rank + i * nElm] * g_res_inv_local *
        g_res_inv_times_f_local;
  out_res[grid_thread_rank] = g_res_inv_times_f_local;
}
template <typename T>
void scalarDividesJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);
    dim3 blockDim;
    dim3 gridDim;
    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    Scalar_divides_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        g.getGradShape(), nElm, f, g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void
scalarDividesJetVectorCUDA<double>(double f, const MegBA::JetVector<double> &g,
                                   MegBA::JetVector<double> *out);

template void
scalarDividesJetVectorCUDA<float>(float f, const MegBA::JetVector<float> &g,
                                  MegBA::JetVector<float> *out);

template <typename T>
__global__ void abs_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nElm, const T *f_res,
                                     const T *f_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_local = f_res[grid_thread_rank];
  int mask_local = static_cast<int>(f_res_local > 0) * 2 - 1;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        mask_local * f_grad[grid_thread_rank + i * nElm];
  out_res[grid_thread_rank] = mask_local * f_res_local;
}
template <typename T>
void absJetVectorCUDA(const MegBA::JetVector<T> &f,
                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    abs_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void absJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                         MegBA::JetVector<double> *out);

template void absJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                        MegBA::JetVector<float> *out);

template <typename T>
__global__ void cos_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nElm, const T *f_res,
                                     const T *f_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_local = f_res[grid_thread_rank];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        -f_grad[grid_thread_rank + i * nElm] * std::sin(f_res_local);
  out_res[grid_thread_rank] = std::cos(f_res_local);
}
template <typename T>
void cosJetVectorCUDA(const MegBA::JetVector<T> &f,
                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    cos_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void cosJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                         MegBA::JetVector<double> *out);

template void cosJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                        MegBA::JetVector<float> *out);

template <typename T>
__global__ void sin_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nElm, const T *f_res,
                                     const T *f_grad, T *out_res, T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_local = f_res[grid_thread_rank];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_grad[grid_thread_rank + i * nElm] * std::cos(f_res_local);
  out_res[grid_thread_rank] = std::sin(f_res_local);
}
template <typename T>
void sinJetVectorCUDA(const MegBA::JetVector<T> &f,
                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    sin_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void sinJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                         MegBA::JetVector<double> *out);

template void sinJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                        MegBA::JetVector<float> *out);

template <typename T>
__global__ void sqrt_JetVector_Kernel(const unsigned int N,
                                      const unsigned int nElm, const T *f_res,
                                      const T *f_grad, T *out_res,
                                      T *out_grad) {
  /*
                 * 1D block and grid
   */
  unsigned int grid_thread_rank = threadIdx.x + blockDim.x * blockIdx.x;
  if (grid_thread_rank >= nElm)
    return;
  T f_res_sqrt_local = std::sqrt(f_res[grid_thread_rank]);
  T f_res_sqrt_half_inv_local = T(0.5) / f_res_sqrt_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[grid_thread_rank + i * nElm] =
        f_grad[grid_thread_rank + i * nElm] * f_res_sqrt_half_inv_local;
  out_res[grid_thread_rank] = f_res_sqrt_local;
}
template <typename T>
void sqrtJetVectorCUDA(const MegBA::JetVector<T> &f,
                         MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nElm = out->getElmNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nElm);
    sqrt_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nElm, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void sqrtJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                          MegBA::JetVector<double> *out);

template void sqrtJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                         MegBA::JetVector<float> *out);

}  // namespace impl
}  // namespace math
}  // namespace MegBA
