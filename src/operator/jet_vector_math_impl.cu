/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <array>

#include "operator/jet_vector.h"
#include "operator/jet_vector_math_impl.cuh"

namespace MegBA {
namespace math {
namespace impl {
inline std::array<dim3, 2> fitGridAndBlock(const unsigned int nItem) {
  std::array<dim3, 2> gridAndDim;
  if (nItem < 256) {
    gridAndDim[1] = dim3(nItem);
    gridAndDim[0] = dim3(1);
  } else {
    gridAndDim[1] = dim3(256);
    gridAndDim[0] = dim3((nItem - 1) / gridAndDim[1].x + 1);
  }
  return gridAndDim;
}

template <typename T>
__global__ void JetVector_add_JetVector_Kernel(const unsigned int N,
                                               const unsigned int nItem,
                                               const T *f_res, const T *f_grad,
                                               const T *g_res, const T *g_grad,
                                               T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        g_grad[tid + i * nItem] + f_grad[tid + i * nItem];
  out_res[tid] = f_res[tid] + g_res[tid];
}

template <typename T>
__global__ void Jet_PVector_add_JetVector_Kernel(const unsigned int nItem,
                                                 const T *f_res,
                                                 const int f_grad_position,
                                                 const T *g_res, T *out_res,
                                                 T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;

  out_grad[tid + f_grad_position * nItem] += 1;

  out_res[tid] = f_res[tid] + g_res[tid];
}

template <typename T>
__global__ void Jet_PVector_add_Jet_PVector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;

  out_grad[tid + f_grad_position * nItem] = 1;
  out_grad[tid + g_grad_position * nItem] += 1;

  out_res[tid] = f_res[tid] + g_res[tid];
}

template <typename T>
void JetVector_add_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                  const MegBA::JetVector<T> &g,
                                  MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        // f is JPV, g is JPV
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        f.getGradShape() * nItem * sizeof(T));
        Jet_PVector_add_Jet_PVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
            g.getCUDAResPtr()[i], g.getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        // f is JPV, g is not JPV
        cudaMemcpyAsync(out->getCUDAGradPtr()[i], g.getCUDAGradPtr()[i],
                        out->getGradShape() * nItem * sizeof(T),
                        cudaMemcpyDeviceToDevice);
        Jet_PVector_add_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
            g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      }
    } else {
      // f is not JPV, g is JPV
      if (g.getGradPosition() != -1) {
        cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                        out->getGradShape() * nItem * sizeof(T),
                        cudaMemcpyDeviceToDevice);
        Jet_PVector_add_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nItem, g.getCUDAResPtr()[i], g.getGradPosition(),
            f.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        JetVector_add_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nItem, f.getCUDAResPtr()[i],
            f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_add_Scalar_Vector_Kernel(const unsigned int nItem,
                                                     const T *f_res,
                                                     const int f_grad_position,
                                                     const T *g_res, T *out_res,
                                                     T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_grad[tid + f_grad_position * nItem] = 1;
  out_res[tid] = f_res[tid] + g_res[tid];
}

template <typename T>
__global__ void JetVector_add_Scalar_Vector_Kernel(const unsigned int nItem,
                                                   const T *f_res,
                                                   const T *g_res, T *out_res) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_res[tid] = f_res[tid] + g_res[tid];
}
template <typename T>
void JetVector_add_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                      const MegBA::JetVector<T> &g,
                                      MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      // f is JPV
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nItem * sizeof(T));
      Jet_PVector_add_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
          g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      // f is not JPV
      cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                      f.getGradShape() * nItem * sizeof(T),
                      cudaMemcpyDeviceToDevice);
      JetVector_add_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          nItem, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
          out->getCUDAResPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_add_Scalar_Vector_Kernel(const unsigned int nItem,
                                                       const T *f_res,
                                                       const T *g_res,
                                                       T *out_res) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_res[tid] = f_res[tid] + g_res[tid];
}
template <typename T>
void Scalar_Vector_add_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                          const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    Scalar_Vector_add_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        nItem, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
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
__global__ void Jet_PVector_minus_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const int f_grad_position, const T *g_res, const T *g_grad, T *out_res,
    T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = -g_grad[tid + i * nItem];

  out_grad[tid + f_grad_position * nItem] += 1;

  out_res[tid] = f_res[tid] - g_res[tid];
}

template <typename T>
__global__ void Jet_PVector_minus_Jet_PVector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_grad[tid + f_grad_position * nItem] = 1;
  out_grad[tid + g_grad_position * nItem] -= 1;
  out_res[tid] = f_res[tid] - g_res[tid];
}

template <typename T>
__global__ void JetVector_minus_Jet_PVector_Kernel(const unsigned int nItem,
                                                   const T *f_res,
                                                   const T *g_res,
                                                   const int g_grad_position,
                                                   T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_grad[tid + g_grad_position * nItem] -= 1;
  out_res[tid] = f_res[tid] - g_res[tid];
}

template <typename T>
__global__ void JetVector_minus_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *f_grad, const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        f_grad[tid + i * nItem] - g_grad[tid + i * nItem];
  out_res[tid] = f_res[tid] - g_res[tid];
}

template <typename T>
void JetVector_minus_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                    const MegBA::JetVector<T> &g,
                                    MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        // f is JPV, g is JPV
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        out->getGradShape() * nItem * sizeof(T));
        Jet_PVector_minus_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
                g.getCUDAResPtr()[i], g.getGradPosition(),
                out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        // f is JPV, g is not JPV
        Jet_PVector_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nItem, f.getCUDAResPtr()[i],
            f.getGradPosition(), g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    } else {
      if (g.getGradPosition() != -1) {
        // f is not JPV, g is JPV
        cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                        f.getGradShape() * nItem * sizeof(T),
                        cudaMemcpyDeviceToDevice);
        JetVector_minus_Jet_PVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            nItem, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
            g.getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        // f is not JPV, g is not JPV
        JetVector_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nItem, f.getCUDAResPtr()[i],
            f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_minus_Scalar_Vector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_grad[tid + f_grad_position * nItem] = 1;
  out_res[tid] = f_res[tid] - g_res[tid];
}

template <typename T>
__global__ void JetVector_minus_Scalar_Vector_Kernel(const unsigned int nItem,
                                                     const T *f_res,
                                                     const T *g_res,
                                                     T *out_res) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_res[tid] = f_res[tid] - g_res[tid];
}
template <typename T>
void JetVector_minus_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                        const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nItem * sizeof(T));
      Jet_PVector_minus_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
              g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);

    } else {
      cudaMemcpyAsync(out->getCUDAGradPtr()[i], f.getCUDAGradPtr()[i],
                      out->getGradShape() * nItem * sizeof(T),
                      cudaMemcpyDeviceToDevice);
      JetVector_minus_Scalar_Vector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          nItem, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
          out->getCUDAResPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_minus_PJetVector_Kernel(const unsigned int nItem,
                                                      const T *f_res,
                                                      const T *g_res,
                                                      const int f_grad_position,
                                                      T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_grad[tid + f_grad_position * nItem] = -1;
  out_res[tid] = f_res[tid] - g_res[tid];
}

template <typename T>
__global__ void Scalar_Vector_minus_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = -g_grad[tid + i * nItem];
  out_res[tid] = f_res[tid] - g_res[tid];
}
template <typename T>
void Scalar_Vector_minus_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                        const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (g.getGradPosition() != -1) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nItem * sizeof(T));
      Scalar_Vector_minus_PJetVector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nItem, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
              g.getGradPosition(), out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      Scalar_Vector_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
          out->getGradShape(), nItem, f.getCUDAResPtr()[i],
          g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_minus_Scalar_Vector_Kernel(
    const unsigned int nItem, const T *f_res, const T *g_res, T *out_res) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_res[tid] = f_res[tid] - g_res[tid];
}
template <typename T>
void Scalar_Vector_minus_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                            const MegBA::JetVector<T> &g,
                                            MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    Scalar_Vector_minus_Scalar_Vector_Kernel<T>
        <<<gridAndDim[0], gridAndDim[1]>>>(nItem, f.getCUDAResPtr()[i],
                                           g.getCUDAResPtr()[i],
                                           out->getCUDAResPtr()[i]);
  }
}

template <typename T>
void vectorSubVectorCUDA(const MegBA::JetVector<T> &f,
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
template void vectorSubVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                          const MegBA::JetVector<double> &g,
                                          MegBA::JetVector<double> *out);

template void vectorSubVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                         const MegBA::JetVector<float> &g,
                                         MegBA::JetVector<float> *out);

template <typename T>
__global__ void JetVector_multiplies_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *f_grad, const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_res_local * g_grad[tid + i * nItem] +
                                g_res_local * f_grad[tid + i * nItem];
  out_res[tid] = f_res_local * g_res_local;
}

template <typename T>
__global__ void Jet_PVector_multiplies_Jet_PVector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  out_grad[tid + f_grad_position * nItem] = g_res_local;
  out_grad[tid + g_grad_position * nItem] += f_res_local;

  out_res[tid] = f_res_local * g_res_local;
}

template <typename T>
__global__ void Jet_PVector_multiplies_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const int f_grad_position, const T *g_res, const T *g_grad, T *out_res,
    T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_res_local * g_grad[tid + i * nItem];
  out_grad[tid + f_grad_position * nItem] += g_res_local;
  out_res[tid] = f_res_local * g_res_local;
}

template <typename T>
void JetVector_multiplies_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                         const MegBA::JetVector<T> &g,
                                         MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        out->getGradShape() * nItem * sizeof(T));
        Jet_PVector_multiplies_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
                g.getCUDAResPtr()[i], g.getGradPosition(),
                out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        Jet_PVector_multiplies_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nItem, f.getCUDAResPtr()[i],
                f.getGradPosition(), g.getCUDAResPtr()[i],
                g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      }
    } else {
      if (g.getGradPosition() != -1) {
        Jet_PVector_multiplies_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nItem, g.getCUDAResPtr()[i],
                g.getGradPosition(), f.getCUDAResPtr()[i],
                f.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      } else {
        JetVector_multiplies_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nItem, f.getCUDAResPtr()[i],
                f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
                g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_multiplies_Scalar_Vector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  out_grad[tid + f_grad_position * nItem] = g_res_local;
  out_res[tid] = f_res_local * g_res_local;
}

template <typename T>
__global__ void JetVector_multiplies_Scalar_Vector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *f_grad, const T *g_res, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = g_res_local * f_grad[tid + i * nItem];
  out_res[tid] = f_res_local * g_res_local;
}

template <typename T>
void JetVector_multiplies_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                             const MegBA::JetVector<T> &g,
                                             MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nItem * sizeof(T));
      Jet_PVector_multiplies_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
              g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      JetVector_multiplies_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              out->getGradShape(), nItem, f.getCUDAResPtr()[i],
              f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
              out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_multiplies_Scalar_Vector_Kernel(
    const unsigned int nItem, const T *f_res, const T *g_res, T *out_res) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_res[tid] = f_res[tid] * g_res[tid];
}
template <typename T>
void Scalar_Vector_multiplies_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                                 const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    Scalar_Vector_multiplies_Scalar_Vector_Kernel<T>
        <<<gridAndDim[0], gridAndDim[1]>>>(nItem, f.getCUDAResPtr()[i],
                                           g.getCUDAResPtr()[i],
                                           out->getCUDAResPtr()[i]);
  }
}

template <typename T>
void vectorMulVectorCUDA(const MegBA::JetVector<T> &f,
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
template void vectorMulVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                          const MegBA::JetVector<double> &g,
                                          MegBA::JetVector<double> *out);

template void vectorMulVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                         const MegBA::JetVector<float> &g,
                                         MegBA::JetVector<float> *out);

template <typename T>
__global__ void Jet_PVector_divides_Jet_PVector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, const int g_grad_position, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  T g_res_inv_local = T(1) / g_res_local;
  T f_res_div_g_res_local = f_res_local * g_res_inv_local;
  bool same_position = f_grad_position == g_grad_position;
  out_grad[tid + f_grad_position * nItem] =
      (1 - f_res_div_g_res_local * same_position) * g_res_inv_local;
  out_grad[tid + g_grad_position * nItem] +=
      (same_position - f_res_div_g_res_local) * g_res_inv_local;
  out_res[tid] = f_res_div_g_res_local;
}

template <typename T>
__global__ void Jet_PVector_divides_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const int f_grad_position, const T *g_res, const T *g_grad, T *out_res,
    T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  T g_res_inv_local = T(1) / g_res_local;
  T f_res_div_g_res_local = f_res_local * g_res_inv_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        -f_res_div_g_res_local * g_grad[tid + i * nItem] * g_res_inv_local;
  out_grad[tid + f_grad_position * nItem] += g_res_inv_local;
  out_res[tid] = f_res_div_g_res_local;
}

template <typename T>
__global__ void JetVector_divides_Jet_PVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *f_grad, const T *g_res, const int g_grad_position, T *out_res,
    T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  T g_res_inv_local = T(1) / g_res_local;
  T f_res_div_g_res_local = f_res_local * g_res_inv_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_grad[tid + i * nItem] * g_res_inv_local;
  out_grad[tid + g_grad_position * nItem] +=
      -f_res_div_g_res_local * g_res_inv_local;
  out_res[tid] = f_res_div_g_res_local;
}

template <typename T>
__global__ void JetVector_divides_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *f_grad, const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  T g_res_inv_local = T(1) / g_res_local;
  T f_res_div_g_res_local = f_res_local * g_res_inv_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        (f_grad[tid + i * nItem] -
         f_res_div_g_res_local * g_grad[tid + i * nItem]) *
        g_res_inv_local;
  out_res[tid] = f_res_div_g_res_local;
}
template <typename T>
void JetVector_divides_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                      const MegBA::JetVector<T> &g,
                                      MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != -1) {
      if (g.getGradPosition() != -1) {
        cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                        out->getGradShape() * nItem * sizeof(T));
        Jet_PVector_divides_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
                g.getCUDAResPtr()[i], g.getGradPosition(),
                out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        Jet_PVector_divides_JetVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nItem, f.getCUDAResPtr()[i],
                f.getGradPosition(), g.getCUDAResPtr()[i],
                g.getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      }
    } else {
      if (g.getGradPosition() != -1) {
        JetVector_divides_Jet_PVector_Kernel<T>
            <<<gridAndDim[0], gridAndDim[1]>>>(
                out->getGradShape(), nItem, f.getCUDAResPtr()[i],
                f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
                g.getGradPosition(), out->getCUDAResPtr()[i],
                out->getCUDAGradPtr()[i]);
      } else {
        JetVector_divides_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
            out->getGradShape(), nItem, f.getCUDAResPtr()[i],
            f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
__global__ void Jet_PVector_divides_Scalar_Vector_Kernel(
    const unsigned int nItem, const T *f_res, const int f_grad_position,
    const T *g_res, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T g_res_inv_local = T(1) / g_res[tid];
  out_grad[tid + f_grad_position * nItem] = g_res_inv_local;
  out_res[tid] = f_res[tid] * g_res_inv_local;
}

template <typename T>
__global__ void JetVector_divides_Scalar_Vector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *f_grad, const T *g_res, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T g_res_inv_local = T(1) / g_res[tid];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_grad[tid + i * nItem] * g_res_inv_local;
  out_res[tid] = f_res[tid] * g_res_inv_local;
}

template <typename T>
void JetVector_divides_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                          const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (f.getGradPosition() != 0) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nItem * sizeof(T));
      Jet_PVector_divides_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nItem, f.getCUDAResPtr()[i], f.getGradPosition(),
              g.getCUDAResPtr()[i], out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      JetVector_divides_Scalar_Vector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              out->getGradShape(), nItem, f.getCUDAResPtr()[i],
              f.getCUDAGradPtr()[i], g.getCUDAResPtr()[i],
              out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
__global__ void Scalar_Vector_divides_Scalar_Vector_Kernel(
    const unsigned int nItem, const T *f_res, const T *g_res, T *out_res) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  out_res[tid] = f_res[tid] / g_res[tid];
}
template <typename T>
void Scalar_Vector_divides_Scalar_Vector_CUDA(const MegBA::JetVector<T> &f,
                                              const MegBA::JetVector<T> &g,
                                              MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    Scalar_Vector_divides_Scalar_Vector_Kernel<T>
        <<<gridAndDim[0], gridAndDim[1]>>>(nItem, f.getCUDAResPtr()[i],
                                           g.getCUDAResPtr()[i],
                                           out->getCUDAResPtr()[i]);
  }
}

template <typename T>
__global__ void Scalar_Vector_divides_Jet_PVector_Kernel(
    const unsigned int nItem, const T *f_res, const T *g_res,
    const int g_grad_position, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  T g_res_inv_local = T(1) / g_res_local;
  T f_res_div_g_res_local = f_res_local * g_res_inv_local;
  out_grad[tid + g_grad_position * nItem] =
      -f_res_div_g_res_local * g_res_inv_local;
  out_res[tid] = f_res_div_g_res_local;
}

template <typename T>
__global__ void Scalar_Vector_divides_JetVector_Kernel(
    const unsigned int N, const unsigned int nItem, const T *f_res,
    const T *g_res, const T *g_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  T g_res_local = g_res[tid];
  T g_res_inv_local = T(1) / g_res_local;
  T f_res_div_g_res_local = f_res_local * g_res_inv_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        -f_res_div_g_res_local * g_grad[tid + i * nItem] * g_res_inv_local;
  out_res[tid] = f_res_div_g_res_local;
}

template <typename T>
void Scalar_Vector_divides_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                          const MegBA::JetVector<T> &g,
                                          MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    if (g.getGradPosition() != 0) {
      cudaMemsetAsync(out->getCUDAGradPtr()[i], 0,
                      out->getGradShape() * nItem * sizeof(T));
      Scalar_Vector_divides_Jet_PVector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              nItem, f.getCUDAResPtr()[i], g.getCUDAResPtr()[i],
              g.getGradPosition(), out->getCUDAResPtr()[i],
              out->getCUDAGradPtr()[i]);
    } else {
      Scalar_Vector_divides_JetVector_Kernel<T>
          <<<gridAndDim[0], gridAndDim[1]>>>(
              out->getGradShape(), nItem, f.getCUDAResPtr()[i],
              g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
              out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
    }
  }
}

template <typename T>
void vectorDivVectorCUDA(const MegBA::JetVector<T> &f,
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
template void vectorDivVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                          const MegBA::JetVector<double> &g,
                                          MegBA::JetVector<double> *out);

template void vectorDivVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                         const MegBA::JetVector<float> &g,
                                         MegBA::JetVector<float> *out);

template <typename T>
__global__ void JetVector_add_Scalar_Kernel(const unsigned int N,
                                            const unsigned int nItem,
                                            const T *f_res, const T *f_grad,
                                            const T g, T *out_res,
                                            T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_grad[tid + i * nItem];
  out_res[tid] = f_res[tid] + g;
}

template <typename T>
void jetVectorAddScalarCUDA(const MegBA::JetVector<T> &f, T g,
                            MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    JetVector_add_Scalar_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i], g,
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
__global__ void JetVector_minus_Scalar_Kernel(const unsigned int N,
                                              const unsigned int nItem,
                                              const T *f_res, const T *f_grad,
                                              const T g, T *out_res,
                                              T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_grad[tid + i * nItem];
  out_res[tid] = f_res[tid] - g;
}
template <typename T>
void jetVectorSubScalarCUDA(const MegBA::JetVector<T> &f, T g,
                            MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    JetVector_minus_Scalar_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i], g,
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void jetVectorSubScalarCUDA<double>(const MegBA::JetVector<double> &f,
                                             double g,
                                             MegBA::JetVector<double> *out);

template void jetVectorSubScalarCUDA<float>(const MegBA::JetVector<float> &f,
                                            float g,
                                            MegBA::JetVector<float> *out);

template <typename T>
__global__ void JetVector_multiplies_Scalar_Kernel(const unsigned int N,
                                                   const unsigned int nItem,
                                                   const T *f_res,
                                                   const T *f_grad, const T g,
                                                   T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_grad[tid + i * nItem] * g;
  out_res[tid] = f_res[tid] * g;
}
template <typename T>
void jetVectorMulScalarCUDA(const MegBA::JetVector<T> &f, T g,
                            MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    JetVector_multiplies_Scalar_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i], g,
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void jetVectorMulScalarCUDA<double>(const MegBA::JetVector<double> &f,
                                             double g,
                                             MegBA::JetVector<double> *out);

template void jetVectorMulScalarCUDA<float>(const MegBA::JetVector<float> &f,
                                            float g,
                                            MegBA::JetVector<float> *out);

template <typename T>
__global__ void Scalar_minus_JetVector_Kernel(const unsigned int N,
                                              const unsigned int nItem,
                                              const T f, const T *g_res,
                                              const T *g_grad, T *out_res,
                                              T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = -g_grad[tid + i * nItem];
  out_res[tid] = f - g_res[tid];
}
template <typename T>
void scalarSubJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    Scalar_minus_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        g.getGradShape(), nItem, f, g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void scalarSubJetVectorCUDA<double>(double f,
                                             const MegBA::JetVector<double> &g,
                                             MegBA::JetVector<double> *out);

template void scalarSubJetVectorCUDA<float>(float f,
                                            const MegBA::JetVector<float> &g,
                                            MegBA::JetVector<float> *out);

template <typename T>
__global__ void Scalar_divides_JetVector_Kernel(const unsigned int N,
                                                const unsigned int nItem,
                                                const T f, const T *g_res,
                                                const T *g_grad, T *out_res,
                                                T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T g_res_inv_local = T(1) / g_res[tid];
  T g_res_inv_times_f_local = f * g_res_inv_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        -g_grad[tid + i * nItem] * g_res_inv_local * g_res_inv_times_f_local;
  out_res[tid] = g_res_inv_times_f_local;
}
template <typename T>
void scalarDivJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    Scalar_divides_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        g.getGradShape(), nItem, f, g.getCUDAResPtr()[i], g.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void scalarDivJetVectorCUDA<double>(double f,
                                             const MegBA::JetVector<double> &g,
                                             MegBA::JetVector<double> *out);

template void scalarDivJetVectorCUDA<float>(float f,
                                            const MegBA::JetVector<float> &g,
                                            MegBA::JetVector<float> *out);

template <typename T>
__global__ void abs_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nItem, const T *f_res,
                                     const T *f_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  int mask_local = static_cast<int>(f_res_local > 0) * 2 - 1;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = mask_local * f_grad[tid + i * nItem];
  out_res[tid] = mask_local * f_res_local;
}
template <typename T>
void absJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    abs_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void absJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> *out);

template void absJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> *out);

template <typename T>
__global__ void cos_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nItem, const T *f_res,
                                     const T *f_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        -f_grad[tid + i * nItem] * std::sin(f_res_local);
  out_res[tid] = std::cos(f_res_local);
}
template <typename T>
void cosJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    cos_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void cosJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> *out);

template void cosJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> *out);

template <typename T>
__global__ void sin_JetVector_Kernel(const unsigned int N,
                                     const unsigned int nItem, const T *f_res,
                                     const T *f_grad, T *out_res, T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_local = f_res[tid];
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] = f_grad[tid + i * nItem] * std::cos(f_res_local);
  out_res[tid] = std::sin(f_res_local);
}
template <typename T>
void sinJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    sin_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
        out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
  }
}
template void sinJetVectorCUDA<double>(const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> *out);

template void sinJetVectorCUDA<float>(const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> *out);

template <typename T>
__global__ void sqrt_JetVector_Kernel(const unsigned int N,
                                      const unsigned int nItem, const T *f_res,
                                      const T *f_grad, T *out_res,
                                      T *out_grad) {
  /*
   * 1D block and grid
   */
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f_res_sqrt_local = std::sqrt(f_res[tid]);
  T f_res_sqrt_half_inv_local = T(0.5) / f_res_sqrt_local;
  for (unsigned int i = 0; i < N; ++i)
    out_grad[tid + i * nItem] =
        f_grad[tid + i * nItem] * f_res_sqrt_half_inv_local;
  out_res[tid] = f_res_sqrt_local;
}
template <typename T>
void sqrtJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    unsigned int nItem = out->getItemNum(i);

    std::array<dim3, 2> gridAndDim = fitGridAndBlock(nItem);
    sqrt_JetVector_Kernel<T><<<gridAndDim[0], gridAndDim[1]>>>(
        f.getGradShape(), nItem, f.getCUDAResPtr()[i], f.getCUDAGradPtr()[i],
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
