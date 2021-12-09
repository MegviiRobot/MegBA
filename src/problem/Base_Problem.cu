/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/BaseProblem.h"
#include "Wrapper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include "operator/Thrust_Transform.h"
#include <resource/Manager.h>
#include <Macro.h>
#include <thrust/async/reduce.h>

namespace MegBA {
template <typename T> void BaseProblem<T>::cudaDeallocateResource() {
  if (option_.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaFree(schur_x_ptr[i]);
      cudaFree(schur_delta_x_ptr[i]);
      cudaFree(schur_delta_x_ptr_backup[i]);
    }
    schur_x_ptr.clear();
    schur_delta_x_ptr.clear();
    schur_delta_x_ptr_backup.clear();
  } else {
    // TODO: implement this
  }
}

template <typename T> void BaseProblem<T>::cudaPrepareUpdateData() {
  if (option_.useSchur) {
    const auto world_size = MemoryPool::getWorldSize();
    schur_x_ptr.resize(world_size);
    schur_delta_x_ptr.resize(world_size);
    schur_delta_x_ptr_backup.resize(world_size);
    for (int i = 0; i < world_size; ++i) {
      cudaSetDevice(i);
      cudaMalloc(&schur_x_ptr[i], Hessian_shape_ * sizeof(T));
      cudaMalloc(&schur_delta_x_ptr[i], Hessian_shape_ * sizeof(T));
      cudaMalloc(&schur_delta_x_ptr_backup[i], Hessian_shape_ * sizeof(T));
      cudaMemsetAsync(schur_delta_x_ptr[i], 0, Hessian_shape_ * sizeof(T));
    }
  } else {
    // TODO: implement this
  }
}

namespace {
template <typename T>
__global__ void H_add_ueye_CUDA(T *csrVal, const int *csrColInd,
                                const int *csrRowPtr, const T u,
                                const int Hessian_shape) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= Hessian_shape)
    return;
  int start_bias_this_row = csrRowPtr[idx];
  int nnz_in_this_row = csrRowPtr[idx + 1] - start_bias_this_row;
  for (int i = 0; i < nnz_in_this_row; ++i) {
    if (csrColInd[start_bias_this_row + i] == idx) {
      csrVal[start_bias_this_row + i] += u;
      return;
    }
  }
}

template <typename T> struct compare_abs_value {
  __host__ __device__ bool operator()(T lhs, T rhs) {
    return std::abs(lhs) < std::abs(rhs);
  }
};

template <typename T>
struct compare_abs_value_ret_max : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T lhs, T rhs) {
    return std::abs(lhs) < std::abs(rhs) ? std::abs(rhs) : std::abs(lhs);
  }
};

template <typename T> inline T L2Norm_pow2(const T *vector, const int size) {
  return thrust::inner_product(thrust::device_ptr<const T>(vector),
                               thrust::device_ptr<const T>{vector + size},
                               thrust::device_ptr<const T>(vector), T(0.));
}

template <typename T> inline T LinfNorm(const T *vector, const int size) {
  return std::abs(*thrust::max_element(
      thrust::device_ptr<const T>{vector},
      thrust::device_ptr<const T>{vector + size}, compare_abs_value<T>{}));
}

template <typename T>
inline auto LinfNorm_async(const T *vector, const int size) {
  return thrust::async::reduce(thrust::device_ptr<const T>{vector},
                               thrust::device_ptr<const T>{vector + size},
                               T(0.), compare_abs_value_ret_max<T>{});
}

namespace {
template <typename T>
__global__ void ExtractOldAndApplyNewDiagKernel(const T a, const int batchSize,
                                                T *csrVal, T *diags) {
  /*
   * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize)
    return;

  const T diag = csrVal[threadIdx.x + threadIdx.x * blockDim.x +
                        tid * blockDim.x * blockDim.x];
  diags[threadIdx.x + tid * blockDim.x] = diag;
  csrVal[threadIdx.x + threadIdx.x * blockDim.x +
         tid * blockDim.x * blockDim.x] = (a + 1) * diag;
}

template <typename T>
__global__ void RecoverDiagKernel(const T *in, const T a, const int batchSize,
                                  T *out) {
  /*
                 * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize)
    return;

  out[threadIdx.x + threadIdx.x * blockDim.x + tid * blockDim.x * blockDim.x] =
      (a + 1) * in[threadIdx.x + tid * blockDim.x];
}
}

template <typename T>
void ExtractOldAndApplyNewDiag(const T a, const int batchSize, const int dim,
                               T *csrVal, T *diag) {
  dim3 block(dim, std::min(decltype(batchSize)(32), batchSize));
  dim3 grid((batchSize - 1) / block.y + 1);
  ExtractOldAndApplyNewDiagKernel<<<grid, block>>>(a, batchSize, csrVal, diag);
}

template <typename T>
void RecoverDiag(const T *diag, const T a, const int batchSize, const int dim,
                 T *csrVal) {
  dim3 block(dim, std::min(decltype(batchSize)(32), batchSize));
  dim3 grid((batchSize - 1) / block.y + 1);
  RecoverDiagKernel<T><<<grid, block>>>(diag, a, batchSize, csrVal);
}

template <typename T>
__global__ void JdxpF(const T *grad, const T *delta_x, const T *res,
                      const int *abs_camera_position,
                      const int *abs_point_position, const int nElm,
                      const int camera_dim, const int camera_num,
                      const int point_dim, T *out) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nElm)
    return;
  T sum{0};
  const int abs_camera_position_local = abs_camera_position[tid];
  const int abs_point_position_local = abs_point_position[tid];
  for (int i = 0; i < camera_dim; ++i) {
    sum += grad[tid + i * nElm] *
           delta_x[i + abs_camera_position_local * camera_dim];
  }
  for (int i = 0; i < point_dim; ++i) {
    sum += grad[tid + (i + camera_dim) * nElm] *
           delta_x[i + camera_dim * camera_num +
                   abs_point_position_local * point_dim];
  }
  out[tid] = (sum + res[tid]) * (sum + res[tid]);
}
}
    template <typename T>
void BaseProblem<T>::SolveLM(int iter, double solver_tol,
                             double solver_refuse_ratio, int solver_max_iter,
                             const double tau, const double epsilon1,
                             const double epsilon2) {
  const auto &cublasHandle = HandleManager::get_cublasHandle();
  MakeVertices();
  Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic> JV_backup;
  int k = 0;
  T new_residual_norm = 0;
  T residual_norm = 0;

  edges.backupDaPtrs();
  edges.rebindDaPtrs();
  JV_backup = edges.forward();
  if (option_.useSchur) {
    edges.buildLinearSystemSchur(JV_backup);
  } else {
    // TODO: implement this
  }

  std::vector<std::vector<T>> new_residual_norm_in_flight;
  new_residual_norm_in_flight.resize(MemoryPool::getWorldSize());
  for (auto &vec : new_residual_norm_in_flight)
    vec.resize(JV_backup.size());
  for (int i = 0; i < JV_backup.rows(); ++i) {
    for (int j = 0; j < MemoryPool::getWorldSize(); ++j) {
      cudaSetDevice(j);
      const T *Res_ptr = JV_backup(i).get_CUDA_Res_ptr()[j];
      Wrapper::cublasGdot::call(cublasHandle[j], MemoryPool::getElmNum(j),
                                Res_ptr, 1, Res_ptr, 1,
                                &new_residual_norm_in_flight[j][i]);
    }
  }
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaStream_t stream;
    cudaSetDevice(i);
    cublasGetStream_v2(cublasHandle[i], &stream);
    cudaStreamSynchronize(stream);
    for (const auto new_residual_norm_landed : new_residual_norm_in_flight[i]) {
      new_residual_norm += new_residual_norm_landed;
    }
  }

  std::cout << "start with error: " << new_residual_norm / 2
            << ", log error: " << std::log10(new_residual_norm / 2)
            << std::endl;

  MemoryPool::redistribute();
  bool stop{false};
  T u = tau;
  T v = 2;
  T rho = 0;

  std::vector<std::array<T *, 2>> ExtractedDiag;
  ExtractedDiag.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    auto &container = edges.schurEquationContainer[i];
    cudaMalloc(&ExtractedDiag[i][0],
               container.nnz[2] / container.dim[0] * sizeof(T));
    cudaMalloc(&ExtractedDiag[i][1],
               container.nnz[3] / container.dim[1] * sizeof(T));
  }
  bool recover_diag{false};
  while (!stop && k < iter) {
    k++;
    if (option_.useSchur) {
      if (recover_diag) {
        for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
          cudaSetDevice(i);
          auto &container = edges.schurEquationContainer[i];
          ASSERT_CUDA_NO_ERROR();
          RecoverDiag(ExtractedDiag[i][0], T(1.) / u,
                      container.nnz[2] / container.dim[0] / container.dim[0],
                      container.dim[0], container.csrVal[2]);
          ASSERT_CUDA_NO_ERROR();
          RecoverDiag(ExtractedDiag[i][1], T(1.) / u,
                      container.nnz[3] / container.dim[1] / container.dim[1],
                      container.dim[1], container.csrVal[3]);
          ASSERT_CUDA_NO_ERROR();
        }
        recover_diag = false;
      } else {
        for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
          cudaSetDevice(i);
          auto &container = edges.schurEquationContainer[i];
          ExtractOldAndApplyNewDiag(
              T(1.) / u, container.nnz[2] / container.dim[0] / container.dim[0],
              container.dim[0], container.csrVal[2], ExtractedDiag[i][0]);
          ExtractOldAndApplyNewDiag(
              T(1.) / u, container.nnz[3] / container.dim[1] / container.dim[1],
              container.dim[1], container.csrVal[3], ExtractedDiag[i][1]);
        }
      }
    } else {
      // TODO: implement this
    }
    bool solver_success =
        SolveLinear(solver_tol, solver_refuse_ratio, solver_max_iter);
    MemoryPool::redistribute();
    ASSERT_CUDA_NO_ERROR();

    T delta_x_l2, x_l2;
    if (option_.useSchur) {
      cudaSetDevice(0);
      delta_x_l2 = L2Norm_pow2(schur_delta_x_ptr[0], Hessian_shape_);
      x_l2 = L2Norm_pow2(schur_x_ptr[0], Hessian_shape_);
    } else {
      // TODO: implement this
    }
    ASSERT_CUDA_NO_ERROR();

    delta_x_l2 = std::sqrt(delta_x_l2);
    x_l2 = std::sqrt(x_l2);
    if (delta_x_l2 <= epsilon2 * (x_l2 + epsilon1)) {
      std::cout << "Stopped for delta_x_l2{" << delta_x_l2 << "} <= epsilon2{"
                << epsilon2 << "} * (x_l2{" << x_l2 << "} + epsilon1{"
                << epsilon1 << "})" << std::endl;
      break;
    } else {
      if (option_.useSchur) {
        edges.updateSchur(schur_delta_x_ptr);
      } else {
        // TODO: implement this
      }

      T rho_Denominator{0};
      if (option_.useSchur) {
        std::vector<std::vector<T *>> Jdx;
        Jdx.resize(MemoryPool::getWorldSize());
        const int camera_dim = edges.schurEquationContainer[0].dim[0];
        const int camera_num =
            edges.schurEquationContainer[0].nnz[2] / camera_dim / camera_dim;
        const int point_dim = edges.schurEquationContainer[0].dim[1];

        std::vector<std::vector<thrust::system::cuda::unique_eager_future<T>>>
            futures;
        futures.resize(MemoryPool::getWorldSize());

        for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
          cudaSetDevice(i);
          const auto nElm = MemoryPool::getElmNum(i);
          const auto &eq_container = edges.schurEquationContainer[i];
          const auto &position_container =
              edges.schurPositionAndRelationContainer[i];
          futures[i].resize(JV_backup.size());
          for (int j = 0; j < JV_backup.size(); ++j) {
            auto &J = JV_backup(j);
            T *ptr;
            MemoryPool::allocateNormal((void **)&ptr, nElm * sizeof(T), i);
            dim3 block(std::min((std::size_t)256, nElm));
            dim3 grid((nElm - 1) / block.x + 1);
            JdxpF<<<grid, block>>>(
                J.get_CUDA_Grad_ptr()[i], schur_delta_x_ptr[i],
                J.get_CUDA_Res_ptr()[i],
                position_container.absolutePositionCamera,
                position_container.absolutePositionPoint, nElm, camera_dim,
                camera_num, point_dim, ptr);
            futures[i][j] = thrust::async::reduce(
                thrust::cuda::par.on(nullptr), thrust::device_ptr<T>{ptr},
                thrust::device_ptr<T>{ptr} + nElm, T(0.), thrust::plus<T>{});
            Jdx[i].push_back(ptr);
          }
        }
        for (int i = 0; i < futures.size(); ++i) {
          for (int j = futures[i].size() - 1; j >= 0; --j) {
            rho_Denominator += futures[i][j].get();
            MemoryPool::deallocateNormal((void *)Jdx[i][j], i);
          }
        }
        rho_Denominator -= new_residual_norm;
      } else {
        // TODO: implement this
      }
      ASSERT_CUDA_NO_ERROR();

      residual_norm = new_residual_norm;
      new_residual_norm = 0.;
      auto JV = edges.forward();
      for (int i = 0; i < JV.size(); ++i) {
        for (int j = 0; j < MemoryPool::getWorldSize(); ++j) {
          cudaSetDevice(j);
          const T *Res_ptr = JV(i).get_CUDA_Res_ptr()[j];
          Wrapper::cublasGdot::call(cublasHandle[j], MemoryPool::getElmNum(j),
                                    Res_ptr, 1, Res_ptr, 1,
                                    &new_residual_norm_in_flight[j][i]);
        }
      }
      for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
        cudaStream_t stream;
        cudaSetDevice(i);
        cublasGetStream_v2(cublasHandle[i], &stream);
        cudaStreamSynchronize(stream);
        for (const auto new_residual_norm_landed :
             new_residual_norm_in_flight[i]) {
          new_residual_norm += new_residual_norm_landed;
        }
      }
      ASSERT_CUDA_NO_ERROR();

      rho = -(residual_norm - new_residual_norm) / rho_Denominator;

      if (residual_norm > new_residual_norm) {
        for (int i = 0; i < JV.size(); ++i)
          JV_backup(i) = JV(i);
        if (option_.useSchur) {
          edges.buildLinearSystemSchur(JV);
        } else {
          // TODO: implement this
        }
        std::cout << k << "-th iter error: " << new_residual_norm / 2
                  << ", log error: " << std::log10(new_residual_norm / 2)
                  << std::endl;

        BackupLM();
        residual_norm = new_residual_norm;
        if (option_.useSchur) {
          cudaSetDevice(0);
          auto &container = edges.schurEquationContainer[0];
          const auto norm = LinfNorm(container.g, Hessian_shape_);
          stop = norm <= epsilon1;
          if (stop)
            std::cout << "Stopped for norm{" << norm << "} <= epsilon1{"
                      << epsilon1 << "}" << std::endl;
        } else {
          // TODO: implement this
        }
        u /= std::max(1. / 3., 1 - std::pow(2 * rho - 1, 3));
        v = 2;
      } else {
        new_residual_norm = residual_norm;
        RollbackLM();
        u /= v;
        v *= 2;
        recover_diag = true;
      }
    }
    if (stop)
      break;
  }
  WriteBack();
  DeallocateResource();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaFree(ExtractedDiag[i][0]);
    cudaFree(ExtractedDiag[i][1]);
  }
}

template <typename T> void BaseProblem<T>::BackupLM() {
  const std::vector<cublasHandle_t> &cublasHandle =
      HandleManager::get_cublasHandle();
  T one = 1.;
  if (option_.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schur_delta_x_ptr_backup[i], schur_delta_x_ptr[i],
                      Hessian_shape_ * sizeof(T), cudaMemcpyDeviceToDevice);
      Wrapper::cublasGaxpy::call(cublasHandle[i], Hessian_shape_, &one,
                                 schur_delta_x_ptr[i], 1, schur_x_ptr[i], 1);
    }
  } else {
    // TODO: implement this
  }
  edges.backupDaPtrs();
}

template <typename T> void BaseProblem<T>::RollbackLM() {
  edges.rollback();
  if (option_.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schur_delta_x_ptr[i], schur_delta_x_ptr_backup[i],
                      Hessian_shape_ * sizeof(T), cudaMemcpyDeviceToDevice);
    }
  } else {
    // TODO: implement this
  }
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}