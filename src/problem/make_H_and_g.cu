/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "Macro.h"
#include <cuda_runtime.h>
#include "edge/BaseEdge.h"
#include "Wrapper.hpp"
#include <resource/Manager.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <Eigen/Sparse>

#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
union AtomicUnion{
    double d_value;
    unsigned long long ull_value;
};

__inline__ __device__ double atomicAdd(double* address, double val) {
    AtomicUnion old, assumed;
    old.d_value = *address;

    do {
        assumed = old;
        old.ull_value = atomicCAS((unsigned long long *)address,
                                  assumed.ull_value,
                                  AtomicUnion{val + assumed.d_value}.ull_value);

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed.ull_value != old.ull_value);

    return old.d_value;
}
#endif

namespace MegBA {
    namespace problem {
        namespace CUDA {
            template<typename T>
            __device__ void make_Hpp(const T *Val_smem, const T Val_i,
                                     const int camera_dim,
                                     const int Hpp_csrRow_i,
                                     T *Hpp_csrVal) {
                for (int i = 0; i < camera_dim; ++i)
                    atomicAdd(&Hpp_csrVal[Hpp_csrRow_i + i], Val_i * Val_smem[i * blockDim.x + threadIdx.x]);
            }

            template<typename T>
            __device__ void make_Hpl(const T *Val_smem, const T Val_i,
                                     const int relative_position_point,
                                     const int point_dim, const int camera_dim,
                                     int Hpl_csrRow, T *Hpl_csrVal) {
                const int Hpl_csrRow_i = Hpl_csrRow + relative_position_point * point_dim;

                for (int i = 0; i < point_dim; ++i) {
                    Hpl_csrVal[Hpl_csrRow_i + i] += Val_i * Val_smem[(i + camera_dim) * blockDim.x + threadIdx.x];
                }
            }

            template<typename T>
            __device__ void make_Hlp(const T *Val_smem, const T Val_i,
                                     const int relative_position_camera,
                                     const int camera_dim,
                                     int Hlp_csrRow, T *Hlp_csrVal) {
                const int Hlp_csrRow_i = Hlp_csrRow + relative_position_camera * camera_dim;

                for (int i = 0; i < camera_dim; ++i) {
                    Hlp_csrVal[Hlp_csrRow_i + i] += Val_i * Val_smem[i * blockDim.x + threadIdx.x];
                }
            }

            template<typename T>
            __device__ void make_Hll(const T *Val_smem, const T Val_i,
                                     const int point_dim, const int camera_dim,
                                     const int Hll_position,
                                     T *Hll_matrix) {

                for (int i = 0; i < point_dim; ++i)
                    atomicAdd(&Hll_matrix[Hll_position + i], Val_i * Val_smem[(i + camera_dim) * blockDim.x + threadIdx.x]);
            }

            template<typename T>
            __global__ void make_H_schur(const T *const *const Val_ptrs, const T *const *const error_ptrs,
                                         const int *absolute_position_camera, const int *absolute_position_point,
                                         const int *relative_position_camera, const int *relative_position_point,
                                         const int *Hpl_csrRowPtr, const int *Hlp_csrRowPtr,
                                         const int res_dim,
                                         const int camera_dim, const int point_dim, const int error_num,
                                         T *g_camera, T *g_point,
                                         T *Hpp_csrVal,
                                         T *Hll_csrVal,
                                         T *Hpl_csrVal,
                                         T *Hlp_csrVal) {
                /*
                 * make sure that blockDim.x % 32 == 0, if so, there won't be any thread divergence within a wrap.
                 */
                const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
                if (tid >= error_num)
                    return;

                T* Val_smem = Wrapper::Shared_Memory<T>::get();

                const int absolute_position_point_local = absolute_position_point[tid];
                const int absolute_position_camera_local = absolute_position_camera[tid];
                const int relative_position_point_local = relative_position_point[tid];
                const int relative_position_camera_local = relative_position_camera[tid];

                T sum_g{0.};
                for (int i = 0; i < res_dim; ++i){
                    const T Val_i = Val_ptrs[i][error_num * threadIdx.y + tid];
                    __syncthreads();
                    Val_smem[threadIdx.y * blockDim.x + threadIdx.x] = Val_i;
                    __syncthreads();

                    if (threadIdx.y < camera_dim) {
                        make_Hpp(Val_smem, Val_i,
                                 camera_dim,
                                 (absolute_position_camera_local * camera_dim + threadIdx.y) * camera_dim,
                                 Hpp_csrVal);
                        make_Hpl(Val_smem, Val_i,
                                 relative_position_point_local,
                                 point_dim, camera_dim,
                                 Hpl_csrRowPtr[absolute_position_camera_local * camera_dim + threadIdx.y],
                                 Hpl_csrVal);
                    } else {
                        make_Hll(Val_smem, Val_i,
                                 point_dim, camera_dim,
                                 absolute_position_point_local * (point_dim * point_dim) +
                                 (threadIdx.y - camera_dim) * point_dim /* Hll_position */,
                                 Hll_csrVal);
                        make_Hlp(Val_smem, Val_i,
                                 relative_position_camera_local,
                                 camera_dim,
                                 Hlp_csrRowPtr[absolute_position_point_local * point_dim + threadIdx.y - camera_dim],
                                 Hlp_csrVal);
                    }
                    sum_g += -Val_i * error_ptrs[i][tid];
                }

                if (threadIdx.y < camera_dim) {
                    atomicAdd(&g_camera[absolute_position_camera_local * camera_dim + threadIdx.y], sum_g);
                } else {
                    atomicAdd(&g_point[absolute_position_point_local * point_dim + threadIdx.y - camera_dim], sum_g);
                }
            }
        }
    }

    template<typename T, int result_weight = 1, int dest_weight = 0>
    __global__ void oursGgemvBatched(const T *csrVal, const T *r, int batchSize, T *dx);

    template<typename T>
    void EdgeVector<T>::buildLinearSystemSchurCUDA(const JVD<T> &jetEstimation) {
        const auto rows = jetEstimation.rows(), cols = jetEstimation.cols();
        const auto camera_dim = edges[0].getGradShape();
        const auto point_dim = edges[1].getGradShape();
        const auto camera_num = num[0];
        const auto point_num = num[1];
        const auto Hpp_rows = camera_dim * camera_num;
        const auto Hll_rows = point_dim * point_num;
        ASSERT_CUDA_NO_ERROR();

        std::vector<T *>d_g_camera{static_cast<std::size_t>(_option.worldSize)};
        std::vector<T *>d_g_point{static_cast<std::size_t>(_option.worldSize)};
        for (int i = 0; i < _option.worldSize; ++i) {
            cudaSetDevice(i);
            cudaMemsetAsync(schurEquationContainer[i].g, 0, (Hpp_rows + Hll_rows) * sizeof(T));
            d_g_camera[i] = &schurEquationContainer[i].g[0];
            d_g_point[i] = &schurEquationContainer[i].g[Hpp_rows];
            ASSERT_CUDA_NO_ERROR();
            cudaMemsetAsync(schurEquationContainer[i].csrVal[0], 0,
                            schurEquationContainer[i].nnz[0] * sizeof(T));
            cudaMemsetAsync(schurEquationContainer[i].csrVal[1], 0,
                            schurEquationContainer[i].nnz[1] * sizeof(T));
            cudaMemsetAsync(schurEquationContainer[i].csrVal[2], 0,
                            schurEquationContainer[i].nnz[2] * sizeof(T));
            cudaMemsetAsync(schurEquationContainer[i].csrVal[3], 0,
                            schurEquationContainer[i].nnz[3] * sizeof(T));
            ASSERT_CUDA_NO_ERROR();
        }

        const auto res_dim = rows * cols;
        std::vector<std::unique_ptr<const T *[]>> total_ptrs{};
        total_ptrs.reserve(_option.worldSize);
        std::vector<const T **> device_total_ptrs{static_cast<std::size_t>(_option.worldSize)};

        std::vector<const T **> val_ptrs{static_cast<std::size_t>(_option.worldSize)};
        std::vector<const T **> device_val_ptrs{static_cast<std::size_t>(_option.worldSize)};

        std::vector<const T **> error_ptrs{static_cast<std::size_t>(_option.worldSize)};
        std::vector<const T **> device_error_ptrs{static_cast<std::size_t>(_option.worldSize)};
        for (int device_rank = 0; device_rank < _option.worldSize; ++device_rank) {
            total_ptrs.emplace_back(new const T *[res_dim * (3 + res_dim)]);
            cudaSetDevice(device_rank);
            cudaMalloc(&device_total_ptrs[device_rank], res_dim * (3 + res_dim) * sizeof(T *));

            val_ptrs[device_rank] = &total_ptrs[device_rank][0];
            device_val_ptrs[device_rank] = &device_total_ptrs[device_rank][0];

            error_ptrs[device_rank] = &total_ptrs[device_rank][res_dim];
            device_error_ptrs[device_rank] = &device_total_ptrs[device_rank][res_dim];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    const auto &Jet_Estimation_inner = jetEstimation(i, j);
                    val_ptrs[device_rank][j + i * cols] = Jet_Estimation_inner.getCUDAGradPtr()[device_rank];
                    error_ptrs[device_rank][j + i * cols] = Jet_Estimation_inner.getCUDAResPtr()[device_rank];
                }
            cudaMemcpyAsync(device_total_ptrs[device_rank], total_ptrs[device_rank].get(), res_dim * 2 * sizeof(T *), cudaMemcpyHostToDevice);
        }

        if (jetInformation.rows() != 0 && jetInformation.cols() != 0) {

        } else {
            for (int i = 0; i < _option.worldSize; ++i) {
                cudaSetDevice(i);
                const auto edge_num = MemoryPool::getElmNum(i);
                dim3 block(std::min((decltype(edge_num))32, edge_num), camera_dim + point_dim);
                dim3 grid((edge_num - 1) / block.x + 1);
                problem::CUDA::make_H_schur<<<grid, block, block.x * block.y * sizeof(T)>>>(
                        device_val_ptrs[i], device_error_ptrs[i],
                    schurPositionAndRelationContainer[i].absolutePositionCamera,
                    schurPositionAndRelationContainer[i].absolutePositionPoint,
                    schurPositionAndRelationContainer[i].relativePositionCamera,
                    schurPositionAndRelationContainer[i].relativePositionPoint,
                    schurEquationContainer[i].csrRowPtr[0],
                    schurEquationContainer[i].csrRowPtr[1],
                        res_dim,
                        camera_dim, point_dim, edge_num,
                        d_g_camera[i], d_g_point[i],
                    schurEquationContainer[i].csrVal[2],
                    schurEquationContainer[i].csrVal[3],
                    schurEquationContainer[i].csrVal[0],
                    schurEquationContainer[i].csrVal[1]);
            }
        }
        ASSERT_CUDA_NO_ERROR();
        for (int i = 0; i < _option.worldSize; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(nullptr);
            cudaFree(device_total_ptrs[i]);
        }

        const auto &comms = HandleManager::get_ncclComm();
        ncclGroupStart();
        for (int i = 0; i < _option.worldSize; ++i) {
            ncclAllReduce(schurEquationContainer[i].csrVal[2],
                        schurEquationContainer[i].csrVal[2],
                        schurEquationContainer[i].nnz[2], Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], nullptr);
            ncclAllReduce(schurEquationContainer[i].csrVal[3],
                          schurEquationContainer[i].csrVal[3],
                          schurEquationContainer[i].nnz[3], Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], nullptr);
            ncclAllReduce(d_g_camera[i], d_g_camera[i], Hpp_rows + Hll_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], nullptr);
        }
        ncclGroupEnd();
    }

    template class EdgeVector<double>;
    template class EdgeVector<float>;
}
