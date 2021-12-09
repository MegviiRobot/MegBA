/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "edge/BaseEdge.h"
#include "problem/BaseProblem.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include <Wrapper.hpp>
#include <resource/Manager.h>
#include <Macro.h>

#if __CUDA_ARCH__ <= 1120
#define CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#endif

namespace MegBA {
    namespace {
        template<typename T>
        __global__ void Fill_Ptr(const T *A_data, T *Ainv_data,
                                 const int batchSize, const int H_rows_num_pow2,
                                 const T **A, T **Ainv) {
            unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= batchSize) return;
            A[tid] = &A_data[tid * H_rows_num_pow2];
            Ainv[tid] = &Ainv_data[tid * H_rows_num_pow2];
        }

        template<typename T>
        void invert(const T * A_dflat, int n, const int point_num, T *C_dflat) {
            cublasHandle_t handle = HandleManager::get_cublasHandle()[0];

            const T **A;
            T **Ainv;
            cudaMalloc(&A, point_num * sizeof(T *));
            cudaMalloc(&Ainv, point_num * sizeof(T *));
            dim3 blockDim(std::min(decltype(point_num)(256), point_num));
            dim3 gridDim((point_num - 1) / blockDim.x + 1);

            Fill_Ptr<<<gridDim, blockDim>>>(A_dflat, C_dflat, point_num, n * n, A, Ainv);
            ASSERT_CUDA_NO_ERROR();
            int *INFO;
            cudaMalloc(&INFO, point_num * sizeof(int));
            Wrapper::cublasGmatinvBatched::call(handle, n, A, n, Ainv, n, INFO, point_num);
            //                PRINT_DMEMORY(A_dflat, point_num * n * n, T);
            ASSERT_CUDA_NO_ERROR();
            cudaDeviceSynchronize();

            cudaFree(A);
            cudaFree(Ainv);
            cudaFree(INFO);
        }

        template<typename T>
        void invertDistributed(const std::vector<T *> A_dflat, int n, const int point_num, std::vector<T *>C_dflat) {
            const auto &handle = HandleManager::get_cublasHandle();
            const auto world_size = MemoryPool::getWorldSize();

            std::vector<const T **> A{static_cast<std::size_t>(world_size)};
            std::vector<T **> Ainv{static_cast<std::size_t>(world_size)};
            std::vector<int *>INFO{static_cast<std::size_t>(world_size)};
            dim3 blockDim(std::min(decltype(point_num)(256), point_num));
            dim3 gridDim((point_num - 1) / blockDim.x + 1);
            ASSERT_CUDA_NO_ERROR();

            for (int i = 0; i < world_size; ++i) {
//                cudaSetDevice(i);
MemoryPool::allocateNormal((void **)&A[i], point_num * sizeof(T *), i);
MemoryPool::allocateNormal((void **)&Ainv[i],
                                            point_num * sizeof(T *), i);
MemoryPool::allocateNormal((void **)&INFO[i],
                                            point_num * sizeof(int), i);

//                cudaMalloc(&A[i], point_num * sizeof(T *));
//                cudaMalloc(&Ainv[i], point_num * sizeof(T *));
//                cudaMalloc(&INFO[i], point_num * sizeof(int));
            }

            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                Fill_Ptr<<<gridDim, blockDim>>>(A_dflat[i], C_dflat[i], point_num, n * n, A[i], Ainv[i]);
                ASSERT_CUDA_NO_ERROR();
                Wrapper::cublasGmatinvBatched::call(handle[i], n, A[i], n, Ainv[i], n, INFO[i], point_num);
            }
            //                PRINT_DMEMORY(A_dflat, point_num * n * n, T);
            ASSERT_CUDA_NO_ERROR();
            for (int i = 0; i < world_size; ++i){
                cudaSetDevice(i);
                cudaDeviceSynchronize();
                MemoryPool::deallocateNormal(INFO[i], i);
                MemoryPool::deallocateNormal(Ainv[i], i);
                MemoryPool::deallocateNormal(A[i], i);
//                cudaFree(A[i]);
//                cudaFree(Ainv[i]);
//                cudaFree(INFO[i]);
            }
        }

        template<typename T, int result_weight = 1, int dest_weight = 0>
        __global__ void oursGgemvBatched(const T *csrVal, const T *r,
                                         int batchSize,
                                         T *dx) {
            /*
                 * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
                 */
            unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
            if (tid >= batchSize) return;

            T *smem = Wrapper::Shared_Memory<T>::get();
            T sum = 0;
            smem[threadIdx.x + threadIdx.y * blockDim.x] = r[threadIdx.x + tid * blockDim.x];
            __syncthreads();
            for (unsigned int i = 0; i < blockDim.x; ++i) {
                sum += csrVal[i + threadIdx.x * blockDim.x + tid * blockDim.x * blockDim.x] * smem[i + threadIdx.y * blockDim.x];
            }
            dx[threadIdx.x + tid * blockDim.x] = result_weight * sum + dest_weight * dx[threadIdx.x + tid * blockDim.x];
        }

        template<typename T>
        __global__ void MinusKernel(const T *in1, const T *in2, int nElm, T *out) {
            unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= nElm) return;

            out[tid] = in1[tid] - in2[tid];
        }

        namespace {
            template<typename T>
            __global__ void weighted_plus_kernel(int nElm, const T *x, const T *y, T weight, T *z) {
                unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid >= nElm) return;
                z[tid] = x[tid] + weight * y[tid];
            }
        }

        template<typename T>
        bool PreconditionedConjugateGradientSolverLargeSchurDistributedCUDA(
                const std::vector<T *> &SpMVbuffer,
                std::size_t max_iter, double solver_refuse_ratio, const double tol,
                const int camera_num, const int point_num,
                const int camera_dim, const int point_dim,
                const std::vector<int> &hpl_nnz,
                const int hpp_rows, const int hll_rows,
                const std::vector<T *> &hpp_csrVal,
                const std::vector<T *> &hpl_csrVal, const std::vector<int *> &hpl_csrColInd, const std::vector<int *> &hpl_csrRowPtr,
                const std::vector<T *> &hlp_csrVal, const std::vector<int *> &hlp_csrColInd, const std::vector<int *> &hlp_csrRowPtr,
                const std::vector<T *> &hll_inv_csrVal,
                const std::vector<T *> &g,
                const std::vector<T *> &d_x) {
            const auto &comms = HandleManager::get_ncclComm();
            const auto world_size = MemoryPool::getWorldSize();
            constexpr auto cudaDataType = Wrapper::declared_cudaDatatype<T>::cuda_dtype;
            const auto &cusparseHandle = HandleManager::get_cusparseHandle();
            const auto &cublasHandle = HandleManager::get_cublasHandle();
            std::vector<cudaStream_t> cusparseStream, cublasStream;
            const T one{1.0}, zero{0.0}, neg_one{-1.0};
            T alpha_n, neg_alpha_n, rho_nm1;
            std::vector<T> dot;
            std::vector<T *> hpp_inv_csrVal, p_n, r_n, Ax_n, temp, d_x_backup;
            std::vector<cusparseSpMatDescr_t> hpl, hlp;
            std::vector<cusparseDnVecDescr_t> vecx, vecp, vecAx, vectemp;
            cusparseStream.resize(world_size);
            cublasStream.resize(world_size);
            dot.resize(world_size);
            hpp_inv_csrVal.resize(world_size);
            p_n.resize(world_size);
            r_n.resize(world_size);
            Ax_n.resize(world_size);
            temp.resize(world_size);
            d_x_backup.resize(world_size);
            hpl.resize(world_size);
            hlp.resize(world_size);
            vecx.resize(world_size);
            vecp.resize(world_size);
            vecAx.resize(world_size);
            vectemp.resize(world_size);
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);
                cublasGetStream_v2(cublasHandle[i], &cublasStream[i]);
                MemoryPool::allocateNormal((void **)&hpp_inv_csrVal[i],
                                            hpp_rows * camera_dim * sizeof(T),
                                            i);
                MemoryPool::allocateNormal((void **)&p_n[i],
                                            hpp_rows * sizeof(T), i);
                MemoryPool::allocateNormal((void **)&r_n[i],
                                            hpp_rows * sizeof(T), i);
                MemoryPool::allocateNormal((void **)&Ax_n[i],
                                            hpp_rows * sizeof(T), i);
                MemoryPool::allocateNormal((void **)&temp[i],
                                            hll_rows * sizeof(T), i);

                MemoryPool::allocateNormal((void **)&d_x_backup[i],
                                            hll_rows * sizeof(T), i);

//                cudaMalloc(&hpp_inv_csrVal[i], hpp_rows * camera_dim * sizeof(T));
//                cudaMalloc((void **) &p_n[i], hpp_rows * sizeof(T));
//                cudaMalloc((void **) &r_n[i], hpp_rows * sizeof(T));
                cudaMemcpyAsync(r_n[i], g[i], hpp_rows * sizeof(T), cudaMemcpyDeviceToDevice);
//                cudaMalloc((void **) &Ax_n[i], hpp_rows * sizeof(T));
//                cudaMalloc((void **) &temp[i], hll_rows * sizeof(T));

                /* Wrap raw data into cuSPARSE generic API objects */
                cusparseCreateCsr(&hpl[i], hpp_rows, hll_rows, hpl_nnz[i], hpl_csrRowPtr[i], hpl_csrColInd[i], hpl_csrVal[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cudaDataType);
                cusparseCreateCsr(&hlp[i], hll_rows, hpp_rows, hpl_nnz[i], hlp_csrRowPtr[i], hlp_csrColInd[i], hlp_csrVal[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cudaDataType);
                cusparseCreateDnVec(&vecx[i], hpp_rows, d_x[i], cudaDataType);
                cusparseCreateDnVec(&vecp[i], hpp_rows, p_n[i], cudaDataType);
                cusparseCreateDnVec(&vecAx[i], hpp_rows, Ax_n[i], cudaDataType);
                cusparseCreateDnVec(&vectemp[i], hll_rows, temp[i], cudaDataType);
            }

            invertDistributed(hpp_csrVal, camera_dim, camera_num, hpp_inv_csrVal);

            /* Allocate workspace for cuSPARSE */
            for (int i = 0; i < world_size; ++i){
                cudaSetDevice(i);
                /* Begin CG */
                // x1 = ET*x
//                size_t bufferSize = 0;
//                cusparseSpMV_bufferSize(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hlp[i], vecx[i], &zero, vectemp[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
//                Memory_Pool::allocate_normal((void **)&SpMVbuffer[i], bufferSize, i);

                cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hlp[i], vecx[i], &zero, vectemp[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
            }
            ASSERT_CUDA_NO_ERROR();

            ncclGroupStart();
            for (int i = 0; i < world_size; ++i) {
                ncclAllReduce(temp[i], temp[i], hll_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
            }
            ncclGroupEnd();

            for (int i = 0; i < world_size; ++i) {
                dim3 block(point_dim, std::min(32, point_num));
                dim3 grid((point_num - 1) / block.y + 1);
                cudaSetDevice(i);
                // borrow p_n as temp workspace
                oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T), cusparseStream[i]>>>(hll_inv_csrVal[i], temp[i], point_num, temp[i]);

                cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hpl[i], vectemp[i], &zero, vecAx[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
            }

            ncclGroupStart();
            for (int i = 0; i < world_size; ++i) {
                ncclAllReduce(Ax_n[i], Ax_n[i], hpp_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
            }
            ncclGroupEnd();

            for (int i = 0; i < world_size; ++i) {
                dim3 block(camera_dim, std::min(32, camera_num));
                dim3 grid((camera_num - 1) / block.y + 1);
                cudaSetDevice(i);
                oursGgemvBatched<T, 1, -1><<<grid, block, block.x * block.y * sizeof(T), cusparseStream[i]>>>(hpp_csrVal[i], d_x[i], camera_num, Ax_n[i]);
            }
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                cudaStreamSynchronize(cusparseStream[i]);
                // r = b - Ax
                Wrapper::cublasGaxpy::call(cublasHandle[i], hpp_rows, &neg_one, Ax_n[i], 1, r_n[i], 1);
            }
            int n{0};
            T rho_n{0};
            T min_rho = INFINITY;
            std::vector<T> rho_n_item;
            rho_n_item.resize(world_size);
            bool done{false};
            do {
                std::size_t offset{0};
                rho_n = 0;
                for (int i = 0; i < world_size; ++i) {
                    dim3 block(camera_dim, std::min(32, camera_num));
                    dim3 grid((camera_num - 1) / block.y + 1);
                    cudaSetDevice(i);
                    // borrow Ax_n
                    oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T), cublasStream[i]>>>(hpp_inv_csrVal[i], r_n[i], camera_num, Ax_n[i]);
//                    cudaMemcpyAsync(Ax_n[i], r_n[i], hpp_rows * sizeof(T), cudaMemcpyDeviceToDevice);

                    // rho_n = rTr
                    const auto nElm = MemoryPool::getElmNum(i, hpp_rows);
                    Wrapper::cublasGdot::call(cublasHandle[i], nElm, &r_n[i][offset], 1, &Ax_n[i][offset], 1, &rho_n_item[i]);
                    offset += nElm;
                }
                for (int i = 0; i < world_size; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(cublasStream[i]);
                    rho_n += rho_n_item[i];
                }
                if (rho_n > solver_refuse_ratio * min_rho) {
                    for (int i = 0; i < world_size; ++ i) {
                        cudaSetDevice(i);
                        cudaMemcpyAsync(d_x[i], d_x_backup[i], hpp_rows * sizeof(T), cudaMemcpyDeviceToDevice);
                    }
//                    std::cout << "cg pre stopped for new rho{" << rho_n << "} > 1.5 * min_rho{" << min_rho << "}, ";
                    break;
                }
                min_rho = std::min(min_rho, rho_n);

                if (n >= 1) {
                    T beta_n = rho_n / rho_nm1;
                    for (int i = 0; i < world_size; ++i) {
                        dim3 block(std::min(256, hpp_rows));
                        dim3 grid((hpp_rows - 1) / block.x + 1);
                        cudaSetDevice(i);
                        weighted_plus_kernel<T><<<grid, block>>>(hpp_rows, Ax_n[i], p_n[i], beta_n, p_n[i]);
                    }
                } else {
                    for (int i = 0; i < world_size; ++i) {
                        cudaSetDevice(i);
                        Wrapper::cublasGcopy::call(cublasHandle[i], hpp_rows, Ax_n[i], 1, p_n[i], 1);
                    }
                }

                for (int i = 0; i < world_size; ++i) {
                    //Ax = Ad ???? q = Ad
                    // x1 = ET*x
                    cudaSetDevice(i);
                    cudaStreamSynchronize(cublasStream[i]);
                    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hlp[i], vecp[i], &zero, vectemp[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
                }

                ncclGroupStart();
                for (int i = 0; i < world_size; ++i) {
                    ncclAllReduce(temp[i], temp[i], hll_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
                }
                ncclGroupEnd();

                for (int i = 0; i < world_size; ++i) {
                    dim3 block(point_dim, std::min(32, point_num));
                    dim3 grid((point_num - 1) / block.y + 1);
                    cudaSetDevice(i);
                    // borrow p_n as temp workspace
                    oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T), cusparseStream[i]>>>(hll_inv_csrVal[i], temp[i], point_num, temp[i]);

                    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hpl[i], vectemp[i], &zero, vecAx[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
                }

                ncclGroupStart();
                for (int i = 0; i < world_size; ++i) {
                    ncclAllReduce(Ax_n[i], Ax_n[i], hpp_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
                }
                ncclGroupEnd();

                for (int i = 0; i < world_size; ++i) {
                    dim3 block(camera_dim, std::min(32, camera_num));
                    dim3 grid((camera_num - 1) / block.y + 1);
                    cudaSetDevice(i);
                    oursGgemvBatched<T, 1, -1><<<grid, block, block.x * block.y * sizeof(T), cusparseStream[i]>>>(hpp_csrVal[i], p_n[i], camera_num, Ax_n[i]);
                }

                offset = 0;
                for (int i = 0; i < world_size; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(cusparseStream[i]);
                    //dot :dTq
                    const auto nElm = MemoryPool::getElmNum(i, hpp_rows);
                    Wrapper::cublasGdot::call(cublasHandle[i], nElm, &p_n[i][offset], 1, &Ax_n[i][offset], 1, &dot[i]);
                    offset += nElm;
                }
                // beta_n: one = rho_n / dTq
                double dot_sum{0};
                for (int i = 0; i < world_size; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(cublasStream[i]);
                    dot_sum += dot[i];
                }
                alpha_n = rho_n / dot_sum;
                for (int i = 0; i < world_size; ++i) {
                    cudaSetDevice(i);
                    // x=x+alpha_n*p_n
                    cudaMemcpyAsync(d_x_backup[i], d_x[i], hpp_rows * sizeof(T), cudaMemcpyDeviceToDevice);
                    Wrapper::cublasGaxpy::call(cublasHandle[i], hpp_rows, &alpha_n, p_n[i], 1, d_x[i], 1);
                }

                neg_alpha_n = -alpha_n;

//                if ((n + 1) % 10 == 0) {
//                    for (int i = 0; i < world_size; ++i) {
//                        //Ax = Ad ???? q = Ad
//                        // x1 = ET*x
//                        cudaSetDevice(i);
//                        cudaStreamSynchronize(cublasStream[i]);
//                        cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hlp[i], vecx[i], &zero, vectemp[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
//                    }
//
//                    ncclGroupStart();
//                    for (int i = 0; i < world_size; ++i) {
//                        ncclAllReduce(temp[i], temp[i], hll_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
//                    }
//                    ncclGroupEnd();
//
//                    blockDim.x = point_dim;
//                    blockDim.y = std::min(32, point_num);
//                    gridDim.x = (point_num - 1) / blockDim.y + 1;
//                    for (int i = 0; i < world_size; ++i) {
//                        cudaSetDevice(i);
//                        // borrow p_n as temp workspace
//                        oursGgemvBatched<<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T), cusparseStream[i]>>>(hll_inv_csrVal[i], temp[i], point_num, temp[i]);
//
//                        cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hpl[i], vectemp[i], &zero, vecAx[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
//                    }
//
//                    ncclGroupStart();
//                    for (int i = 0; i < world_size; ++i) {
//                        ncclAllReduce(Ax_n[i], Ax_n[i], hpp_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
//                    }
//                    ncclGroupEnd();
//
//                    blockDim.x = camera_dim;
//                    blockDim.y = std::min(32, camera_num);
//                    gridDim.x = (camera_num - 1) / blockDim.y + 1;
//                    for (int i = 0; i < world_size; ++i) {
//                        cudaSetDevice(i);
//                        oursGgemvBatched<T, 1, -1><<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T), cusparseStream[i]>>>(hpp_csrVal[i], d_x[i], camera_num, Ax_n[i]);
//                        dim3 block(std::min(256, hpp_rows));
//                        dim3 grid((hpp_rows - 1) / block.x + 1);
//                        MinusKernel<T><<<grid, block>>>(g[i], Ax_n[i], hpp_rows, r_n[i]);
//                    }
//                } else {
//                    for (int i = 0; i < world_size; ++i) {
//                        cudaSetDevice(i);
//                        // r = r - alpha_n*Ax = r - alpha_n*q
//                        Wrapper::cublasGaxpy::call(cublasHandle[i], hpp_rows, &neg_alpha_n, Ax_n[i], 1, r_n[i], 1);
//                    }
//                }

                for (int i = 0; i < world_size; ++i) {
                    cudaSetDevice(i);
                    // r = r - alpha_n*Ax = r - alpha_n*q
                    Wrapper::cublasGaxpy::call(cublasHandle[i], hpp_rows, &neg_alpha_n, Ax_n[i], 1, r_n[i], 1);
                }
                rho_nm1 = rho_n;
                //                printf("iteration = %3d, residual = %f\n", n, std::abs(rho_n));
                ++n;
                done = std::abs(rho_n) < tol;
            } while (!done && n < max_iter);
//            cudaSetDevice(0);
//            PRINT_DMEMORY_SEGMENT(d_x[0], 0, 2, T);
//            std::cout << "CG iteration: " << n << ", with error: " << std::abs(rho_n) << std::endl;
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                cusparseDestroySpMat(hpl[i]);
                cusparseDestroySpMat(hlp[i]);
                cusparseDestroyDnVec(vecx[i]);
                cusparseDestroyDnVec(vecAx[i]);
                cusparseDestroyDnVec(vecp[i]);
                cusparseDestroyDnVec(vectemp[i]);

                MemoryPool::deallocateNormal(d_x_backup[i], i);
                MemoryPool::deallocateNormal(temp[i], i);
                MemoryPool::deallocateNormal(Ax_n[i], i);
                MemoryPool::deallocateNormal(r_n[i], i);
                MemoryPool::deallocateNormal(p_n[i], i);
                MemoryPool::deallocateNormal(hpp_inv_csrVal[i], i);

//                cudaFree(p_n[i]);
//                cudaFree(r_n[i]);
//                cudaFree(Ax_n[i]);
//                cudaFree(temp[i]);
//                cudaFree(hpp_inv_csrVal[i]);
            }
            return done;
        }

        template<typename T>
        void SchurMakeVDistributed(
                std::vector<T *> &SpMVbuffer,
                const int point_num, const int point_dim,
                const std::vector<int> &hpl_nnz, const int hpp_rows, const int hll_rows,
                const std::vector<T *> &hpl_csrVal,
                const std::vector<int *> &hpl_csrColInd,
                const std::vector<int *> &hpl_csrRowPtr,
                const std::vector<T *> &hll_inv_csrVal,
                const std::vector<T *> &d_r) {
            const auto &comms = HandleManager::get_ncclComm();
            const auto world_size = MemoryPool::getWorldSize();
            const auto &cusparseHandle = HandleManager::get_cusparseHandle();
            constexpr auto cudaDataType = Wrapper::declared_cudaDatatype<T>::cuda_dtype;

            std::vector<T *> v, w;
            std::vector<cudaStream_t> cusparseStream;
            std::vector<cusparseDnVecDescr_t> vecv, vecw;
            std::vector<cusparseSpMatDescr_t> hpl;
            v.resize(world_size);
            w.resize(world_size);
            cusparseStream.resize(world_size);
            vecv.resize(world_size);
            vecw.resize(world_size);
            hpl.resize(world_size);
            for (int i = 0; i < world_size; ++i) {
                cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);
                v[i] = &d_r[i][0];
                w[i] = &d_r[i][hpp_rows];
                cusparseCreateDnVec(&vecv[i], hpp_rows, v[i], cudaDataType);
                cusparseCreateDnVec(&vecw[i], hll_rows, w[i], cudaDataType);
                cusparseCreateCsr(&hpl[i], hpp_rows, hll_rows, hpl_nnz[i], hpl_csrRowPtr[i], hpl_csrColInd[i], hpl_csrVal[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cudaDataType);
            }

            dim3 blockDim(point_dim, std::min(32, point_num));
            dim3 gridDim((point_num - 1) / blockDim.y + 1);
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                // notably, w here is changed(w = C^{-1}w), so later w = C^{-1}(w - ETv) = C^{-1}w - C^{-1}ETv -> w = w - C^{-1}ETv
                oursGgemvBatched<<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T), cusparseStream[i]>>>(hll_inv_csrVal[i], w[i], point_num, w[i]);
            }

            T alpha{-1.0}, beta = T(1. / world_size);

            SpMVbuffer.resize(world_size);
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
//                PRINT_DMEMORY(hpl_csrVal[i], hpl_nnz[i], T);
//                PRINT_DMEMORY(hpl_csrColInd[i], hpl_nnz[i], int);
//                PRINT_DMEMORY(hpl_csrRowPtr[i], hpp_rows + 1, int);
//                PRINT_DCSR(hpl_csrVal[i], hpl_csrColInd[i], hpl_csrRowPtr[i], hpp_rows, T);
//                PRINT_DMEMORY(w[i], hll_rows, T);
                size_t bufferSize = 0;
                cusparseSpMV_bufferSize(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, hpl[i], vecw[i], &beta, vecv[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
                MemoryPool::allocateNormal((void **)&SpMVbuffer[i], bufferSize,
                                            i);
//                cudaMalloc(&SpMVbuffer[i], bufferSize);
                cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, hpl[i], vecw[i], &beta, vecv[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
//                PRINT_DMEMORY(v[i], hpp_rows, T);
            }
            ASSERT_CUDA_NO_ERROR();

            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                cudaStreamSynchronize(cusparseStream[i]);

                cusparseDestroySpMat(hpl[i]);
                cusparseDestroyDnVec(vecv[i]);
                cusparseDestroyDnVec(vecw[i]);
            }
            ncclGroupStart();
            for (int i = 0; i < world_size; ++i) {
                ncclAllReduce(v[i], v[i], hpp_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
            }
            ncclGroupEnd();
//            cudaSetDevice(0);
//            PRINT_DMEMORY(v[0], hpp_rows, T);
        }

        template<typename T>
        void SchurSolveW(
                const int point_num, const int point_dim,
                const int hpl_nnz,
                const int hpp_rows, const int hll_rows,
                T *hlp_csrVal, int *hlp_csrColInd, int *hlp_csrRowPtr,
                T *hll_inv_csrVal,
                T *d_r,
                T *d_x) {
            T *xc = &d_x[0];
            T *xp = &d_x[hpp_rows];
            T *w = &d_r[hpp_rows];

            cusparseHandle_t cusparseHandle = HandleManager::get_cusparseHandle()[0];

            cudaStream_t cusparseStream;
            cusparseGetStream(cusparseHandle, &cusparseStream);

            const auto cudaDataType = Wrapper::declared_cudaDatatype<T>::get();
            cusparseDnVecDescr_t vecxc;
            cusparseCreateDnVec(&vecxc, hpp_rows, xc, cudaDataType);
            cusparseDnVecDescr_t vecxp;
            cusparseCreateDnVec(&vecxp, hll_rows, xp, cudaDataType);
            cusparseDnVecDescr_t vecw;
            cusparseCreateDnVec(&vecw, hll_rows, w, cudaDataType);

            cusparseSpMatDescr_t hlp{nullptr};
            cusparseCreateCsr(&hlp, hll_rows, hpp_rows, hpl_nnz, hlp_csrRowPtr, hlp_csrColInd, hlp_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cudaDataType);

            T alpha{1.0}, beta{0.0};
            size_t bufferSize = 0;
            cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, hlp, vecxc, &beta, vecxp, cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
            void *buffer1 = nullptr;
            cudaMalloc(&buffer1, bufferSize);

            /* Begin CG */
            // x1 = ET*x
            cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, hlp, vecxc, &beta, vecxp, cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, buffer1);

            dim3 blockDim(point_dim, std::min(32, point_num));
            dim3 gridDim((point_num - 1) / blockDim.y + 1);
            oursGgemvBatched<T, -1, 1><<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T), cusparseStream>>>(hll_inv_csrVal, xp, point_num, w);
            cudaMemcpyAsync(xp, w, hll_rows * sizeof(T), cudaMemcpyDeviceToDevice, cusparseStream);
            cudaStreamSynchronize(cusparseStream);

            cusparseDestroySpMat(hlp);
            cusparseDestroyDnVec(vecxc);
            cusparseDestroyDnVec(vecw);

            cudaFree(buffer1);
        }

        template<typename T>
        void SchurSolveWDistributed(
                const std::vector<T *> &SpMVbuffer,
                const int point_num, const int point_dim,
                const std::vector<int> &hpl_nnz,
                const int hpp_rows, const int hll_rows,
                const std::vector<T *> &hlp_csrVal, const std::vector<int *> &hlp_csrColInd, const std::vector<int *> &hlp_csrRowPtr,
                const std::vector<T *> &hll_inv_csrVal,
                const std::vector<T *> &d_r,
                const std::vector<T *> &d_x) {
            const auto comms = HandleManager::get_ncclComm();
            const auto world_size = MemoryPool::getWorldSize();
            constexpr auto cudaDataType = Wrapper::declared_cudaDatatype<T>::cuda_dtype;

            std::vector<T *> xc, xp, w;
            xc.resize(world_size);
            xp.resize(world_size);
            w.resize(world_size);
            for (int i = 0; i < world_size; ++i) {
                xc[i] = &d_x[i][0];
                xp[i] = &d_x[i][hpp_rows];
                w[i] = &d_r[i][hpp_rows];
            }

            const auto &cusparseHandle = HandleManager::get_cusparseHandle();

            std::vector<cudaStream_t> cusparseStream;
            std::vector<cusparseDnVecDescr_t> vecxc, vecxp, vecw;
            std::vector<cusparseSpMatDescr_t> hlp;
            cusparseStream.resize(world_size);
            vecxc.resize(world_size);
            vecxp.resize(world_size);
            vecw.resize(world_size);
            hlp.resize(world_size);

            for (int i = 0; i < world_size; ++i){
                cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);

                cusparseCreateDnVec(&vecxc[i], hpp_rows, xc[i], cudaDataType);
                cusparseCreateDnVec(&vecxp[i], hll_rows, xp[i], cudaDataType);
                cusparseCreateDnVec(&vecw[i], hll_rows, w[i], cudaDataType);
                cusparseCreateCsr(&hlp[i], hll_rows, hpp_rows, hpl_nnz[i], hlp_csrRowPtr[i], hlp_csrColInd[i], hlp_csrVal[i], CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cudaDataType);
            }


            T alpha{1.0}, beta{0.0};
            for (int i = 0; i < world_size; ++i){
                cudaSetDevice(i);
                // x1 = ET*x
                cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, hlp[i], vecxc[i], &beta, vecxp[i], cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
            }

            ncclGroupStart();
            for (int i = 0; i < world_size; ++i) {
                ncclAllReduce(xp[i], xp[i], hll_rows, Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum, comms[i], cusparseStream[i]);
            }
            ncclGroupEnd();

            dim3 blockDim(point_dim, std::min(32, point_num));
            dim3 gridDim((point_num - 1) / blockDim.y + 1);
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                oursGgemvBatched<T, -1, 1><<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T), cusparseStream[i]>>>(hll_inv_csrVal[i], xp[i], point_num, w[i]);
                cudaMemcpyAsync(xp[i], w[i], hll_rows * sizeof(T), cudaMemcpyDeviceToDevice, cusparseStream[i]);
            }

            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                cudaStreamSynchronize(cusparseStream[i]);

                cusparseDestroySpMat(hlp[i]);
                cusparseDestroyDnVec(vecxc[i]);
                cusparseDestroyDnVec(vecw[i]);
            }
        }
    }

    template<typename T>
    bool SchurSolverDistributed(
            double tol, double solver_refuse_ratio, std::size_t max_iter,
            const std::vector<T *> &Hpp_csrVal,
            const std::vector<T *> &Hll_csrVal,
            const std::vector<T *> &Hpl_csrVal, const std::vector<int *> &Hpl_csrColInd, const std::vector<int *> &Hpl_csrRowPtr,
            const std::vector<T *> &Hlp_csrVal, const std::vector<int *> &Hlp_csrColInd, const std::vector<int *> &Hlp_csrRowPtr,
            const std::vector<T *> &d_g,
            int camera_dim, int camera_num,
            int point_dim, int point_num,
            const std::vector<int>  &Hpl_nnz,
            int Hpp_rows, int Hll_rows,
            const std::vector<T *> &delta_x) {
        ASSERT_CUDA_NO_ERROR();
        // hll inverse-----------------------------------------------------------
        const auto world_size = MemoryPool::getWorldSize();

        ASSERT_CUDA_NO_ERROR();
//        cudaSetDevice(0);
//        PRINT_DMEMORY(delta_x[0], 9, T);
        std::vector<T *> SpMVbuffer;

        std::vector<T *> Hll_inv_csrVal;
        Hll_inv_csrVal.resize(world_size);
        for (int i = 0; i < world_size; ++i) {
          MemoryPool::allocateNormal((void **)&Hll_inv_csrVal[i],
                                      Hll_rows * point_dim * sizeof(T), i);
        }
        invertDistributed(Hll_csrVal, point_dim, point_num, Hll_inv_csrVal);

//        cudaSetDevice(0);
//        PRINT_DMEMORY(Hll_inv_csrVal[0], 9, T);
        ASSERT_CUDA_NO_ERROR();

        SchurMakeVDistributed(
                SpMVbuffer,
                point_num, point_dim,
                Hpl_nnz, Hpp_rows, Hll_rows,
                Hpl_csrVal,
                Hpl_csrColInd,
                Hpl_csrRowPtr,
                Hll_inv_csrVal,
                d_g);
        bool PCG_success = PreconditionedConjugateGradientSolverLargeSchurDistributedCUDA(
                SpMVbuffer,
                max_iter, solver_refuse_ratio, tol,
                camera_num, point_num,
                camera_dim, point_dim,
                Hpl_nnz,
                Hpp_rows, Hll_rows,
                Hpp_csrVal,
                Hpl_csrVal, Hpl_csrColInd, Hpl_csrRowPtr,
                Hlp_csrVal, Hlp_csrColInd, Hlp_csrRowPtr,
                Hll_inv_csrVal,
                d_g,
                delta_x);
        SchurSolveWDistributed(
                SpMVbuffer,
                point_num, point_dim,
                Hpl_nnz,
                Hpp_rows, Hll_rows,
                Hlp_csrVal, Hlp_csrColInd, Hlp_csrRowPtr,
                Hll_inv_csrVal,
                d_g,
                delta_x);
        for (int i = 0; i < world_size; ++i) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            MemoryPool::deallocateNormal(SpMVbuffer[i], i);
            MemoryPool::deallocateNormal(Hll_inv_csrVal[i], i);
        }
//        cudaSetDevice(0);
//        PRINT_DMEMORY(delta_x[0], 9, T);
        return PCG_success;
    }

    template<typename T>
    bool BaseProblem<T>::cudaSolveLinear(double tol, double solver_refuse_ratio, std::size_t max_iter) {
        bool success;

        if (option_.useSchur) {
            // TODO: need great change
            const auto world_size = MemoryPool::getWorldSize();
            std::vector<T *> Hpp_csrVal{static_cast<std::size_t>(world_size)};
            std::vector<T *> Hll_csrVal{static_cast<std::size_t>(world_size)};
            std::vector<T *> Hpl_csrVal{static_cast<std::size_t>(world_size)};
            std::vector<T *> Hlp_csrVal{static_cast<std::size_t>(world_size)};
            std::vector<int *> Hpl_csrColInd{static_cast<std::size_t>(world_size)};
            std::vector<int *> Hlp_csrColInd{static_cast<std::size_t>(world_size)};
            std::vector<int *> Hpl_csrRowPtr{static_cast<std::size_t>(world_size)};
            std::vector<int *> Hlp_csrRowPtr{static_cast<std::size_t>(world_size)};
            std::vector<T *> g{static_cast<std::size_t>(world_size)};
            int camera_dim;
            int camera_num;
            int point_dim;
            int point_num;
            std::vector<int> Hpl_nnz{};
            Hpl_nnz.resize(world_size);
            int Hpp_rows;
            int Hll_rows;
            std::vector<T *> delta_x{static_cast<std::size_t>(world_size)};

            for (int i = 0; i < world_size; ++i) {
                auto &schur_equation_container = edges.schurEquationContainer[i];
                Hpp_csrVal[i] = schur_equation_container.csrVal[2];
                Hll_csrVal[i] = schur_equation_container.csrVal[3];
                Hpl_csrVal[i] = schur_equation_container.csrVal[0];
                Hlp_csrVal[i] = schur_equation_container.csrVal[1];
                Hpl_csrColInd[i] = schur_equation_container.csrColInd[0];
                Hlp_csrColInd[i] = schur_equation_container.csrColInd[1];
                Hpl_csrRowPtr[i] = schur_equation_container.csrRowPtr[0];
                Hlp_csrRowPtr[i] = schur_equation_container.csrRowPtr[1];
                g[i] = schur_equation_container.g;
                camera_dim = schur_equation_container.dim[0];
                camera_num = schur_equation_container.nnz[2] / schur_equation_container.dim[0] / schur_equation_container.dim[0];
                point_dim = schur_equation_container.dim[1];
                point_num = schur_equation_container.nnz[3] / schur_equation_container.dim[1] / schur_equation_container.dim[1];
                Hpl_nnz[i] = schur_equation_container.nnz[0];
                Hpp_rows = schur_equation_container.nnz[2] / schur_equation_container.dim[0];
                Hll_rows = schur_equation_container.nnz[3] / schur_equation_container.dim[1];
                delta_x[i] = schur_delta_x_ptr[i];
            }

            success = SchurSolverDistributed(
                    tol, solver_refuse_ratio, max_iter,
                    Hpp_csrVal,
                    Hll_csrVal,
                    Hpl_csrVal, Hpl_csrColInd, Hpl_csrRowPtr,
                    Hlp_csrVal, Hlp_csrColInd, Hlp_csrRowPtr,
                    g,
                    camera_dim, camera_num,
                    point_dim, point_num,
                    Hpl_nnz,
                    Hpp_rows, Hll_rows,
                    delta_x);
        } else {
          // TODO: implement this
        }
        return success;
    }

    template class BaseProblem<double>;
    template class BaseProblem<float>;
}
