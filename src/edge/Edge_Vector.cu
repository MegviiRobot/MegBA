/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <edge/BaseEdge.h>
#include <Macro.h>

namespace MegBA {
    template<typename T>
    void EdgeVector<T>::backupDaPtrs() {
        if (option_.use_schur) {
            const auto grad_shape = getGradShape();
            for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
                cudaSetDevice(i);
                cudaMemcpyAsync(schur_da_ptrs_old[0][i], schur_da_ptrs[0][i], Memory_Pool::getElmNum(i) * grad_shape * sizeof(T), cudaMemcpyDeviceToDevice, schur_stream_LM_memcpy_[i]);
            }
        } else {
          // TODO: implement this
        }
    }

    namespace {
        template<typename T>
        __global__ void BroadCastCsrColInd(const int *input, const int other_dim, const int nElm, int *output) {
            unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= nElm) return;
            for (int i = 0; i < other_dim; ++i) {
                output[i + tid * other_dim] = i + input[tid] * other_dim;
            }
        }
    }

    template<typename T>
    void EdgeVector<T>::preparePositionAndRelationDataCUDA() {

        if (option_.use_schur) {
            std::vector<std::array<int *, 2>> CompressedCsrColInd;
            CompressedCsrColInd.resize(option_.world_size);
            for (int i = 0; i < option_.world_size; ++i) {
                cudaSetDevice(i);
                const auto edge_num = Memory_Pool::getElmNum(i);

                cudaMalloc(&schur_equation_container_[i].csrRowPtr[0], (num[0] * schur_equation_container_[i].dim[0] + 1) * sizeof(int));
                cudaMalloc(&schur_equation_container_[i].csrRowPtr[1], (num[1] * schur_equation_container_[i].dim[1] + 1) * sizeof(int));
                cudaMemcpyAsync(schur_equation_container_[i].csrRowPtr[0], schur_csrRowPtr[i][0].get(), (num[0] * schur_equation_container_[i].dim[0] + 1) * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(schur_equation_container_[i].csrRowPtr[1], schur_csrRowPtr[i][1].get(), (num[1] * schur_equation_container_[i].dim[1] + 1) * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&schur_equation_container_[i].csrVal[0], schur_equation_container_[i].nnz[0] * sizeof(T));// hpl
                cudaMalloc(&schur_equation_container_[i].csrColInd[0], schur_equation_container_[i].nnz[0] * sizeof(int));
                {
                    const std::size_t entries_in_rows = schur_equation_container_[i].nnz[0] / schur_equation_container_[i].dim[1];
                    dim3 block(std::min(entries_in_rows, (std::size_t)512));
                    dim3 grid((entries_in_rows - 1) / block.x + 1);
                    cudaMalloc(&CompressedCsrColInd[i][0], entries_in_rows * sizeof(int));
                    cudaMemcpyAsync(CompressedCsrColInd[i][0], schur_H_entrance_[i].csrColInd_[0].get(), entries_in_rows * sizeof(int), cudaMemcpyHostToDevice);
                    ASSERT_CUDA_NO_ERROR();
                    BroadCastCsrColInd<T> <<<grid, block>>> (CompressedCsrColInd[i][0], schur_equation_container_[i].dim[1], entries_in_rows, schur_equation_container_[i].csrColInd[0]);
                    ASSERT_CUDA_NO_ERROR();
                }

                cudaMalloc(&schur_equation_container_[i].csrVal[1], schur_equation_container_[i].nnz[1] * sizeof(T));// hlp
                cudaMalloc(&schur_equation_container_[i].csrColInd[1], schur_equation_container_[i].nnz[1] * sizeof(int));
                {
                    const std::size_t entries_in_rows = schur_equation_container_[i].nnz[1] / schur_equation_container_[i].dim[0];
                    dim3 block(std::min(entries_in_rows, (std::size_t)512));
                    dim3 grid((entries_in_rows - 1) / block.x + 1);
                    cudaMalloc(&CompressedCsrColInd[i][1], entries_in_rows * sizeof(int));
                    cudaMemcpyAsync(CompressedCsrColInd[i][1], schur_H_entrance_[i].csrColInd_[1].get(), entries_in_rows * sizeof(int), cudaMemcpyHostToDevice);
                    ASSERT_CUDA_NO_ERROR();
                    BroadCastCsrColInd<T> <<<grid, block>>> (CompressedCsrColInd[i][1], schur_equation_container_[i].dim[0], entries_in_rows, schur_equation_container_[i].csrColInd[1]);
                    ASSERT_CUDA_NO_ERROR();
                }

                cudaMalloc(&schur_equation_container_[i].csrVal[2], schur_equation_container_[i].nnz[2] * sizeof(T));// hpp

                cudaMalloc(&schur_equation_container_[i].csrVal[3], schur_equation_container_[i].nnz[3] * sizeof(T));// hll

                cudaMalloc(&schur_equation_container_[i].g, (num[0] * schur_equation_container_[i].dim[0] + num[1] * schur_equation_container_[i].dim[1]) * sizeof(T));

                cudaMalloc(&schur_position_and_relation_container_[i].relative_position_camera, edge_num * sizeof(int));
                cudaMemcpyAsync(schur_position_and_relation_container_[i].relative_position_camera, schur_relative_position[0][i].data(),
                           edge_num * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&schur_position_and_relation_container_[i].relative_position_point, edge_num * sizeof(int));
                cudaMemcpyAsync(schur_position_and_relation_container_[i].relative_position_point, schur_relative_position[1][i].data(),
                           edge_num * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&schur_position_and_relation_container_[i].absolute_position_camera, camera_vertex_num * edge_num * sizeof(int));
                cudaMemcpyAsync(schur_position_and_relation_container_[i].absolute_position_camera, schur_absolute_position[0][i].data(), edge_num * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&schur_position_and_relation_container_[i].absolute_position_point, point_vertex_num * edge_num * sizeof(int));
                cudaMemcpyAsync(schur_position_and_relation_container_[i].absolute_position_point, schur_absolute_position[1][i].data(), edge_num * sizeof(int), cudaMemcpyHostToDevice);
            }
            for (int i = 0; i < option_.world_size; ++i) {
                cudaSetDevice(i);
                cudaDeviceSynchronize();
                cudaFree(CompressedCsrColInd[i][0]);
                cudaFree(CompressedCsrColInd[i][1]);
            }
        } else {
          // TODO: implement this
        }
  }

    template<typename T>
    void EdgeVector<T>::cudaPrepareUpdateData() {
        if (option_.use_schur) {
            const auto world_size = Memory_Pool::getWorldSize();
            const auto grad_shape = getGradShape();
            schur_stream_LM_memcpy_.resize(world_size);
            std::vector<T *> da_ptrs_, da_ptrs_old_;
            da_ptrs_.resize(world_size);
            da_ptrs_old_.resize(world_size);
            for (int i = 0; i < world_size; ++i) {
                cudaSetDevice(i);
                cudaStreamCreateWithFlags(&schur_stream_LM_memcpy_[i], CU_STREAM_NON_BLOCKING);
                T *da_ptr, *da_ptr_old;
                const auto nElm = Memory_Pool::getElmNum(i);
                cudaMalloc(&da_ptr, nElm * grad_shape * sizeof(T));
                cudaMalloc(&da_ptr_old, nElm * grad_shape * sizeof(T));
                da_ptrs_[i] = da_ptr;
                da_ptrs_old_[i] = da_ptr_old;
            }
            schur_da_ptrs.resize(camera_vertex_num + point_vertex_num);
            schur_da_ptrs_old.resize(camera_vertex_num + point_vertex_num);
            for (int i = 0; i < camera_vertex_num + point_vertex_num; ++i) {
                schur_da_ptrs[i].resize(world_size);
                schur_da_ptrs_old[i].resize(world_size);
            }
            for (int i = 0, i_unfixed = 0, offset = 0; i < edges.size(); ++i) {
                if (edges[i][0]->get_Fixed())
                    continue;
                for (int j = 0; j < world_size; ++j) {
                    const auto nElm = Memory_Pool::getElmNum(j);
                    schur_da_ptrs[i_unfixed][j] = &da_ptrs_[j][offset * nElm];
                    schur_da_ptrs_old[i_unfixed][j] = &da_ptrs_old_[j][offset * nElm];
                }
                i_unfixed++;
                const auto &estimation = edges[i][0]->get_Estimation();
                offset += estimation.rows() * estimation.cols();
            }
        } else {
          // TODO: implement this
        }
    }

    template<typename T>
    void EdgeVector<T>::deallocateResourceCUDA() {
        if (option_.use_schur) {
            for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
                cudaSetDevice(i);
                schur_position_and_relation_container_[i].freeCUDA();
                cudaFree(schur_da_ptrs[0][i]);
                cudaFree(schur_da_ptrs[1][i]);
                cudaFree(schur_da_ptrs_old[0][i]);
                cudaFree(schur_da_ptrs_old[1][i]);
                cudaStreamDestroy(schur_stream_LM_memcpy_[i]);
            }

            for (auto &edge : edges)
                edge.CPU();
        } else {
          // TODO: implement this
        }
    }

    template<typename T>
    void EdgeVector<T>::SchurEquationContainer::freeCUDA() {
        for (int i = 0; i < 2; ++i)
            cudaFree(csrRowPtr[i]);
        for (int i = 0; i < 4; ++i)
            cudaFree(csrVal[i]);
        for (int i = 0; i < 2; ++i)
            cudaFree(csrColInd[i]);
        cudaFree(g);
    }

    template<typename T>
    void EdgeVector<T>::PositionAndRelationContainer::freeCUDA() {
        cudaFree(relative_position_camera);
        cudaFree(relative_position_point);
        cudaFree(absolute_position_camera);
        cudaFree(absolute_position_point);
        cudaFree(connection_num_point);
    }

    template class EdgeVector<float>;
    template class EdgeVector<double>;
}