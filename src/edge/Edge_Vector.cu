/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <edge/BaseEdge.h>
#include <Macro.h>

namespace MegBA {
template <typename T> void EdgeVector<T>::backupDaPtrs() {
  if (_option.use_schur) {
    const auto grad_shape = getGradShape();
    for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schurDaPtrsOld[0][i], schurDaPtrs[0][i],
                      Memory_Pool::getElmNum(i) * grad_shape * sizeof(T),
                      cudaMemcpyDeviceToDevice, schurStreamLmMemcpy[i]);
    }
  } else {
    // TODO: implement this
  }
}

namespace {
template <typename T>
__global__ void BroadCastCsrColInd(const int *input, const int other_dim,
                                   const int nElm, int *output) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nElm)
    return;
  for (int i = 0; i < other_dim; ++i) {
    output[i + tid * other_dim] = i + input[tid] * other_dim;
  }
}
}

    template <typename T> void EdgeVector<T>::preparePositionAndRelationDataCUDA() {

  if (_option.use_schur) {
    std::vector<std::array<int *, 2>> CompressedCsrColInd;
    CompressedCsrColInd.resize(_option.world_size);
    for (int i = 0; i < _option.world_size; ++i) {
      cudaSetDevice(i);
      const auto edge_num = Memory_Pool::getElmNum(i);

      cudaMalloc(&schurEquationContainer[i].csrRowPtr[0],
                 (num[0] * schurEquationContainer[i].dim[0] + 1) * sizeof(int));
      cudaMalloc(&schurEquationContainer[i].csrRowPtr[1],
                 (num[1] * schurEquationContainer[i].dim[1] + 1) * sizeof(int));
      cudaMemcpyAsync(
          schurEquationContainer[i].csrRowPtr[0], schurCsrRowPtr[i][0].get(),
          (num[0] * schurEquationContainer[i].dim[0] + 1) * sizeof(int),
          cudaMemcpyHostToDevice);
      cudaMemcpyAsync(
          schurEquationContainer[i].csrRowPtr[1], schurCsrRowPtr[i][1].get(),
          (num[1] * schurEquationContainer[i].dim[1] + 1) * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurEquationContainer[i].csrVal[0],
                 schurEquationContainer[i].nnz[0] * sizeof(T)); // hpl
      cudaMalloc(&schurEquationContainer[i].csrColInd[0],
                 schurEquationContainer[i].nnz[0] * sizeof(int));
      {
        const std::size_t entries_in_rows =
            schurEquationContainer[i].nnz[0] / schurEquationContainer[i].dim[1];
        dim3 block(std::min(entries_in_rows, (std::size_t)512));
        dim3 grid((entries_in_rows - 1) / block.x + 1);
        cudaMalloc(&CompressedCsrColInd[i][0], entries_in_rows * sizeof(int));
        cudaMemcpyAsync(CompressedCsrColInd[i][0],
                        schurHEntrance[i].csrColInd_[0].get(),
                        entries_in_rows * sizeof(int), cudaMemcpyHostToDevice);
        ASSERT_CUDA_NO_ERROR();
        BroadCastCsrColInd<T><<<grid, block>>>(
            CompressedCsrColInd[i][0], schurEquationContainer[i].dim[1],
            entries_in_rows, schurEquationContainer[i].csrColInd[0]);
        ASSERT_CUDA_NO_ERROR();
      }

      cudaMalloc(&schurEquationContainer[i].csrVal[1],
                 schurEquationContainer[i].nnz[1] * sizeof(T)); // hlp
      cudaMalloc(&schurEquationContainer[i].csrColInd[1],
                 schurEquationContainer[i].nnz[1] * sizeof(int));
      {
        const std::size_t entries_in_rows =
            schurEquationContainer[i].nnz[1] / schurEquationContainer[i].dim[0];
        dim3 block(std::min(entries_in_rows, (std::size_t)512));
        dim3 grid((entries_in_rows - 1) / block.x + 1);
        cudaMalloc(&CompressedCsrColInd[i][1], entries_in_rows * sizeof(int));
        cudaMemcpyAsync(CompressedCsrColInd[i][1],
                        schurHEntrance[i].csrColInd_[1].get(),
                        entries_in_rows * sizeof(int), cudaMemcpyHostToDevice);
        ASSERT_CUDA_NO_ERROR();
        BroadCastCsrColInd<T><<<grid, block>>>(
            CompressedCsrColInd[i][1], schurEquationContainer[i].dim[0],
            entries_in_rows, schurEquationContainer[i].csrColInd[1]);
        ASSERT_CUDA_NO_ERROR();
      }

      cudaMalloc(&schurEquationContainer[i].csrVal[2],
                 schurEquationContainer[i].nnz[2] * sizeof(T)); // hpp

      cudaMalloc(&schurEquationContainer[i].csrVal[3],
                 schurEquationContainer[i].nnz[3] * sizeof(T)); // hll

      cudaMalloc(&schurEquationContainer[i].g,
                 (num[0] * schurEquationContainer[i].dim[0] +
                  num[1] * schurEquationContainer[i].dim[1]) *
                     sizeof(T));

      cudaMalloc(&schurPositionAndRelationContainer[i].relativePositionCamera,
                 edge_num * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].relativePositionCamera,
          schurRelativePosition[0][i].data(), edge_num * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurPositionAndRelationContainer[i].relativePositionPoint,
                 edge_num * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].relativePositionPoint,
          schurRelativePosition[1][i].data(), edge_num * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurPositionAndRelationContainer[i].absolutePositionCamera,
                 cameraVertexNum * edge_num * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].absolutePositionCamera,
          schurAbsolutePosition[0][i].data(), edge_num * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurPositionAndRelationContainer[i].absolutePositionPoint,
                 pointVertexNum * edge_num * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].absolutePositionPoint,
          schurAbsolutePosition[1][i].data(), edge_num * sizeof(int),
          cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < _option.world_size; ++i) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      cudaFree(CompressedCsrColInd[i][0]);
      cudaFree(CompressedCsrColInd[i][1]);
    }
  } else {
    // TODO: implement this
  }
}

template <typename T> void EdgeVector<T>::cudaPrepareUpdateData() {
  if (_option.use_schur) {
    const auto world_size = Memory_Pool::getWorldSize();
    const auto grad_shape = getGradShape();
    schurStreamLmMemcpy.resize(world_size);
    std::vector<T *> da_ptrs_, da_ptrs_old_;
    da_ptrs_.resize(world_size);
    da_ptrs_old_.resize(world_size);
    for (int i = 0; i < world_size; ++i) {
      cudaSetDevice(i);
      cudaStreamCreateWithFlags(&schurStreamLmMemcpy[i],
                                CU_STREAM_NON_BLOCKING);
      T *da_ptr, *da_ptr_old;
      const auto nElm = Memory_Pool::getElmNum(i);
      cudaMalloc(&da_ptr, nElm * grad_shape * sizeof(T));
      cudaMalloc(&da_ptr_old, nElm * grad_shape * sizeof(T));
      da_ptrs_[i] = da_ptr;
      da_ptrs_old_[i] = da_ptr_old;
    }
    schurDaPtrs.resize(cameraVertexNum + pointVertexNum);
    schurDaPtrsOld.resize(cameraVertexNum + pointVertexNum);
    for (int i = 0; i < cameraVertexNum + pointVertexNum; ++i) {
      schurDaPtrs[i].resize(world_size);
      schurDaPtrsOld[i].resize(world_size);
    }
    for (int i = 0, i_unfixed = 0, offset = 0; i < edges.size(); ++i) {
      if (edges[i][0]->get_Fixed())
        continue;
      for (int j = 0; j < world_size; ++j) {
        const auto nElm = Memory_Pool::getElmNum(j);
        schurDaPtrs[i_unfixed][j] = &da_ptrs_[j][offset * nElm];
        schurDaPtrsOld[i_unfixed][j] = &da_ptrs_old_[j][offset * nElm];
      }
      i_unfixed++;
      const auto &estimation = edges[i][0]->get_Estimation();
      offset += estimation.rows() * estimation.cols();
    }
  } else {
    // TODO: implement this
  }
}

template <typename T> void EdgeVector<T>::deallocateResourceCUDA() {
  if (_option.use_schur) {
    for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      schurPositionAndRelationContainer[i].clearCUDA();
      cudaFree(schurDaPtrs[0][i]);
      cudaFree(schurDaPtrs[1][i]);
      cudaFree(schurDaPtrsOld[0][i]);
      cudaFree(schurDaPtrsOld[1][i]);
      cudaStreamDestroy(schurStreamLmMemcpy[i]);
    }

    for (auto &edge : edges)
      edge.CPU();
  } else {
    // TODO: implement this
  }
}

template <typename T> void EdgeVector<T>::SchurEquationContainer::clearCUDA() {
  for (int i = 0; i < 2; ++i)
    cudaFree(csrRowPtr[i]);
  for (int i = 0; i < 4; ++i)
    cudaFree(csrVal[i]);
  for (int i = 0; i < 2; ++i)
    cudaFree(csrColInd[i]);
  cudaFree(g);
}

template <typename T>
void EdgeVector<T>::PositionAndRelationContainer::clearCUDA() {
  cudaFree(relativePositionCamera);
  cudaFree(relativePositionPoint);
  cudaFree(absolutePositionCamera);
  cudaFree(absolutePositionPoint);
  cudaFree(connectionNumPoint);
}

template class EdgeVector<float>;
template class EdgeVector<double>;
}