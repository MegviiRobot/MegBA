/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <edge/base_edge.h>

namespace MegBA {
template <typename T> void EdgeVector<T>::backupDaPtrs() {
  if (_option.useSchur) {
    const auto gradShape = getGradShape();
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schurDaPtrsOld[0][i], schurDaPtrs[0][i],
                      MemoryPool::getElmNum(i) * gradShape * sizeof(T),
                      cudaMemcpyDeviceToDevice, schurStreamLmMemcpy[i]);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

namespace {
template <typename T>
__global__ void broadCastCsrColInd(const int *input, const int other_dim,
                                   const int nElm, int *output) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nElm)
    return;
  for (int i = 0; i < other_dim; ++i) {
    output[i + tid * other_dim] = i + input[tid] * other_dim;
  }
}
}  // namespace

template <typename T> void EdgeVector<T>::preparePositionAndRelationDataCUDA() {
  if (_option.useSchur) {
    std::vector<std::array<int *, 2>> compressedCsrColInd;
    compressedCsrColInd.resize(_option.worldSize);
    for (int i = 0; i < _option.worldSize; ++i) {
      cudaSetDevice(i);
      const auto edgeNum = MemoryPool::getElmNum(i);

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
                 schurEquationContainer[i].nnz[0] * sizeof(T));  // hpl
      cudaMalloc(&schurEquationContainer[i].csrColInd[0],
                 schurEquationContainer[i].nnz[0] * sizeof(int));
      {
        const std::size_t entriesInRows =
            schurEquationContainer[i].nnz[0] / schurEquationContainer[i].dim[1];
        dim3 block(std::min(entriesInRows, (std::size_t)512));
        dim3 grid((entriesInRows - 1) / block.x + 1);
        cudaMalloc(&compressedCsrColInd[i][0], entriesInRows * sizeof(int));
        cudaMemcpyAsync(compressedCsrColInd[i][0],
                        schurHEntrance[i].csrColInd[0].get(),
                        entriesInRows * sizeof(int), cudaMemcpyHostToDevice);
        broadCastCsrColInd<T><<<grid, block>>>(
            compressedCsrColInd[i][0], schurEquationContainer[i].dim[1],
            entriesInRows, schurEquationContainer[i].csrColInd[0]);
      }

      cudaMalloc(&schurEquationContainer[i].csrVal[1],
                 schurEquationContainer[i].nnz[1] * sizeof(T));  // hlp
      cudaMalloc(&schurEquationContainer[i].csrColInd[1],
                 schurEquationContainer[i].nnz[1] * sizeof(int));
      {
        const std::size_t entriesInRows =
            schurEquationContainer[i].nnz[1] / schurEquationContainer[i].dim[0];
        dim3 block(std::min(entriesInRows, (std::size_t)512));
        dim3 grid((entriesInRows - 1) / block.x + 1);
        cudaMalloc(&compressedCsrColInd[i][1], entriesInRows * sizeof(int));
        cudaMemcpyAsync(compressedCsrColInd[i][1],
                        schurHEntrance[i].csrColInd[1].get(),
                        entriesInRows * sizeof(int), cudaMemcpyHostToDevice);
        broadCastCsrColInd<T><<<grid, block>>>(
            compressedCsrColInd[i][1], schurEquationContainer[i].dim[0],
            entriesInRows, schurEquationContainer[i].csrColInd[1]);
      }

      cudaMalloc(&schurEquationContainer[i].csrVal[2],
                 schurEquationContainer[i].nnz[2] * sizeof(T));  // hpp

      cudaMalloc(&schurEquationContainer[i].csrVal[3],
                 schurEquationContainer[i].nnz[3] * sizeof(T));  // hll

      cudaMalloc(&schurEquationContainer[i].g,
                 (num[0] * schurEquationContainer[i].dim[0] +
                  num[1] * schurEquationContainer[i].dim[1]) *
                     sizeof(T));

      cudaMalloc(&schurPositionAndRelationContainer[i].relativePositionCamera,
                 edgeNum * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].relativePositionCamera,
          schurRelativePosition[0][i].data(), edgeNum * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurPositionAndRelationContainer[i].relativePositionPoint,
                 edgeNum * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].relativePositionPoint,
          schurRelativePosition[1][i].data(), edgeNum * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurPositionAndRelationContainer[i].absolutePositionCamera,
                 cameraVertexNum * edgeNum * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].absolutePositionCamera,
          schurAbsolutePosition[0][i].data(), edgeNum * sizeof(int),
          cudaMemcpyHostToDevice);

      cudaMalloc(&schurPositionAndRelationContainer[i].absolutePositionPoint,
                 pointVertexNum * edgeNum * sizeof(int));
      cudaMemcpyAsync(
          schurPositionAndRelationContainer[i].absolutePositionPoint,
          schurAbsolutePosition[1][i].data(), edgeNum * sizeof(int),
          cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < _option.worldSize; ++i) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      cudaFree(compressedCsrColInd[i][0]);
      cudaFree(compressedCsrColInd[i][1]);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T> void EdgeVector<T>::PrepareUpdateDataCUDA() {
  if (_option.useSchur) {
    const auto worldSize = MemoryPool::getWorldSize();
    const auto gradShape = getGradShape();
    schurStreamLmMemcpy.resize(worldSize);
    std::vector<T *> daPtrs, daPtrsOld;
    daPtrs.resize(worldSize);
    daPtrsOld.resize(worldSize);
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      cudaStreamCreateWithFlags(&schurStreamLmMemcpy[i],
                                CU_STREAM_NON_BLOCKING);
      T *daPtr, *daPtrOld;
      const auto nElm = MemoryPool::getElmNum(i);
      cudaMalloc(&daPtr, nElm * gradShape * sizeof(T));
      cudaMalloc(&daPtrOld, nElm * gradShape * sizeof(T));
      daPtrs[i] = daPtr;
      daPtrsOld[i] = daPtrOld;
    }
    schurDaPtrs.resize(cameraVertexNum + pointVertexNum);
    schurDaPtrsOld.resize(cameraVertexNum + pointVertexNum);
    for (int i = 0; i < cameraVertexNum + pointVertexNum; ++i) {
      schurDaPtrs[i].resize(worldSize);
      schurDaPtrsOld[i].resize(worldSize);
    }
    for (int i = 0, iUnfixed = 0, offset = 0; i < edges.size(); ++i) {
      if (edges[i][0]->fixed)
        continue;
      for (int j = 0; j < worldSize; ++j) {
        const auto nElm = MemoryPool::getElmNum(j);
        schurDaPtrs[iUnfixed][j] = &daPtrs[j][offset * nElm];
        schurDaPtrsOld[iUnfixed][j] = &daPtrsOld[j][offset * nElm];
      }
      iUnfixed++;
      const auto &estimation = edges[i][0]->getEstimation();
      offset += estimation.rows() * estimation.cols();
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T> void EdgeVector<T>::deallocateResourceCUDA() {
  if (_option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
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
    // TODO(Jie Ren): implement this
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
}  // namespace MegBA
