/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <Eigen/Sparse>
#include <cassert>

#if DEBUG
#define PRINT_DMEMORY(d_ptr, nItem, T)                                     \
  do {                                                                     \
    T *__ptr = new T[nItem];                                               \
    cudaMemcpy(__ptr, d_ptr, (nItem) * sizeof(T), cudaMemcpyDeviceToHost); \
    std::cout << #d_ptr << ": ";                                           \
    for (std::size_t __i = 0; __i < (nItem); ++__i)                        \
      std::cout << __ptr[__i] << " ";                                      \
    std::cout << std::endl;                                                \
    delete[] __ptr;                                                        \
  } while (false)

#define PRINT_DMEMORY_ONE_ELEMENT(d_ptr, idx, T)                            \
  do {                                                                      \
    cudaDeviceSynchronize();                                                \
    T element;                                                              \
    cudaMemcpy(&element, &(d_ptr)[idx], sizeof(T), cudaMemcpyDeviceToHost); \
    std::cout << #idx << "-th element in " << #d_ptr << ": " << element     \
              << std::endl;                                                 \
  } while (false)

#define PRINT_DMEMORY_SEGMENT(d_ptr, start, nItem, T)                 \
  do {                                                                \
    assert((nItem) > 0);                                              \
    cudaDeviceSynchronize();                                          \
    T *__ptr = new T[nItem];                                          \
    cudaMemcpy(__ptr, &(d_ptr)[start], nItem * sizeof(T),             \
               cudaMemcpyDeviceToHost);                               \
    std::cout << (nItem) << " elements from " << (start) << "-th in " \
              << #d_ptr << ": " << std::endl;                         \
    for (std::size_t __i = 0; __i < (nItem); ++__i)                   \
      std::cout << __ptr[__i] << " ";                                 \
    std::cout << std::endl;                                           \
    delete[] __ptr;                                                   \
  } while (false)

#define ASSERT_CUDA_NO_ERROR()                                 \
  do {                                                         \
    cudaDeviceSynchronize();                                   \
    auto error = cudaGetLastError();                           \
    if (error != cudaSuccess)                                  \
      throw std::runtime_error(                                \
          std::string("Failed at: ") + std::string(__FILE__) + \
          std::string(", line: ") + std::to_string(__LINE__) + \
          std::string(", with error string: ") +               \
          std::string(cudaGetErrorString(error)));             \
  } while (false)

#define PRINT_DCSR(csrVal, csrColInd, csrRowPtr, rows_num, T)              \
  do {                                                                     \
    int *h_csrColInd, *h_csrRowPtr = new int[(rows_num) + 1];              \
    cudaMemcpy(h_csrRowPtr, csrRowPtr, ((rows_num) + 1) * sizeof(int),     \
               cudaMemcpyDeviceToHost);                                    \
    std::size_t nnz = h_csrRowPtr[rows_num];                               \
    h_csrColInd = new int[nnz];                                            \
    T *h_csrVal = new T[nnz];                                              \
    cudaMemcpy(h_csrColInd, csrColInd, nnz * sizeof(int),                  \
               cudaMemcpyDeviceToHost);                                    \
    cudaMemcpy(h_csrVal, csrVal, nnz * sizeof(T), cudaMemcpyDeviceToHost); \
    Eigen::Map<Eigen::SparseMatrix<T>, Eigen::RowMajor> SpMat{             \
        (Eigen::Index)(rows_num),                                          \
        (Eigen::Index)(rows_num),                                          \
        (Eigen::Index)nnz,                                                 \
        h_csrRowPtr,                                                       \
        h_csrColInd,                                                       \
        h_csrVal};                                                         \
    std::cout << #csrVal << ":\n";                                         \
    std::cout << SpMat << std::endl;                                       \
    delete[] h_csrColInd;                                                  \
    delete[] h_csrRowPtr;                                                  \
    delete[] h_csrVal;                                                     \
  } while (false);

#define ASSERT_HOST_NO_MEM_ERROR()         \
  do {                                     \
    std::vector<int> v;                    \
    v.resize(2);                           \
    auto __tmp0 = new T[2];                \
    auto __tmp1 = new T[2];                \
    memcpy(__tmp0, __tmp1, 2 * sizeof(T)); \
    delete[] __tmp0;                       \
    delete[] __tmp1;                       \
  } while (false)
#else
#define ASSERT_CUDA_NO_ERROR()

#define PRINT_DMEMORY(d_ptr, nItem, T)

#define PRINT_DMEMORY_ONE_ELEMENT(d_ptr, idx, T)

#define PRINT_DMEMORY_SEGMENT(d_ptr, start, nItem, T)

#define PRINT_DCSR(csrVal_, csrColInd_, csrRowPtr_, rows_num, T)

#define ASSERT_HOST_NO_MEM_ERROR()
#endif
