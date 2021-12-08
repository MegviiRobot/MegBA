/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "edge/BaseEdge.h"
#include <Macro.h>

namespace MegBA {
    namespace problem {
        namespace CUDA {
            template<typename T>
            __global__ void update_delta_x_two_Vertices(const T *delta_x, const int *absolute_position_camera,
                                                        const int *absolute_position_point, T LR,
                                                        const int camera_dim, const int point_dim, const int camera_num,
                                                        const int edge_num,
                                                        T *camera_x, T *point_x) {
                const unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
                if (ix >= edge_num)
                    return;
                //     ix : index in edges
                //     absolute_position_camera[ix] : index in cameras
                //     absolute_position_camera[ix] * camera_dim : starting position of its camera_block(dim = camera_dim)
                //     threadIdx.y : offset of its camera_block
                unsigned int idx = ix;
                for(int i = 0; i < camera_dim; ++i) {
                     camera_x[idx] += LR * delta_x[absolute_position_camera[ix] * camera_dim + i];
                    idx += edge_num;
                }

                for(int i = 0; i < point_dim; ++i) {
                    point_x[idx - edge_num * camera_dim] += LR * delta_x[absolute_position_point[ix] * point_dim + i + camera_num * camera_dim];
                    idx += edge_num;
                }
            }
        }
    }

    template<typename T>
    void EdgeVector<T>::updateSchur(const std::vector<T *> &deltaXPtr) {
        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(schurStreamLmMemcpy[i]);
        }

        const double LR = 1.;
        const auto camera_dim = edges[0][0]->getGradShape();
        const auto camera_num =
            verticesSetPtr->find(edges[0][0]->kind())->second.size();
        const auto point_dim = edges[1][0]->getGradShape();

        // TODO: merge into method 'solve_Linear'

        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            const auto edge_num = Memory_Pool::getElmNum(i);
            dim3 block(std::min((decltype(edge_num)) 256, edge_num));
            dim3 grid((edge_num - 1) / block.x + 1);
            ASSERT_CUDA_NO_ERROR();
            problem::CUDA::update_delta_x_two_Vertices<T><<<grid, block>>>(
                deltaXPtr[i],
                schurPositionAndRelationContainer[i].absolutePositionCamera,
                schurPositionAndRelationContainer[i].absolutePositionPoint,
                    LR,
                    camera_dim, point_dim,
                    camera_num, edge_num,
                schurDaPtrs[0][i], schurDaPtrs[1][i]);
            ASSERT_CUDA_NO_ERROR();
        }
//        cudaSetDevice(0);
//        PRINT_DMEMORY(schur_da_ptrs[0][0], 5, T);
    }

    template<typename T>
    void EdgeVector<T>::rebindDaPtrs() {
      int vertexKindIdxUnfixed = 0;
      for (auto &vertexVector : edges) {
        if (vertexVector[0]->get_Fixed())
          continue;
        auto &jetEstimation = vertexVector.get_Jet_Estimation();
        auto &jetObservation = vertexVector.get_Jet_Observation();

        const auto worldSize = Memory_Pool::getWorldSize();
        for (int i = 0; i < vertexVector[0]->get_Estimation().size(); ++i) {
          // bind da_ptr_ for CUDA
          if (_option.use_schur) {
            std::vector<T *> daPtrs;
            daPtrs.resize(worldSize);
            for (int k = 0; k < worldSize; ++k) {
              daPtrs[k] = &schurDaPtrs[vertexKindIdxUnfixed][k]
                                      [i * Memory_Pool::getElmNum(k)];
            }
            jetEstimation(i).bind_da_ptr(std::move(daPtrs));
          } else {
            // TODO: implement this
          }
        }
        vertexKindIdxUnfixed++;
      }
    }
    template class EdgeVector<double>;
    template class EdgeVector<float>;
}
