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
    void EdgeVector<T>::update_schur(const std::vector<T *> &delta_x_ptr) {
        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(schur_stream_LM_memcpy_[i]);
        }

        const double LR = 1.;
        const auto camera_dim = edges[0][0]->get_Grad_Shape();
        const auto camera_num = vertices_set_ptr_->find(edges[0][0]->kind())->second.size();
        const auto point_dim = edges[1][0]->get_Grad_Shape();

//        cudaSetDevice(0);
//        PRINT_DMEMORY(schur_da_ptrs[0][0], 5, T);
        // TODO: merge into method 'solve_Linear'

        for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
            cudaSetDevice(i);
            const auto edge_num = Memory_Pool::getElmNum(i);
            dim3 block(std::min((decltype(edge_num)) 256, edge_num));
            dim3 grid((edge_num - 1) / block.x + 1);
            ASSERT_CUDA_NO_ERROR();
            problem::CUDA::update_delta_x_two_Vertices<T><<<grid, block>>>(
                    delta_x_ptr[i], schur_position_and_relation_container_[i].absolute_position_camera, schur_position_and_relation_container_[i].absolute_position_point,
                    LR,
                    camera_dim, point_dim,
                    camera_num, edge_num,
                    schur_da_ptrs[0][i], schur_da_ptrs[1][i]);
            ASSERT_CUDA_NO_ERROR();
        }
//        cudaSetDevice(0);
//        PRINT_DMEMORY(schur_da_ptrs[0][0], 5, T);
    }

    template class EdgeVector<double>;
    template class EdgeVector<float>;
}
