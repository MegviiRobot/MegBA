/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/BaseProblem.h"
#include <thread>
#include <condition_variable>
#include <Macro.h>

namespace MegBA {
    namespace {
        template<typename SchurHEntrance_t>
        void InternalBuildRandomAccess(int i, SchurHEntrance_t & schur_H_entrance) {
            const auto &H_entrance_block_matrix_ = schur_H_entrance.nra_[i];
            const auto dim_other = schur_H_entrance.dim_[1^i];
            auto &csrRowPtr_kind = schur_H_entrance.csrRowPtr_[i];
            auto &csrColInd_kind = schur_H_entrance.csrColInd_[i];
            csrRowPtr_kind.reset(new int[H_entrance_block_matrix_.size() * schur_H_entrance.dim_[i] + 1]);
            csrColInd_kind.reset(new int[schur_H_entrance.counter * schur_H_entrance.dim_[i]]);
            csrRowPtr_kind[0] = 0;
            std::size_t row_counter{0};
            std::size_t nnz_counter{0};
            auto &H_entrance_ra_block_matrix_ = schur_H_entrance.ra_[i];
            H_entrance_ra_block_matrix_.clear();
            H_entrance_ra_block_matrix_.reserve(H_entrance_block_matrix_.size());
            // row
            for (const auto &row_iter : H_entrance_block_matrix_) {
                const auto &H_entrance_block_row_ = row_iter.second;
                const auto row_size = H_entrance_block_row_.size();
                typename std::decay_t<decltype(H_entrance_ra_block_matrix_)>::value_type H_entrance_ra_block_row_;
                H_entrance_ra_block_row_.reserve(row_size);
                // col
                for (const auto &col : H_entrance_block_row_) {
                    H_entrance_ra_block_row_.push_back(col);
                    csrColInd_kind[nnz_counter] = col->absolutePosition;
                    nnz_counter++;
                }
                for (int j = 0; j < schur_H_entrance.dim_[i]; ++j) {
                    csrRowPtr_kind[row_counter + 1] = csrRowPtr_kind[row_counter] + row_size * dim_other;
                    ++row_counter;
                    if (j > 0) {
                        memcpy(&csrColInd_kind[nnz_counter], &csrColInd_kind[nnz_counter - row_size], row_size * sizeof(int));
                        nnz_counter += row_size;
                    }
                }
                H_entrance_ra_block_matrix_.push_back(std::move(H_entrance_ra_block_row_));
            }
            schur_H_entrance.nnz_in_E = csrRowPtr_kind[row_counter];
        }
    }// namespace

    template<typename T>
    void SchurHEntrance<T>::BuildRandomAccess() {
        // camera and point
        std::vector<std::thread> threads;
        threads.emplace_back(std::thread{InternalBuildRandomAccess<SchurHEntrance<T>>, 0, std::ref(*this)});
        threads.emplace_back(std::thread{InternalBuildRandomAccess<SchurHEntrance<T>>, 1, std::ref(*this)});
        for (auto &thread : threads)
            thread.join();
    }

    template<typename T>
    HEntrance_t<T>::HEntrance_t() : nra_{std::make_pair(CAMERA, std::array<BlockMatrix_t, 2>{}), std::make_pair(POINT, std::array<BlockMatrix_t, 2>{})},
                    ra_{std::make_pair(CAMERA, std::array<BlockMatrixRA_t, 2>{}), std::make_pair(POINT, std::array<BlockMatrixRA_t, 2>{})} {};

    template<typename T>
    void HEntrance_t<T>::BuildRandomAccess() {
        // camera and point
        for (const auto &entrance_iter : nra_) {
            auto &H_entrance_ra_kind_ = ra_[entrance_iter.first];
            // pp & pl || ll & lp
            for (int i = 0; i < entrance_iter.second.size(); ++i) {
                const auto &H_entrance_block_matrix_ = entrance_iter.second[i];
                auto &H_entrance_ra_block_matrix_ = H_entrance_ra_kind_[i];
                H_entrance_ra_block_matrix_.clear();
                H_entrance_ra_block_matrix_.reserve(H_entrance_block_matrix_.size());
                // row
                for (const auto &row_iter : H_entrance_block_matrix_) {
                    const auto &H_entrance_block_row_ = row_iter.second;
                    typename std::decay_t<decltype(H_entrance_ra_block_matrix_)>::value_type H_entrance_ra_block_row_;
                    H_entrance_ra_block_row_.reserve(H_entrance_block_row_.size());
                    // col
                    for (const auto &col : H_entrance_block_row_)
                        H_entrance_ra_block_row_.push_back(col);
                    H_entrance_ra_block_matrix_.push_back(std::move(H_entrance_ra_block_row_));
                }
            }
        }
    }

    template<typename T>
    BaseProblem<T>::BaseProblem(ProblemOption option) : option_(option) {
        if (option.N != -1 && option.nElm != -1)
            Memory_Pool::reset_Pool(option.N, option.nElm, sizeof(T), option.worldSize);
        if (option.useSchur) {
            schur_ws_.split_size_ = option.nElm / option.worldSize + 1;
            schur_ws_.working_device_ = 0;
            schur_ws_.schur_H_entrance_.resize(option.worldSize);
            schur_ws_.schur_H_entrance_.shrink_to_fit();
        }
    }

    template<typename T>
    const device_t &BaseProblem<T>::get_Device() const {
        return option_.device;
    }

    template<typename T>
    void BaseProblem<T>::append_Vertex(int ID, BaseVertex<T> &vertex) {
        append_Vertex(ID, &vertex);
    }

    template<typename T>
    void BaseProblem<T>::append_Vertex(int ID, BaseVertex<T> *vertex) {
        vertices.insert(std::make_pair(ID, vertex));
    }

    template<typename T>
    void BaseProblem<T>::append_Edge(BaseEdge<T> &edge) {
        bool success = edges.tryPushBack(edge);
        if (!success) {
          edges.tryPushBack(edge);
        }
        for (int vertex_idx = edge.size() - 1; vertex_idx >= 0; --vertex_idx) {
            auto vertex = edge[vertex_idx];
            auto kind = vertex->kind();
            auto find = vertices_sets.find(kind);
            if (find == vertices_sets.end())
                vertices_sets.emplace(vertex->kind(), std::set<BaseVertex<T> *>{vertex});
            else
                find->second.emplace(vertex);

            if (option_.useSchur) {
                for (int i = 0; i < option_.worldSize; ++i) {
                    auto &working_schur_H_entrance = schur_H_entrance_[i];
                    working_schur_H_entrance.dim_[kind] = vertex->getGradShape();
                    auto &connection_block_matrix = working_schur_H_entrance.nra_[kind];
                    auto connection_find = connection_block_matrix.find(vertex);
                    if (connection_find == connection_block_matrix.end()) {
                        connection_find = connection_block_matrix.emplace(vertex, typename HEntrance_t<T>::BlockRow_t{}).first;
                    }
                    if (i == schur_ws_.working_device_) {
                        connection_find->second.emplace(edge[1^ vertex_idx]);
                    }
                }
            } else {
              // TODO: implement this
            }
        }
        if (option_.useSchur) {
            auto &working_schur_H_entrance = schur_H_entrance_[schur_ws_.working_device_];
            working_schur_H_entrance.counter++;
            if (working_schur_H_entrance.counter >= schur_ws_.split_size_)
                schur_ws_.working_device_++;
        } else {
          // TODO: implement this
        }
    }

    template<typename T>
    void BaseProblem<T>::append_Edge(BaseEdge<T> *edge) {
        append_Edge(*edge);
    }

    template<typename T> BaseVertex<T> &BaseProblem<T>::get_Vertex(int ID) {
        auto vertex = vertices.find(ID);
        if (vertex == vertices.end())
            throw std::runtime_error("The ID " + std::to_string(ID) + " does not exist in the current graph.");
        return *vertex->second;
    }

    template<typename T>
    const BaseVertex<T> &BaseProblem<T>::get_Vertex(int ID) const {
        const auto vertex = vertices.find(ID);
        if (vertex == vertices.end())
            throw std::runtime_error("The ID " + std::to_string(ID) + " does not exist in the current graph.");
        return *vertex->second;
    }

    template<typename T>
    void BaseProblem<T>::eraseVertex(int ID) {
        const auto vertex = vertices.find(ID);
        if (vertex == vertices.end())
            throw std::runtime_error("The ID " + std::to_string(ID) + " does not exist in the current graph.");
        edges.eraseVertex(*vertex->second);
        vertices.erase(ID);

        for (auto &vertices_set : vertices_sets) {
            vertices_set.second.erase(vertex->second);
        }
    }

    template<typename T>
    void BaseProblem<T>::DeallocateResource() {
      edges.deallocateResource();
        switch (option_.device) {
            case CUDA_t:
              cudaDeallocateResource();
                break;
            default:
                throw std::runtime_error("Not Implemented.");
        }
    }

    template<typename T>
    unsigned int BaseProblem<T>::get_Hessian_Shape() const {
        unsigned int Grad_Shape = 0;
        for (const auto &vertex_set_pair : vertices_sets) {
            const auto &vertex_set = vertex_set_pair.second;
            BaseVertex<T> const *vertex_ptr = *vertex_set.begin();
            Grad_Shape += vertex_set.size() * vertex_ptr->getGradShape();
        }
        return Grad_Shape;
    }

    template<typename T>
    void BaseProblem<T>::PrepareUpdateData() {
        switch (option_.device) {
            case CUDA_t:
              cudaPrepareUpdateData();
                break;
            default:
                throw std::runtime_error("Not Implemented.");
        }
    }

    template<typename T>
    void BaseProblem<T>::MakeVertices() {
        Hessian_shape_ = get_Hessian_Shape();
        PrepareUpdateData();
        set_absolute_position();
        if (option_.useSchur) {
            std::vector<std::thread> threads;
            for (auto &schur_H_entrance : schur_H_entrance_) {
                threads.emplace_back(std::thread{[&](){schur_H_entrance.BuildRandomAccess();}});
            }
            for (auto &thread : threads) {
                thread.join();
            }
        } else {
          // TODO: implement this
        }

        edges.verticesSetPtr = &vertices_sets;
        edges.allocateResourcePre();
        edges.makeVertices();
        edges.allocateResourcePost();
        edges.fitDevice();

    }

    template<typename T>
    void BaseProblem<T>::set_absolute_position() {
        T *hx_ptr = new T[Hessian_shape_];
        std::size_t entrance_bias{0};
        for (auto &set_pair : vertices_sets) {
            auto &set = set_pair.second;
            int absolute_position_counter = 0;
            bool fixed = (*set.begin())->get_Fixed();
            std::size_t nnz_each_item = (*set.begin())->get_Estimation().rows() * (*set.begin())->get_Estimation().cols();
            for (auto &vertex : set) {
                vertex->absolutePosition = absolute_position_counter;
                ++absolute_position_counter;
                if (!fixed) {
                    memcpy(&hx_ptr[entrance_bias], vertex->get_Estimation().data(), nnz_each_item * sizeof(T));
                    entrance_bias += nnz_each_item;
                }
            }
        }
        if (option_.useSchur) {
            for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
                cudaSetDevice(i);
                cudaMemcpyAsync(schur_x_ptr[i], hx_ptr, Hessian_shape_ * sizeof(T), cudaMemcpyHostToDevice);
            }
        } else {
          // TODO: implement this
        }
        delete[] hx_ptr;
    }

    template<typename T>
    bool BaseProblem<T>::SolveLinear(double tol, double solver_refuse_ratio, std::size_t max_iter) {
        switch (option_.device) {
            case CUDA_t:
                return cudaSolveLinear(tol, solver_refuse_ratio, max_iter);
            default:
                throw std::runtime_error("Not Implemented.");
        }
    }

    template<typename T>
    void BaseProblem<T>::WriteBack() {
        T *hx_ptr = new T[Hessian_shape_];
        std::size_t entrance_bias{0};
        if (option_.useSchur) {
            cudaSetDevice(0);
            cudaMemcpy(hx_ptr, schur_x_ptr[0], Hessian_shape_ * sizeof(T), cudaMemcpyDeviceToHost);
        } else {
          // TODO: implement this
        }
        for (auto &vertex_set_pair : vertices_sets) {
            auto &vertex_set = vertex_set_pair.second;
            if ((*vertex_set.begin())->get_Fixed())
                continue;
            const auto nnz_each_item = (*vertex_set.begin())->get_Estimation().rows() * (*vertex_set.begin())->get_Estimation().cols();
            for (auto &vertex : vertex_set) {
                memcpy(vertex->get_Estimation().data(), &hx_ptr[entrance_bias], nnz_each_item * sizeof(T));
                entrance_bias += nnz_each_item;
            }
        }
        delete[] hx_ptr;
    }

    template class BaseProblem<double>;
    template class BaseProblem<float>;
}