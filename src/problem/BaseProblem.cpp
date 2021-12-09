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
        template<typename SchurHEntrance>
        void internalBuildRandomAccess(int i, SchurHEntrance &schurHEntrance) {
            const auto &hEntranceBlockMatrix = schurHEntrance.nra[i];
            const auto dimOther = schurHEntrance.dim[1^i];
            auto &csrRowPtrKind = schurHEntrance.csrRowPtr[i];
            auto &csrColIndKind = schurHEntrance.csrColInd[i];
            csrRowPtrKind.reset(new int[hEntranceBlockMatrix.size() * schurHEntrance.dim[i] + 1]);
            csrColIndKind.reset(new int[schurHEntrance.counter * schurHEntrance.dim[i]]);
            csrRowPtrKind[0] = 0;
            std::size_t rowCounter{0};
            std::size_t nnzCounter{0};
            auto &HEntranceRaBlockMatrix = schurHEntrance.ra[i];
            HEntranceRaBlockMatrix.clear();
            HEntranceRaBlockMatrix.reserve(hEntranceBlockMatrix.size());
            // row
            for (const auto &rowIter : hEntranceBlockMatrix) {
                const auto &hEntranceBlockRow = rowIter.second;
                const auto rowSize = hEntranceBlockRow.size();
                typename std::decay_t<decltype(HEntranceRaBlockMatrix)>::value_type HEntranceRaBlockRow;
                HEntranceRaBlockRow.reserve(rowSize);
                // col
                for (const auto &col : hEntranceBlockRow) {
                  HEntranceRaBlockRow.push_back(col);
                    csrColIndKind[nnzCounter] = col->absolutePosition;
                    nnzCounter++;
                }
                for (int j = 0; j < schurHEntrance.dim[i]; ++j) {
                  csrRowPtrKind[rowCounter + 1] =
                      csrRowPtrKind[rowCounter] + rowSize * dimOther;
                    ++rowCounter;
                    if (j > 0) {
                        memcpy(&csrColIndKind[nnzCounter], &csrColIndKind[nnzCounter - rowSize], rowSize * sizeof(int));
                        nnzCounter += rowSize;
                    }
                }
                HEntranceRaBlockMatrix.push_back(std::move(HEntranceRaBlockRow));
            }
            schurHEntrance.nnzInE = csrRowPtrKind[rowCounter];
        }
    }// namespace

    template<typename T>
    void SchurHEntrance<T>::buildRandomAccess() {
        // camera and point
        std::vector<std::thread> threads;
        threads.emplace_back(std::thread{internalBuildRandomAccess<SchurHEntrance<T>>, 0, std::ref(*this)});
        threads.emplace_back(std::thread{internalBuildRandomAccess<SchurHEntrance<T>>, 1, std::ref(*this)});
        for (auto &thread : threads)
            thread.join();
    }

    template<typename T>
    BaseProblem<T>::BaseProblem(ProblemOption option) : option(option) {
        if (option.N != -1 && option.nElm != -1)
          MemoryPool::resetPool(option.N, option.nElm, sizeof(T),
                                 option.worldSize);
        if (option.useSchur) {
          schurWS.splitSize = option.nElm / option.worldSize + 1;
          schurWS.workingDevice = 0;
          schurWS.schurHEntrance.resize(option.worldSize);
          schurWS.schurHEntrance.shrink_to_fit();
        }
    }

    template<typename T>
    const device_t &BaseProblem<T>::getDevice() const {
        return option.device;
    }

    template<typename T>
    void BaseProblem<T>::appendVertex(int ID, BaseVertex<T> *vertex) {
        vertices.insert(std::make_pair(ID, vertex));
    }

    template<typename T>
    void BaseProblem<T>::appendEdge(BaseEdge<T> *edge) {
      bool success = edges.tryPushBack(edge);
      if (!success) {
        edges.tryPushBack(edge);
      }
      for (int vertex_idx = edge->size() - 1; vertex_idx >= 0; --vertex_idx) {
        auto vertex = edge->operator[](vertex_idx);
        auto kind = vertex->kind();
        auto find = verticesSets.find(kind);
        if (find == verticesSets.end())
          verticesSets.emplace(vertex->kind(), std::set<BaseVertex<T> *>{vertex});
        else
          find->second.emplace(vertex);

        if (option.useSchur) {
          for (int i = 0; i < option.worldSize; ++i) {
            auto &working_schur_H_entrance = schurWS.schurHEntrance[i];
            working_schur_H_entrance.dim[kind] = vertex->getGradShape();
            auto &connection_block_matrix = working_schur_H_entrance.nra[kind];
            auto connection_find = connection_block_matrix.find(vertex);
            if (connection_find == connection_block_matrix.end()) {
              connection_find = connection_block_matrix.emplace(vertex, typename SchurHEntrance<T>::BlockRow{}).first;
            }
            if (i == schurWS.workingDevice) {
              connection_find->second.emplace(edge->operator[](1^ vertex_idx));
            }
          }
        } else {
          // TODO: implement this
        }
      }
      if (option.useSchur) {
        auto &working_schur_H_entrance = schurWS.schurHEntrance[schurWS.workingDevice];
        working_schur_H_entrance.counter++;
        if (working_schur_H_entrance.counter >= schurWS.splitSize)
          schurWS.workingDevice++;
      } else {
        // TODO: implement this
      }
    }

    template<typename T> BaseVertex<T> &BaseProblem<T>::getVertex(int ID) {
        auto vertex = vertices.find(ID);
        if (vertex == vertices.end())
            throw std::runtime_error("The ID " + std::to_string(ID) + " does not exist in the current graph.");
        return *vertex->second;
    }

    template<typename T>
    const BaseVertex<T> &BaseProblem<T>::getVertex(int ID) const {
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

        for (auto &vertices_set : verticesSets) {
            vertices_set.second.erase(vertex->second);
        }
    }

    template<typename T>
    void BaseProblem<T>::deallocateResource() {
      edges.deallocateResource();
        switch (option.device) {
            case CUDA_t:
              deallocateResourceCUDA();
                break;
            default:
                throw std::runtime_error("Not Implemented.");
        }
    }

    template<typename T>
    unsigned int BaseProblem<T>::getHessianShape() const {
        unsigned int Grad_Shape = 0;
        for (const auto &vertex_set_pair : verticesSets) {
            const auto &vertex_set = vertex_set_pair.second;
            BaseVertex<T> const *vertex_ptr = *vertex_set.begin();
            Grad_Shape += vertex_set.size() * vertex_ptr->getGradShape();
        }
        return Grad_Shape;
    }

    template<typename T>
    void BaseProblem<T>::prepareUpdateData() {
        switch (option.device) {
            case CUDA_t:
              prepareUpdateDataCUDA();
                break;
            default:
                throw std::runtime_error("Not Implemented.");
        }
    }

    template<typename T>
    void BaseProblem<T>::makeVertices() {
      hessianShape = getHessianShape();
      prepareUpdateData();
        setAbsolutePosition();
        if (option.useSchur) {
            std::vector<std::thread> threads;
            for (auto &schur_H_entrance : schurWS.schurHEntrance) {
                threads.emplace_back(std::thread{[&](){ schur_H_entrance.buildRandomAccess();}});
            }
            for (auto &thread : threads) {
                thread.join();
            }
        } else {
          // TODO: implement this
        }

        edges.verticesSetPtr = &verticesSets;
        edges.allocateResourcePre();
        edges.makeVertices();
        edges.allocateResourcePost();
        edges.fitDevice();

    }

    template<typename T>
    void BaseProblem<T>::setAbsolutePosition() {
        T *hx_ptr = new T[hessianShape];
        std::size_t entrance_bias{0};
        for (auto &set_pair : verticesSets) {
            auto &set = set_pair.second;
            int absolute_position_counter = 0;
            bool fixed = (*set.begin())->fixed;
            std::size_t nnz_each_item = (*set.begin())->getEstimation().rows() *
                (*set.begin())->getEstimation().cols();
            for (auto &vertex : set) {
                vertex->absolutePosition = absolute_position_counter;
                ++absolute_position_counter;
                if (!fixed) {
                    memcpy(&hx_ptr[entrance_bias], vertex->getEstimation().data(), nnz_each_item * sizeof(T));
                    entrance_bias += nnz_each_item;
                }
            }
        }
        if (option.useSchur) {
            for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
                cudaSetDevice(i);
                cudaMemcpyAsync(schurXPtr[i], hx_ptr, hessianShape * sizeof(T), cudaMemcpyHostToDevice);
            }
        } else {
          // TODO: implement this
        }
        delete[] hx_ptr;
    }

    template<typename T>
    bool BaseProblem<T>::solveLinear(double tol, double solverRefuseRatio, std::size_t maxIter) {
        switch (option.device) {
            case CUDA_t:
                return solveLinearCUDA(tol, solverRefuseRatio, maxIter);
            default:
                throw std::runtime_error("Not Implemented.");
        }
    }

    template<typename T>
    void BaseProblem<T>::writeBack() {
        T *hx_ptr = new T[hessianShape];
        std::size_t entrance_bias{0};
        if (option.useSchur) {
            cudaSetDevice(0);
            cudaMemcpy(hx_ptr, schurXPtr[0], hessianShape * sizeof(T), cudaMemcpyDeviceToHost);
        } else {
          // TODO: implement this
        }
        for (auto &vertex_set_pair : verticesSets) {
            auto &vertex_set = vertex_set_pair.second;
            if ((*vertex_set.begin())->fixed)
                continue;
            const auto nnz_each_item =
                (*vertex_set.begin())->getEstimation().rows() *
                (*vertex_set.begin())->getEstimation().cols();
            for (auto &vertex : vertex_set) {
                memcpy(vertex->getEstimation().data(), &hx_ptr[entrance_bias], nnz_each_item * sizeof(T));
                entrance_bias += nnz_each_item;
            }
        }
        delete[] hx_ptr;
    }

    template class BaseProblem<double>;
    template class BaseProblem<float>;
}