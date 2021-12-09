/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <vector>
#include <map>
#include <memory>
#include <Common.h>
#include <edge/BaseEdge.h>
#include <vertex/BaseVertex.h>
#include <problem/HEntrance.h>
#include <cusparse_v2.h>

namespace MegBA {
    template<typename T>
    class BaseProblem {
        void DeallocateResource();

        void cudaDeallocateResource();

        unsigned int get_Hessian_Shape() const;

        void MakeVertices();

        const ProblemOption option_;

        std::size_t Hessian_shape_{0};
        std::unordered_map<int, BaseVertex<T> *> vertices{};
        std::unordered_map<VertexKind, std::set<BaseVertex<T> *>> vertices_sets{};
        struct SchurWorkingSpace_t {
            // first: working index, second: body
            std::size_t split_size_{0};
            int working_device_{0};
            std::vector<SchurHEntrance<T>> schur_H_entrance_;
        } schur_ws_{};
        std::vector<SchurHEntrance<T>> &schur_H_entrance_{schur_ws_.schur_H_entrance_};
        EdgeVector<T> edges{option_, schur_H_entrance_};

        std::vector<T *> schur_x_ptr{nullptr};
        std::vector<T *> schur_delta_x_ptr{nullptr};
        std::vector<T *> schur_delta_x_ptr_backup{nullptr};
    public:
        explicit BaseProblem(ProblemOption option= ProblemOption{});

        ~BaseProblem() = default;

        const device_t &getDevice() const;

        void append_Vertex(int ID, BaseVertex<T> &vertex);

        void append_Vertex(int ID, BaseVertex<T> *vertex);

        void append_Edge(BaseEdge<T> &edge);

        void append_Edge(BaseEdge<T> *edge);

        BaseVertex<T> &get_Vertex(int ID);

        const BaseVertex<T> &get_Vertex(int ID) const;

        void eraseVertex(int ID);

        void set_absolute_position();

        bool SolveLinear(double tol, double solver_refuse_ratio, std::size_t max_iter);

        bool cudaSolveLinear(double tol, double solver_refuse_ratio, std::size_t max_iter);

        void PrepareUpdateData();

        void WriteBack();

        void cudaPrepareUpdateData();

        void SolveLM(int iter, double solver_tol, double solver_refuse_ratio, int solver_max_iter, double tau, double epsilon1, double epsilon2);

        void BackupLM();

        void RollbackLM();
    };
}
