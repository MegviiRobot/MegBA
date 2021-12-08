/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <vertex/BaseVertex.h>

namespace MegBA {
    struct ProblemOption_t {
        bool use_schur{true};
        int world_size{1};
        device_t device{CUDA_t};
        int N{-1};
        int64_t nElm{-1};
    };

    template<typename T>
    struct SchurHEntrance_t {
        // first is camera
        using BlockRow_t = std::set<BaseVertex<T> *>;
        using BlockMatrix_t = std::map<BaseVertex<T> *, BlockRow_t>;
        std::array<BlockMatrix_t, 2> nra_;
        using BlockRowRA_t = std::vector<BaseVertex<T> *>;
        using BlockMatrixRA_t = std::vector<BlockRowRA_t>;
        std::array<BlockMatrixRA_t, 2> ra_;
        std::array<std::unique_ptr<int[]>, 2> csrRowPtr_;
        std::array<std::unique_ptr<int[]>, 2> csrColInd_;
        std::size_t counter{0};
        std::array<int, 2> dim_{};
        std::size_t nnz_in_E{};

        SchurHEntrance_t() = default;

        void BuildRandomAccess();
    };

    template<typename T>
    struct HEntrance_t {
        using BlockRow_t = std::set<BaseVertex<T> *>;
        using BlockMatrix_t = std::map<BaseVertex<T> *, BlockRow_t>;
        std::unordered_map<VertexKind, std::array<BlockMatrix_t, 2>> nra_;
        using BlockRowRA_t = std::vector<BaseVertex<T> *>;
        using BlockMatrixRA_t = std::vector<BlockRowRA_t>;
        std::unordered_map<VertexKind, std::array<BlockMatrixRA_t, 2>> ra_;

        HEntrance_t();

        void BuildRandomAccess();
    };
}
