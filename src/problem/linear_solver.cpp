/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <cholmod.h>

#include <functional>
#include <memory>

namespace MegAutoBA {
void CholmodSolverImpl(const double *csrVal, const int *csrColInd,
                       const int *csrRowPtr, const double *b, std::size_t rows,
                       std::size_t cols, std::size_t nnz, bool ReAnalyze,
                       double *delta_x) {
  cholmod_sparse Sparse{};
  Sparse.stype = 1;
  Sparse.itype = CHOLMOD_INT;
  Sparse.xtype = CHOLMOD_REAL;
  Sparse.dtype = CHOLMOD_DOUBLE;
  Sparse.sorted = 1;
  Sparse.packed = 1;

  Sparse.i = const_cast<int *>(csrColInd);
  Sparse.x = const_cast<double *>(csrVal);
  Sparse.p = const_cast<int *>(csrRowPtr);
  Sparse.z = nullptr;
  Sparse.nz = nullptr;
  Sparse.nzmax = nnz;
  Sparse.ncol = cols;
  Sparse.nrow = rows;

  static std::shared_ptr<cholmod_common> Common{new cholmod_common};
  cholmod_start(Common.get());
  Common->nmethods = 1;
  Common->method[0].ordering = CHOLMOD_AMD;
  Common->supernodal = CHOLMOD_AUTO;

  static std::unique_ptr<cholmod_factor, std::function<void(cholmod_factor *&)>>
      Factor{cholmod_analyze(&Sparse, Common.get()),
             [Common = Common](cholmod_factor *&factor) mutable {
               cholmod_free_factor(&factor, Common.get());
               Common.reset();
             }};
  if (ReAnalyze) {
    Factor.reset(cholmod_analyze(&Sparse, Common.get()));
    Factor.get_deleter() = [Common = Common](cholmod_factor *&factor) mutable {
      cholmod_free_factor(&factor, Common.get());
      Common.reset();
    };
  }
  cholmod_factorize(&Sparse, Factor.get(), Common.get());
  cholmod_dense B;
  B.nrow = B.d = Sparse.ncol;
  B.ncol = 1;
  B.x = const_cast<double *>(b);
  B.xtype = CHOLMOD_REAL;
  B.dtype = CHOLMOD_DOUBLE;

  auto X = (cholmod_dense *)malloc(sizeof(cholmod_dense));
  static auto Y = (cholmod_dense *)malloc(sizeof(cholmod_dense));
  static std::shared_ptr<bool> Y_changed{new bool{false}};

  static std::shared_ptr<cholmod_dense> Y_guard{
      Y, [Common = Common, Y_changed = Y_changed](cholmod_dense *&y) mutable {
        if (!(*Y_changed)) cholmod_free_dense(&y, Common.get());
        Y_changed.reset();
        Common.reset();
      }};

  X->nrow = X->d = Sparse.nrow;
  X->ncol = 1;
  X->x = delta_x;
  X->xtype = CHOLMOD_REAL;
  X->dtype = CHOLMOD_DOUBLE;
  cholmod_solve2(CHOLMOD_A, Factor.get(), &B, nullptr, &X, nullptr, &Y, nullptr,
                 Common.get());
  if (Y != Y_guard.get()) {
    *Y_changed = true;
    Y_guard.reset(
        Y, [Common = Common, Y_changed = Y_changed](cholmod_dense *&y) mutable {
          if (!(*Y_changed)) cholmod_free_dense(&y, Common.get());
          Y_changed.reset();
          Common.reset();
        });
    *Y_changed = false;
  }
  free(X);
}
}  // namespace MegAutoBA