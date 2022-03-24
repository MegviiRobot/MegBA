/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "common.h"

namespace MegBA {
namespace math {
namespace impl {
template <typename T>
void vectorAddVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                        JetVector<T> *out);

template <typename T>
void vectorSubVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                        JetVector<T> *out);

template <typename T>
void vectorMulVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                        JetVector<T> *out);

template <typename T>
void vectorDivVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                        JetVector<T> *out);

template <typename T>
void jetVectorAddScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorSubScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorMulScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorDivScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void scalarSubJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void scalarDivJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void absJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void cosJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void sinJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void sqrtJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);
}  // namespace impl
}  // namespace math
}  // namespace MegBA
