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
void vectorAddVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                         JetVector<T> *out);

template <typename T>
void vectorSubVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                         JetVector<T> *out);

template <typename T>
void vectorMulVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                         JetVector<T> *out);

template <typename T>
void vectorDivVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                         JetVector<T> *out);

template <typename T>
void jetVectorAddScalarCUDA(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorSubScalarCUDA(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorMulScalarCUDA(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void scalarSubJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void scalarDivJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void absJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void cosJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void sinJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void sqrtJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);
}  // namespace impl
}  // namespace math
}  // namespace MegBA
