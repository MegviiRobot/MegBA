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
namespace function {
template <typename T>
void vectorAddVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                         JetVector<T> *out);

template <typename T>
void vectorMinusVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                           JetVector<T> *out);

template <typename T>
void vectorMultipliesVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                                JetVector<T> *out);

template <typename T>
void vectorDividesVectorCUDA(const JetVector<T> &f, const JetVector<T> &g,
                             JetVector<T> *out);

template <typename T>
void jetVectorAddScalarCUDA(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorMinusScalarCUDA(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorMultipliesScalarCUDA(const JetVector<T> &f, T g,
                                   JetVector<T> *out);

template <typename T>
void scalarMinusJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void scalarDividesJetVectorCUDA(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void absJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void cosJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void sinJetVectorCUDA(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out);

template <typename T>
void sqrtJetVectorCUDA(const MegBA::JetVector<T> &f,
                         MegBA::JetVector<T> *out);
}  // namespace function
}  // namespace math
}  // namespace MegBA
