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
void vectorMinusVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                          JetVector<T> *out);

template <typename T>
void vectorMultipliesVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                               JetVector<T> *out);

template <typename T>
void vectorDividesVectorCPU(const JetVector<T> &f, const JetVector<T> &g,
                            JetVector<T> *out);

template <typename T>
void jetVectorAddScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorMinusScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void jetVectorMultipliesScalarCPU(const JetVector<T> &f, T g,
                                  JetVector<T> *out);

template <typename T>
void jetVectorDividesScalarCPU(const JetVector<T> &f, T g, JetVector<T> *out);

template <typename T>
void scalarMinusJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out);

template <typename T>
void scalarDividesJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out);

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
