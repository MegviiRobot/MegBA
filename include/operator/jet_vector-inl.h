/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once

namespace MegBA {
namespace CHK {
template <typename T>
inline bool Device_Same(const JetVector<T> &f, const JetVector<T> &g) {
  return f.getDevice() == g.getDevice();
}

template <typename T>
inline void deviceThrow(const JetVector<T> &f, const JetVector<T> &g) {

  if (!Device_Same(f, g))
    throw std::runtime_error("Different _device for item #1 is on " +
                             std::to_string(f.getDevice()) + " item #2 is on " +
                             std::to_string(g.getDevice()));
}

template <typename T>
inline bool Shape_Same(const JetVector<T> &f, const JetVector<T> &g) {
  const auto fGradShape = f.getGradShape();
  const auto gGradShape = g.getGradShape();
  return (fGradShape == 0 || gGradShape == 0 || fGradShape == gGradShape) &&
         f.getElmNum() == g.getElmNum();
}

template <typename T>
inline void shapeThrow(const JetVector<T> &f, const JetVector<T> &g) {
  if (!Shape_Same(f, g))
    throw std::runtime_error("Different shape for gradient of item #1 is " +
                             std::to_string(f.getGradShape()) + " item #2 is " +
                             std::to_string(g.getGradShape()) +
                             ", element number of item #1 is " +
                             std::to_string(f.getElmNum()) + " item #2 is " +
                             std::to_string(g.getElmNum()));
}
}
}

