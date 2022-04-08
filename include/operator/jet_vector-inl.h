/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once

namespace MegBA {
namespace Check {
template <typename T>
inline bool deviceSame(const JetVector<T> &f, const JetVector<T> &g) {
  return f.getDevice() == g.getDevice();
}

template <typename T>
inline void deviceThrow(const JetVector<T> &f, const JetVector<T> &g) {
  if (!deviceSame(f, g))
    throw std::runtime_error("Different _device for item #1 is on " +
                             std::to_string(f.getDevice()) + " item #2 is on " +
                             std::to_string(g.getDevice()));
}

template <typename T>
inline bool shapeSame(const JetVector<T> &f, const JetVector<T> &g) {
  const auto fGradShape = f.getGradShape();
  const auto gGradShape = g.getGradShape();
  return (fGradShape == 0 || gGradShape == 0 || fGradShape == gGradShape) &&
         f.getItemNum() == g.getItemNum();
}

template <typename T>
inline void shapeThrow(const JetVector<T> &f, const JetVector<T> &g) {
  if (!shapeSame(f, g))
    throw std::runtime_error("Different shape for gradient of item #1 is " +
                             std::to_string(f.getGradShape()) + " item #2 is " +
                             std::to_string(g.getGradShape()) +
                             ", element number of item #1 is " +
                             std::to_string(f.getItemNum()) + " item #2 is " +
                             std::to_string(g.getItemNum()));
}
}  // namespace Check
}  // namespace MegBA
