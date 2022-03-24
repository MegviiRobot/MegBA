/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "vertex/base_vertex.h"

namespace MegBA {
template <typename T>
struct HessianEntrance {
  // first is camera
  using BlockRow = std::set<BaseVertex<T> *>;
  using BlockMatrix = std::map<BaseVertex<T> *, BlockRow>;
  std::array<BlockMatrix, 2> nra{};
  using BlockRowRA = std::vector<BaseVertex<T> *>;
  using BlockMatrixRA = std::vector<BlockRowRA>;
  std::array<BlockMatrixRA, 2> ra{};
  std::size_t counter{0};
  std::array<int, 2> dim{};

  HessianEntrance() = default;

  void buildRandomAccess();
};
}  // namespace MegBA
