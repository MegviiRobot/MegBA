/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <cusparse_v2.h>

#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "operator/jet_vector.h"
#include "problem/hessian_entrance.h"
#include "vertex/base_vertex.h"

namespace MegBA {
enum EdgeKind { ONE, ONE_CAMERA_ONE_POINT, TWO_CAMERA, MULTI };

template <typename T>
class BaseEdge {
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PlainMatrix;

  std::vector<BaseVertex<T> *> m_data;
  EdgeWrapper<T> _edgeWrapper;
  PlainMatrix _measurement;
  PlainMatrix _information;

 public:
  std::size_t size() { return m_data.size(); }
  BaseVertex<T> *&operator[](std::size_t n) { return m_data[n]; }
  BaseVertex<T> *const &operator[](std::size_t n) const { return m_data[n]; }

  void appendVertex(BaseVertex<T> *vertex);

  bool existVertex(const BaseVertex<T> &vertex) const;

  virtual JVD<T> forward() = 0;

  template <typename PlainMatrix>
  void setMeasurement(PlainMatrix &&measurement) {
    _measurement = std::forward<PlainMatrix>(measurement);
  }

  template <typename PlainMatrix>
  void setInformation(PlainMatrix &&information) {
    _information = std::forward<PlainMatrix>(information);
  }

  const PlainMatrix &_getMeasurement() const { return _measurement; }

  const PlainMatrix &_getInformation() const { return _information; }

  void bindEdgeVector(const EdgeVector<T> *ev) {
    _edgeWrapper.bindEdgeVector(ev);
  }

 protected:
  const EdgeWrapper<T> &getVertices() { return _edgeWrapper; }

  JVD<T> const &getMeasurement() const { return _edgeWrapper.getMeasurement(); }
};

template <typename T>
class EdgeVector {
  const ProblemOption &option;
  // kind -> worldSize
  struct PositionContainer {
    std::vector<int *> relativePosition{};
    std::vector<int *> absolutePosition{};
  };

  std::vector<PositionContainer> positionContainers;
  std::vector<BaseEdge<T> *> edgesPtr;
  std::size_t nameHash{};
  std::vector<VertexVector<T>> edges;
  // kind -> worldSize -> ptr
  std::vector<std::vector<T *>> schurValueDevicePtrs;
  std::vector<std::vector<T *>> schurValueDevicePtrsOld;
  unsigned int cameraVertexNum{0};
  unsigned int pointVertexNum{0};
  JVD<T> jetMeasurement;
  JVD<T> jetInformation;
  EdgeKind edgeKind{};

  void decideEdgeKind();

 public:
  EdgeVector() = delete;

  explicit EdgeVector(const ProblemOption &option);

  void backup() const;

  void rollback() const;

  std::vector<VertexVector<T>> &getVertexVectors() { return edges; }

  const std::vector<VertexVector<T>> &getVertexVectors() const { return edges; }

  JVD<T> &getEstimation(int i) { return edges[i].getJVEstimation(); }

  const JVD<T> &getEstimation(int i) const {
    return edges[i].getJVEstimation();
  }

  JVD<T> &getObservation(int i) { return edges[i].getJVObservation(); }

  const JVD<T> &getObservation(int i) const {
    return edges[i].getJVObservation();
  }

  auto &getMeasurement() { return jetMeasurement; }

  const auto &getMeasurement() const { return jetMeasurement; }

  auto &getInformation() { return jetInformation; }

  auto &getInformation() const { return jetInformation; }

  auto &getPositionContainers() { return positionContainers; }

  const auto &getPositionContainers() const { return positionContainers; }

  bool tryPushBack(BaseEdge<T> &edge);

  void eraseVertex(const BaseVertex<T> &vertex);

  unsigned int getGradShape() const;

  void allocateResource();

  void deallocateResource();

  void deallocateResourceCUDA();

  void allocateResourceCUDA();

  JVD<T> forward() const;

  void fitDevice();

  void buildLinearSystem(const JVD<T> &jetEstimation,
                         const BaseLinearSystem<T> &linearSystem) const;

  void buildLinearSystemCUDA(const JVD<T> &jetEstimation,
                             const BaseLinearSystem<T> &linearSystem) const;

  void update(const BaseLinearSystem<T> &linearSystem) const;

  void bindCUDAGradPtrs();

  void buildPositionContainer(
      const std::vector<HessianEntrance<T>> &hessianEntrance);
};
}  // namespace MegBA
