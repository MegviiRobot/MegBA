/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <cusparse_v2.h>
#include <utility>
#include <set>
#include <vector>
#include <memory>
#include <unordered_map>
#include "common.h"
#include "operator/jet_vector.h"
#include "vertex/base_vertex.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T>
struct SchurLMLinearSystemManager;

enum EdgeKind { ONE, ONE_CAMERA_ONE_POINT, TWO_CAMERA, MULTI };

template <typename T> class EdgeVector;

template <typename T> class BaseEdge : public std::vector<BaseVertex<T> *> {
  typedef std::vector<BaseVertex<T> *> parent;

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PlainMatrix;

  friend EdgeVector<T>;

  BaseEdgeWrapper<T> _edge;
  PlainMatrix _measurement;
  PlainMatrix _information;

 public:
  virtual ~BaseEdge() = default;

  void appendVertex(BaseVertex<T> *vertex);

  bool existVertex(const BaseVertex<T> &vertex) const;

  virtual JVD<T> forward() = 0;

  template <typename PlainMatrix>
  void setMeasurement(PlainMatrix &&measurement) {
    _measurement = std::forward<PlainMatrix>(measurement);
  }

  JVD<T> const &getMeasurement() const { return _edge.getMeasurement(); }

  template <typename PlainMatrix>
  void setInformation(PlainMatrix &&information) {
    _information = std::forward<PlainMatrix>(information);
  }

 protected:
  const BaseEdgeWrapper<T> &getVertices() { return _edge; }
};

template <typename T> class EdgeVector {
  friend BaseProblem<T>;

  const ProblemOption &option;
  // kind -> worldSize
  std::vector<std::vector<std::vector<int>>> schurRelativePosition;
  std::vector<std::vector<std::vector<int>>> schurAbsolutePosition;
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
  std::vector<cudaStream_t> schurStreamLmMemcpy{};
  EdgeKind edgeKind{};

  void decideEdgeKind();

 public:
  EdgeVector() = delete;

  explicit EdgeVector(const ProblemOption &option);

  struct PositionAndRelationContainer {
    explicit PositionAndRelationContainer(const Device &device)
        : device(device) {}

    ~PositionAndRelationContainer() { clear(); }

    void clear();

    void clearCUDA();

    const Device &device;
    int *relativePositionCamera{nullptr}, *relativePositionPoint{nullptr};
    int *absolutePositionCamera{nullptr}, *absolutePositionPoint{nullptr};
  };

  void backup() const;

  void rollback() const;

  std::vector<VertexVector<T>> &getEdges() { return edges; }

  const std::vector<VertexVector<T>> &getEdges() const { return edges; }

  JVD<T> &getEstimation(int i) { return edges[i].getJVEstimation(); }

  const JVD<T> &getEstimation(int i) const {
    return edges[i].getJVEstimation();
  }

  JVD<T> &getObservation(int i) { return edges[i].getJVObservation(); }

  const JVD<T> &getObservation(int i) const {
    return edges[i].getJVObservation();
  }

  JVD<T> &getMeasurement() { return jetMeasurement; }

  const JVD<T> &getMeasurement() const { return jetMeasurement; }

  JVD<T> &getInformation() { return jetInformation; }

  const JVD<T> &getInformation() const { return jetInformation; }

  bool tryPushBack(BaseEdge<T> *edge);

  void eraseVertex(const BaseVertex<T> &vertex);

  unsigned int getGradShape() const;

  void allocateResourcePre();

  void allocateResourcePost();

  void preparePositionAndRelationDataCUDA();

  void deallocateResource();

  void deallocateResourceCUDA();

  void makeVertices();

  void makeSchurVertices();

  void PrepareUpdateDataCUDA();

  JVD<T> forward() const;

  void fitDevice();

  void buildLinearSystemSchur(const JVD<T> &jetEstimation, const BaseLinearSystemManager<T> &linearSystemManager) const;

  void buildLinearSystemSchurCUDA(const JVD<T> &jetEstimation, const BaseLinearSystemManager<T> &linearSystemManager) const;

  void updateSchur(const SchurLMLinearSystemManager<T> &linearSystemManager) const;

  void bindCUDAGradPtrs();

  std::vector<PositionAndRelationContainer> schurPositionAndRelationContainer;
};
}  // namespace MegBA
