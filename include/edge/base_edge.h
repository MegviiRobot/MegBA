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

  const ProblemOption &_option;
  const std::vector<SchurHessianEntrance<T>> &schurHessianEntrance;
  // total number for each vertex kind
  std::unique_ptr<int[]> num{nullptr};
  std::vector<std::vector<int>> absolutePosition;
  // kind -> worldSize
  std::vector<std::vector<std::vector<int>>> schurRelativePosition;
  std::vector<std::vector<std::vector<int>>> schurAbsolutePosition;
  std::vector<BaseEdge<T> *> edgesPtr;
  std::size_t nameHash{};
  std::vector<VertexVector<T>> edges;
  std::unique_ptr<int[]> csrRowPtr{nullptr};
  std::vector<std::array<std::unique_ptr<int[]>, 2>> schurCsrRowPtr;
  // kind -> worldSize -> ptr
  std::vector<std::vector<T *>> schurValueDevicePtrs;
  std::vector<std::vector<T *>> schurValueDevicePtrsOld;
  unsigned int cameraVertexNum{0};
  unsigned int pointVertexNum{0};
  JVD<T> jetMeasurement;
  JVD<T> jetInformation;
  std::vector<cudaStream_t> schurStreamLmMemcpy{};
  EdgeKind edgeKind{};
  std::unordered_map<VertexKind, std::set<BaseVertex<T> *>> const
      *verticesSetPtr;

  void decideEdgeKind();

 public:
  EdgeVector() = delete;

  EdgeVector(const ProblemOption &option,
             const std::vector<SchurHessianEntrance<T>> &schurHessianEntrance);

  struct SchurEquationContainer {
    explicit SchurEquationContainer(const Device &device) : _device(device) {}

    SchurEquationContainer(const SchurEquationContainer &container)
        : _device(container._device) {}

    ~SchurEquationContainer() { clear(); }

    void clear();

    void clearCUDA();
    const Device &_device;
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
    std::array<int, 2> dim{0, 0};
  };

  struct PositionAndRelationContainer {
    explicit PositionAndRelationContainer(const Device &device)
        : _device(device) {}

    ~PositionAndRelationContainer() { clear(); }

    void clear();

    void clearCUDA();

    const Device &_device;
    int *relativePositionCamera{nullptr}, *relativePositionPoint{nullptr};
    int *absolutePositionCamera{nullptr}, *absolutePositionPoint{nullptr};
    int *connectionNumPoint{nullptr};
  };

  void backupValueDevicePtrs();

  void rollback();

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

  JVD<T> forward();

  void fitDevice();

  void buildLinearSystemSchur(const JVD<T> &jetEstimation);

  void buildLinearSystemSchurCUDA(const JVD<T> &jetEstimation);

  void updateSchur(const std::vector<T *> &deltaXPtr);

  void bindCUDAGradPtrs();

  std::vector<SchurEquationContainer> schurEquationContainer;

  std::vector<PositionAndRelationContainer> schurPositionAndRelationContainer;
};
}  // namespace MegBA
