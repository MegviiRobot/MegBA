/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <memory>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <Common.h>
#include <operator/JetVector.h>
#include <vertex/BaseVertex.h>
#include <problem/HEntrance.h>

namespace MegBA {
namespace {
enum EdgeKind { ONE, ONE_CAMERA_ONE_POINT, TWO_CAMERA, MULTI };
}
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

  void appendVertex(BaseVertex<T> &vertex);

  bool existVertex(const BaseVertex<T> &vertex) const;

  virtual JVD<T> forward() = 0;

  template <typename PlainMatrix>
  void setMeasurement(PlainMatrix &&measurement) {
    _measurement = std::forward<PlainMatrix>(measurement);
  };

  JVD<T> const &getMeasurement() const { return _edge.getMeasurement(); };

  template <typename PlainMatrix>
  void setInformation(PlainMatrix &&information) {
    _information = std::forward<PlainMatrix>(information);
  };

protected:
  const BaseEdgeWrapper<T> &getVertices() { return _edge; }
};

template <typename T> class EdgeVector {
  friend BaseProblem<T>;

  const ProblemOption &_option;
  const std::vector<SchurHEntrance<T>> &schurHEntrance;
  // total number for each vertex kind
  std::unique_ptr<int[]> num{nullptr};
  std::vector<std::vector<int>> absolutePosition;
  // kind -> world_size
  std::vector<std::vector<std::vector<int>>> schurRelativePosition;
  std::vector<std::vector<std::vector<int>>> schurAbsolutePosition;
  std::vector<BaseEdge<T> *> edgesPtr;
  std::size_t nameHash{};
  std::vector<Vertex_Vector<T>> edges;
  std::unique_ptr<int[]> csrRowPtr{nullptr};
  std::vector<std::array<std::unique_ptr<int[]>, 2>> schurCsrRowPtr;
  // kind -> world_size -> ptr
  std::vector<std::vector<T *>> schurDaPtrs;
  std::vector<std::vector<T *>> schurDaPtrsOld;
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

  EdgeVector(const ProblemOption &option, const std::vector<SchurHEntrance<T>> &schurHEntrance);

  struct SchurEquationContainer {
    explicit SchurEquationContainer(const device_t &device) : _device(device){};

    SchurEquationContainer(const SchurEquationContainer &container)
        : _device(container._device){};

    ~SchurEquationContainer() { clear(); }

    void clear();

    void clearCUDA();
    const device_t &_device;
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
    std::array<int, 2> dim{0, 0};
  };

  struct PositionAndRelationContainer {
    explicit PositionAndRelationContainer(const device_t &device)
        : _device(device){};

    ~PositionAndRelationContainer() { clear(); }

    void clear();

    void clearCUDA();

    const device_t &_device;
    int *relativePositionCamera{nullptr}, *relativePositionPoint{nullptr};
    int *absolutePositionCamera{nullptr}, *absolutePositionPoint{nullptr};
    int *connectionNumPoint{nullptr};
  };

  void backupDaPtrs();

  void rollback();

  std::vector<Vertex_Vector<T>> &getEdges() { return edges; };

  const std::vector<Vertex_Vector<T>> &getEdges() const { return edges; };

  JVD<T> &getEstimation(int i) { return edges[i].get_Jet_Estimation(); };

  const JVD<T> &getEstimation(int i) const {
    return edges[i].get_Jet_Estimation();
  };

  JVD<T> &getObservation(int i) { return edges[i].get_Jet_Observation(); };

  const JVD<T> &getObservation(int i) const {
    return edges[i].get_Jet_Observation();
  };

  JVD<T> &getMeasurement() { return jetMeasurement; };

  const JVD<T> &getMeasurement() const { return jetMeasurement; };

  JVD<T> &getInformation() { return jetInformation; };

  const JVD<T> &getInformation() const { return jetInformation; };

  bool tryPushBack(BaseEdge<T> &edge);

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

  void buildLinearSystemSchur(JVD<T> &jetEstimation);

  void buildLinearSystemSchurCUDA(const JVD<T> &jetEstimation);

  void updateSchur(const std::vector<T *> &deltaXPtr);

  void rebindDaPtrs();

  std::vector<SchurEquationContainer> schurEquationContainer;

  std::vector<PositionAndRelationContainer> schurPositionAndRelationContainer;
};
}
