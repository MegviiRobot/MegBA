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
#include <Macro.h>

namespace MegBA {
namespace {
enum EdgeKind_t { ONE, ONE_CAMERA_ONE_POINT, TWO_CAMERA, MULTI };
}
template <typename T> class EdgeVector;

template <typename T> class BaseEdge : public std::vector<BaseVertex<T> *> {
  typedef std::vector<BaseVertex<T> *> parent;
  using Measurement_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  BaseEdgeWrapper<T> Edge;
  Measurement_t measurement_;
  Measurement_t information_;

  Measurement_t &getMeasurement_() { return measurement_; };

  const Measurement_t &getMeasurement_() const { return measurement_; };

  Measurement_t &getInformation_() { return information_; };

  const Measurement_t &getInformation_() const { return information_; };

  void bindEdgeVector(const EdgeVector<T> *EV) { Edge.bindEdgeVector(EV); };

public:
  virtual ~BaseEdge() = default;

  friend EdgeVector<T>;

  void appendVertex(BaseVertex<T> &vertex);

  bool existVertex(const BaseVertex<T> &vertex) const {
    return std::find(parent::begin(), parent::end(), &vertex) != parent::end();
  }

  virtual JVD<T> forward() = 0;

  void setMeasurement(const Measurement_t &measurement) {
    measurement_ = measurement;
  };

  JVD<T> const &getMeasurement() const { return Edge.getMeasurement(); };

  void setInformation(const Measurement_t &information) {
    information_ = information;
  };

protected:
  const BaseEdgeWrapper<T> &getVertices() { return Edge; }
};

template <typename T> class EdgeVector {
private:
  const ProblemOption_t &option_;
  const std::vector<SchurHEntrance_t<T>> &schur_H_entrance_;
  // total number for each vertex kind
  std::unique_ptr<int[]> num{nullptr};
  std::vector<std::vector<int>> absolute_position;
  // kind -> world_size
  std::vector<std::vector<std::vector<int>>> schur_relative_position;
  std::vector<std::vector<std::vector<int>>> schur_absolute_position;
  std::vector<BaseEdge<T> *> edges_ptr;
  std::size_t name_hash{};
  std::vector<Vertex_Vector<T>> edges;
  std::unique_ptr<int[]> csrRowPtr{nullptr};
  std::vector<std::array<std::unique_ptr<int[]>, 2>> schur_csrRowPtr;
  // kind -> world_size -> ptr
  std::vector<std::vector<T *>> schur_da_ptrs;
  std::vector<std::vector<T *>> schur_da_ptrs_old;
  unsigned int Hessian_shape_{0};
  unsigned int camera_vertex_num{0};
  unsigned int point_vertex_num{0};
  JVD<T> Jet_measurement_;
  JVD<T> Jet_information_;
  std::vector<cudaStream_t> schur_stream_LM_memcpy_{};
  EdgeKind_t EdgeKind_{};

  void DecideEdgeKind() {
    if (camera_vertex_num + point_vertex_num == 1)
      EdgeKind_ = ONE;
    else if (camera_vertex_num == 1 && point_vertex_num == 1)
      EdgeKind_ = ONE_CAMERA_ONE_POINT;
    else if (camera_vertex_num == 2 && point_vertex_num == 0)
      EdgeKind_ = TWO_CAMERA; // MULTI;
    else
      EdgeKind_ = MULTI;
  }

public:
  struct SchurEquationContainer {
    explicit SchurEquationContainer(const device_t &device) : device_(device){};

    SchurEquationContainer(const SchurEquationContainer &container)
        : device_(container.device_){};

    ~SchurEquationContainer() { clear(); }

    void clear() {
      switch (device_) {
      case CPU_t: {
        for (int i = 0; i < 2; ++i)
          free(csrRowPtr[i]);
        for (int i = 0; i < 4; ++i)
          free(csrVal[i]);
        for (int i = 0; i < 2; ++i)
          free(csrColInd[i]);
        free(g);
        break;
      }
      case CUDA_t: {
        freeCUDA();
        break;
      }
      }
      for (int i = 0; i < 2; ++i)
        csrRowPtr[i] = nullptr;
      for (int i = 0; i < 4; ++i)
        csrVal[i] = nullptr;
      for (int i = 0; i < 2; ++i)
        csrColInd[i] = nullptr;
      g = nullptr;
      nnz[0] = 0;
      nnz[1] = 0;
      nnz[2] = 0;
      nnz[3] = 0;
      dim[0] = 0;
      dim[1] = 0;
    };
    void freeCUDA();
    const device_t &device_;
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
    std::array<int, 2> dim{0, 0};
  };

  struct PositionAndRelationContainer {
    explicit PositionAndRelationContainer(const device_t &device)
        : device_(device){};

    ~PositionAndRelationContainer() {
      switch (device_) {
      case CPU_t: {
        free(relative_position_camera);
        free(relative_position_point);
        free(absolute_position_camera);
        free(absolute_position_point);
        free(connection_num_point);
        break;
      }
      case CUDA_t: {
        freeCUDA();
        break;
      }
      }
      relative_position_camera = nullptr;
      relative_position_point = nullptr;
      absolute_position_camera = nullptr;
      absolute_position_point = nullptr;
      connection_num_point = nullptr;
    }
    void freeCUDA();
    const device_t &device_;
    int *relative_position_camera{nullptr}, *relative_position_point{nullptr};
    int *absolute_position_camera{nullptr}, *absolute_position_point{nullptr};
    int *connection_num_point{nullptr};
  };

  EdgeVector() = delete;

  EdgeVector(const ProblemOption_t &option, const HEntrance_t<T> &H_entrance,
             const std::vector<SchurHEntrance_t<T>> &schur_H_entrance)
      : option_{option}, schur_H_entrance_{schur_H_entrance},
        schur_csrRowPtr(option.world_size) {
    schur_equation_container_.reserve(option.world_size);
    for (int i = 0; i < option.world_size; ++i) {
      schur_equation_container_.emplace_back(option.device);
      schur_position_and_relation_container_.emplace_back(option.device);
    }
  };

  void backupDaPtrs();

  void rollback() {
    if (option_.use_schur) {
      schur_da_ptrs.swap(schur_da_ptrs_old);
    } else {
      // TODO: implement this
    }
    rebindDaPtrs();
    backupDaPtrs();
  };

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

  JVD<T> &getMeasurement() { return Jet_measurement_; };

  const JVD<T> &getMeasurement() const { return Jet_measurement_; };

  JVD<T> &getInformation() { return Jet_information_; };

  const JVD<T> &getInformation() const { return Jet_information_; };

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

  void cudaPrepareUpdateData();

  JVD<T> forward();

  // TODO: not const now, fix later
  void bindVerticesSet(
      std::unordered_map<VertexKind_t, std::set<BaseVertex<T> *>> const
          *vertices_set_ptr) {
    vertices_set_ptr_ = vertices_set_ptr;
  };

  void setHessianShape(unsigned int Hessian_Shape) {
    Hessian_shape_ = Hessian_Shape;
  };

  void fitDevice();

  void buildLinearSystemSchur(JVD<T> &Jet_Estimation);

  void buildLinearSystemSchurCUDA(const JVD<T> &Jet_Estimation);

  void updateSchur(const std::vector<T *> &delta_x_ptr);

  void rebindDaPtrs() {
    int vertex_kind_idx_unfixed = 0;
    for (auto &vertex_vector : edges) {
      if (vertex_vector[0]->get_Fixed())
        continue;
      auto &Jet_estimation = vertex_vector.get_Jet_Estimation();
      auto &Jet_observation = vertex_vector.get_Jet_Observation();

      const auto world_size = Memory_Pool::getWorldSize();
      for (int i = 0; i < vertex_vector[0]->get_Estimation().size(); ++i) {
        // bind da_ptr_ for CUDA
        if (option_.use_schur) {
          std::vector<T *> da_ptrs_;
          da_ptrs_.resize(world_size);
          for (int k = 0; k < world_size; ++k) {
            da_ptrs_[k] = &schur_da_ptrs[vertex_kind_idx_unfixed][k]
                                        [i * Memory_Pool::getElmNum(k)];
          }
          Jet_estimation(i).bind_da_ptr(std::move(da_ptrs_));
        } else {
          // TODO: implement this
        }
      }
      vertex_kind_idx_unfixed++;
    }
  }

  // TODO: not const now, fix later
  std::unordered_map<VertexKind_t, std::set<BaseVertex<T> *>> const
      *vertices_set_ptr_;

  std::vector<SchurEquationContainer> schur_equation_container_;

  std::vector<PositionAndRelationContainer>
      schur_position_and_relation_container_;
};
}
