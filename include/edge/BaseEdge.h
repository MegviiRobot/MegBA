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

  Measurement_t &get_Measurement_() { return measurement_; };

  const Measurement_t &get_Measurement_() const { return measurement_; };

  Measurement_t &get_Information_() { return information_; };

  const Measurement_t &get_Information_() const { return information_; };

  void bind_Edge_Vector(const EdgeVector<T> *EV) {
    Edge.bind_Edge_Vector(EV);
  };

public:
  virtual ~BaseEdge() = default;

  friend EdgeVector<T>;

  void append_Vertex(BaseVertex<T> &vertex);

  bool exist_Vertex(const BaseVertex<T> &vertex) const {
    return std::find(parent::begin(), parent::end(), &vertex) != parent::end();
  }

  virtual JVD<T> forward() = 0;

  void set_Measurement(const Measurement_t &measurement) {
    measurement_ = measurement;
  };

  JVD<T> const &get_Measurement() const {
    return Edge.get_Measurement();
  };

  void set_Information(const Measurement_t &information) {
    information_ = information;
  };

protected:
  const BaseEdgeWrapper<T> &get_Vertices() { return Edge; }
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
  struct Schur_Equation_Container {
    explicit Schur_Equation_Container(const device_t &device)
        : device_(device){};

    Schur_Equation_Container(const Schur_Equation_Container &container)
        : device_(container.device_){};

    ~Schur_Equation_Container() { clear(); }

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
        free_CUDA();
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
    void free_CUDA();
    const device_t &device_;
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
    std::array<int, 2> dim{0, 0};
  };

  struct Position_and_Relation_Container {
    explicit Position_and_Relation_Container(const device_t &device)
        : device_(device){};

    ~Position_and_Relation_Container() {
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
        free_CUDA();
        break;
      }
      }
      relative_position_camera = nullptr;
      relative_position_point = nullptr;
      absolute_position_camera = nullptr;
      absolute_position_point = nullptr;
      connection_num_point = nullptr;
    }
    void free_CUDA();
    const device_t &device_;
    int *relative_position_camera{nullptr}, *relative_position_point{nullptr};
    int *absolute_position_camera{nullptr}, *absolute_position_point{nullptr};
    int *connection_num_point{nullptr};
  };

  EdgeVector() = delete;

  EdgeVector(const ProblemOption_t &option, const HEntrance_t<T> &H_entrance,
              const std::vector<SchurHEntrance_t<T>> &schur_H_entrance)
      : option_{option},
        schur_H_entrance_{schur_H_entrance}, schur_csrRowPtr(option.world_size) {
    schur_equation_container_.reserve(option.world_size);
    for (int i = 0; i < option.world_size; ++i) {
      schur_equation_container_.emplace_back(option.device);
      schur_position_and_relation_container_.emplace_back(option.device);
    }
  };

  void backup_da_ptrs();

  void rollback() {
    if (option_.use_schur) {
      schur_da_ptrs.swap(schur_da_ptrs_old);
    } else {
      // TODO: implement this
    }
    rebind_da_ptrs();
    backup_da_ptrs();
  };

  std::vector<Vertex_Vector<T>> &get_Edges() { return edges; };

  const std::vector<Vertex_Vector<T>> &get_Edges() const { return edges; };

  JVD<T> &get_Jet_Estimation_i(int i) {
    return edges[i].get_Jet_Estimation();
  };

  const JVD<T> &get_Jet_Estimation_i(int i) const {
    return edges[i].get_Jet_Estimation();
  };

  JVD<T> &get_Jet_Observation_i(int i) {
    return edges[i].get_Jet_Observation();
  };

  const JVD<T> &get_Jet_Observation_i(int i) const {
    return edges[i].get_Jet_Observation();
  };

  JVD<T> &get_Measurement() { return Jet_measurement_; };

  const JVD<T> &get_Measurement() const {
    return Jet_measurement_;
  };

  JVD<T> &get_Information() { return Jet_information_; };

  const JVD<T> &get_Information() const {
    return Jet_information_;
  };

  bool try_push_back(BaseEdge<T> &edge);

  void erase_Vertex(const BaseVertex<T> &vertex);

  unsigned int get_Grad_Shape() const;

  void allocate_resource_pre();

  void allocate_resource_post();

  void prepare_position_and_relation_data_CUDA();

  void DeallocateResource();

  void cudaDeallocateResource();

  void make_Vertices();

  void make_schur_Vertices();

  void prepare_update_data_CUDA();

  JVD<T> forward();

  // TODO: not const now, fix later
  void bind_Vertices_Set(
      std::unordered_map<VertexKind_t, std::set<BaseVertex<T> *>> const
          *vertices_set_ptr) {
    vertices_set_ptr_ = vertices_set_ptr;
  };

  void set_Hessian_Shape(unsigned int Hessian_Shape) {
    Hessian_shape_ = Hessian_Shape;
  };

  void fit_Device();

  void make_H_and_g_schur(JVD<T> &Jet_Estimation);

  void make_H_and_g_schur_CUDA(const JVD<T> &Jet_Estimation);

  void update_schur(const std::vector<T *> &delta_x_ptr);

  void rebind_da_ptrs() {
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
            da_ptrs_[k] = &schur_da_ptrs[vertex_kind_idx_unfixed][k][i * Memory_Pool::getElmNum(k)];
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

  std::vector<Schur_Equation_Container> schur_equation_container_;

  std::vector<Position_and_Relation_Container>
      schur_position_and_relation_container_;
};
}
