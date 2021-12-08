/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <utility>

#include <edge/BaseEdge.h>
#include <Macro.h>
#include <omp.h>

namespace MegBA {
template <typename T> void BaseEdge<T>::appendVertex(BaseVertex<T> &vertex) {
  parent::push_back(&vertex);
}

template <typename T> bool EdgeVector<T>::tryPushBack(BaseEdge<T> &edge) {
  /*
         * Try to push the coming edge into the back of the EdgeVector, return true if success, false if failed.
   */

  // Check whether the EdgeVector is empty. If empty, init.
  const std::size_t hash_of_input_edge = typeid(edge).hash_code();
  const auto vertex_num_in_edge = edge.size();
  if (edges.empty()) {
    name_hash = hash_of_input_edge;
    edges.resize(vertex_num_in_edge);
    // TODO: not consider the situation that vertex is fixed
    unsigned int accumulated_grad_shape = 0;
    unsigned int offset[vertex_num_in_edge];
    bool same_vertex = true;
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      offset[i] = accumulated_grad_shape;
      accumulated_grad_shape += edge[i]->getGradShape();
      if (same_vertex) {
        camera_vertex_num += edge[i]->get_Fixed() ? 0 : 1;
        same_vertex &= edge[i]->kind() ==
                       edge[i == vertex_num_in_edge - 1 ? i : i + 1]->kind();
      } else {
        point_vertex_num += edge[i]->get_Fixed() ? 0 : 1;
        assert(edge[i]->kind() ==
               edge[i == vertex_num_in_edge - 1 ? i : i + 1]->kind());
      }
    }
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      const auto &estimation = edge[i]->get_Estimation();
      const auto &observation = edge[i]->get_Observation();
      edges[i].resize_Jet_Estimation(estimation.rows(), estimation.cols());
      edges[i].resize_Jet_Observation(observation.rows(), observation.cols());
      edges[i].set_Fixed(edge[i]->get_Fixed());
      edges[i].set_Grad_Shape_and_Offset(accumulated_grad_shape, offset[i]);
    }

    const auto &measurement = edge.getMeasurement_();
    Jet_measurement_.resize(measurement.rows(), measurement.cols());
    const auto &information = edge.getInformation_();
    Jet_information_.resize(information.rows(), information.cols());
  } else if (name_hash != hash_of_input_edge)
    return false;

  for (int i = 0; i < vertex_num_in_edge; ++i)
    edges[i].push_back(edge[i]);
  edges_ptr.push_back(&edge);

  const auto &measurement = edge.getMeasurement_();
  for (int i = 0; i < measurement.rows(); ++i) {
    for (int j = 0; j < measurement.cols(); ++j) {
      Jet_measurement_(i, j).append_Jet(measurement(i, j));
    }
  }

  const auto &information = edge.getInformation_();
  for (int i = 0; i < information.rows(); ++i) {
    for (int j = 0; j < information.cols(); ++j) {
      Jet_information_(i, j).append_Jet(information(i, j));
    }
  }
  return true;
};

template <typename T>
void EdgeVector<T>::eraseVertex(const BaseVertex<T> &vertex) {
  for (int i = 0; i < edges_ptr.size(); ++i)
    if (edges_ptr[i]->existVertex(vertex)) {
      for (int j = 0; j < edges.size(); ++j) {
        edges[j].erase(i);
      }
      edges_ptr.erase(edges_ptr.begin() + i);
      i--;
      auto rows = Jet_measurement_.rows(), cols = Jet_measurement_.cols();
      for (int j = 0; j < rows; ++j) {
        for (int k = 0; k < cols; ++k) {
          Jet_measurement_(j, k).erase(i);
        }
      }

      rows = Jet_information_.rows(), cols = Jet_information_.cols();
      for (int j = 0; j < rows; ++j) {
        for (int k = 0; k < cols; ++k) {
          Jet_information_(j, k).erase(i);
        }
      }
    }
}

template <typename T> unsigned int EdgeVector<T>::getGradShape() const {
  unsigned int Grad_Shape = 0;
  for (const auto &vertex_vector : edges)
    Grad_Shape += vertex_vector.getGradShape();
  return Grad_Shape;
}

template <typename T> void EdgeVector<T>::allocateResourcePre() {
  DecideEdgeKind();
  // TODO: num is a global variable
  num.reset(new int[camera_vertex_num + point_vertex_num]);

  if (option_.use_schur) {
    schur_absolute_position.resize(2);
    for (auto &vs : schur_absolute_position) {
      vs.resize(Memory_Pool::getWorldSize());
      for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
        vs[i].resize(Memory_Pool::getElmNum(i));
      }
    }

    schur_relative_position.resize(2);
    for (auto &vs : schur_relative_position) {
      vs.resize(Memory_Pool::getWorldSize());
      for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
        vs[i].resize(Memory_Pool::getElmNum(i));
      }
    }
  } else {
    // TODO: implement this
  }
  switch (option_.device) {
  case CUDA_t: {
    cudaPrepareUpdateData();
    break;
  }
  default: {
    throw std::runtime_error("Not implemented.");
  }
  }
}

template <typename T> void EdgeVector<T>::allocateResourcePost() {
  switch (option_.device) {
  case CUDA_t: {
    preparePositionAndRelationDataCUDA();
    break;
  }
  default: {
    //                assert(0 && "Not implemented.");
    break;
  }
  }
}

template <typename T> void EdgeVector<T>::deallocateResource() {
  csrRowPtr.reset();
  for (auto &ptrs : schur_csrRowPtr)
    for (auto &ptr : ptrs)
      ptr.reset();

  switch (option_.device) {
  case CUDA_t: {
    deallocateResourceCUDA();
    break;
  }
  default: {
    assert(0 && "Not implemented.");
    break;
  }
  }
}

template <typename T> void EdgeVector<T>::makeVertices() {
  if (option_.use_schur) {
    makeSchurVertices();
  } else {
    // TODO: implement this
  }
}

template <typename T> void EdgeVector<T>::makeSchurVertices() {
  for (int vertex_kind_idx = 0; vertex_kind_idx < 2; ++vertex_kind_idx) {
    const auto &vertex_vector = edges[vertex_kind_idx];

    // TODO: global setting
    const auto &vertices_set =
        vertices_set_ptr_->find(vertex_vector[0]->kind())->second;

    auto &relative_position_inner = schur_relative_position[vertex_kind_idx];
    auto &absolute_position_inner = schur_absolute_position[vertex_kind_idx];

    const auto &vertex_vector_other = edges[1 ^ vertex_kind_idx];
    const auto other_kind = vertex_vector_other[0]->kind();

    // iterate element, fill data in Jet_estimation_ and prepare data for make_H_and_g_without_Info_two_Vertices

    std::size_t total_vertex_idx{0};
    for (int i = 0; i < option_.world_size; ++i) {
      const auto &schur_H_entrance_other = schur_H_entrance_[i].ra_[other_kind];
      omp_set_num_threads(16);
#pragma omp parallel for
      for (int j = 0; j < schur_H_entrance_[i].counter; ++j) {
        const auto &row =
            schur_H_entrance_other[vertex_vector_other[total_vertex_idx + j]
                                       ->absolute_position];
        relative_position_inner[i][j] = std::distance(
            row.begin(), std::lower_bound(row.begin(), row.end(),
                                          vertex_vector[total_vertex_idx + j]));
        absolute_position_inner[i][j] =
            vertex_vector[total_vertex_idx + j]->absolute_position;
      }
      total_vertex_idx += schur_H_entrance_[i].counter;

      schur_csrRowPtr[i][vertex_kind_idx] = std::move(
          const_cast<std::vector<SchurHEntrance_t<T>> &>(schur_H_entrance_)[i]
              .csrRowPtr_[vertex_kind_idx]);
      // fill csrRowPtr_. next row's csrRowPtr_ = this row's csrRowPtr_ + this row's non-zero element number.
      const unsigned int rows = vertex_vector[0]->get_Estimation().rows();
      const unsigned int cols = vertex_vector[0]->get_Estimation().cols();
      num[vertex_kind_idx] = vertices_set.size();

      schur_equation_container_[i].nnz[vertex_kind_idx] =
          schur_H_entrance_[i].nnz_in_E;
      schur_equation_container_[i].nnz[vertex_kind_idx + 2] =
          num[vertex_kind_idx] * rows * cols * rows * cols;
      schur_equation_container_[i].dim[vertex_kind_idx] = rows * cols;
    }
  }
}

template <typename T> JVD<T> EdgeVector<T>::forward() {
  edges_ptr[0]->bindEdgeVector(this);
  return edges_ptr[0]->forward();
}

template <typename T> void EdgeVector<T>::fitDevice() {
  if (option_.device == CUDA_t)
    rebindDaPtrs();

  for (int vertex_kind_idx = 0; vertex_kind_idx < edges.size();
       ++vertex_kind_idx) {
    const auto &vertex_vector = edges[vertex_kind_idx];
    // set device_
    auto &Jet_estimation = edges[vertex_kind_idx].get_Jet_Estimation();
    auto &Jet_observation = edges[vertex_kind_idx].get_Jet_Observation();

    auto rows = vertex_vector[0]->get_Estimation().rows();
    auto cols = vertex_vector[0]->get_Estimation().cols();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Jet_estimation(i, j).to(option_.device);
      }
    }

    rows = Jet_observation.rows();
    cols = Jet_observation.cols();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Jet_observation(i, j).to(option_.device);
      }
    }
  }
  auto rows = Jet_measurement_.rows(), cols = Jet_measurement_.cols();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      Jet_measurement_(i, j).to(option_.device);
    }
  }

  rows = Jet_information_.rows(), cols = Jet_information_.cols();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      Jet_information_(i, j).to(option_.device);
    }
  }
}

template <typename T>
void EdgeVector<T>::buildLinearSystemSchur(JVD<T> &Jet_Estimation) {
  switch (option_.device) {
  case CUDA_t: {
    buildLinearSystemSchurCUDA(Jet_Estimation);
    break;
  }
  default:
    throw std::runtime_error("Not Implemented.");
  }
}

template class BaseEdge<float>;
template class BaseEdge<double>;

template class EdgeVector<float>;
template class EdgeVector<double>;
}