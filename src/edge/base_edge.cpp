/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "edge/base_edge.h"
#include <omp.h>
#include <utility>

namespace MegBA {
template <typename T> void BaseEdge<T>::appendVertex(BaseVertex<T> *vertex) {
  parent::push_back(vertex);
}

template <typename T>
bool BaseEdge<T>::existVertex(const BaseVertex<T> &vertex) const {
  return std::find(parent::begin(), parent::end(), &vertex) != parent::end();
}

template <typename T> void EdgeVector<T>::decideEdgeKind() {
  if (cameraVertexNum + pointVertexNum == 1)
    edgeKind = ONE;
  else if (cameraVertexNum == 1 && pointVertexNum == 1)
    edgeKind = ONE_CAMERA_ONE_POINT;
  else if (cameraVertexNum == 2 && pointVertexNum == 0)
    edgeKind = TWO_CAMERA;
  else
    edgeKind = MULTI;
}

template <typename T> void EdgeVector<T>::SchurEquationContainer::clear() {
  switch (_device) {
  case Device::CPU: {
    for (int i = 0; i < 2; ++i)
      free(csrRowPtr[i]);
    for (int i = 0; i < 4; ++i)
      free(csrVal[i]);
    for (int i = 0; i < 2; ++i)
      free(csrColInd[i]);
    free(g);
    break;
  }
  case Device::CUDA: {
    clearCUDA();
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
}

template <typename T>
void EdgeVector<T>::PositionAndRelationContainer::clear() {
  switch (_device) {
  case Device::CPU:
    free(relativePositionCamera);
    free(relativePositionPoint);
    free(absolutePositionCamera);
    free(absolutePositionPoint);
    free(connectionNumPoint);
    break;
  case Device::CUDA:
    clearCUDA();
    break;
  }
  relativePositionCamera = nullptr;
  relativePositionPoint = nullptr;
  absolutePositionCamera = nullptr;
  absolutePositionPoint = nullptr;
  connectionNumPoint = nullptr;
}

template <typename T>
EdgeVector<T>::EdgeVector(const ProblemOption &option,
                          const std::vector<SchurHessianEntrance<T>> &schurHessianEntrance)
    : _option{option}, schurHessianEntrance{schurHessianEntrance},
      schurCsrRowPtr(option.deviceUsed.size()) {
  schurEquationContainer.reserve(option.deviceUsed.size());
  for (int i = 0; i < option.deviceUsed.size(); ++i) {
    schurEquationContainer.emplace_back(option.device);
    schurPositionAndRelationContainer.emplace_back(option.device);
  }
}

template <typename T> void EdgeVector<T>::rollback() {
  if (_option.useSchur) {
    schurValueDevicePtrs.swap(schurValueDevicePtrsOld);
  } else {
    // TODO(Jie Ren): implement this
  }
  bindCUDAGradPtrs();
  backupValueDevicePtrs();
}

template <typename T> bool EdgeVector<T>::tryPushBack(BaseEdge<T> *edge) {
  /*
   * Try to push the coming edge into the back of the EdgeVector, return true if
   * success, false if failed.
   */

  // Check whether the EdgeVector is empty. If empty, init.
  const std::size_t hash_of_input_edge = typeid(edge).hash_code();
  const auto vertex_num_in_edge = edge->size();
  if (edges.empty()) {
    nameHash = hash_of_input_edge;
    edges.resize(vertex_num_in_edge);
    // TODO(Jie Ren): not consider the situation that vertex is fixed
    unsigned int accumulated_grad_shape = 0;
    unsigned int offset[vertex_num_in_edge];
    bool same_vertex = true;
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      offset[i] = accumulated_grad_shape;
      accumulated_grad_shape += edge->operator[](i)->getGradShape();
      if (same_vertex) {
        cameraVertexNum += edge->operator[](i)->fixed ? 0 : 1;
        same_vertex &= edge->operator[](i)->kind() ==
                       edge->operator[](i == vertex_num_in_edge - 1 ? i : i + 1)->kind();
      } else {
        pointVertexNum += edge->operator[](i)->fixed ? 0 : 1;
        assert(edge->operator[](i)->kind() ==
               edge->operator[](i == vertex_num_in_edge - 1 ? i : i + 1)->kind());
      }
    }
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      const auto &estimation = edge->operator[](i)->getEstimation();
      const auto &observation = edge->operator[](i)->getObservation();
      edges[i].resizeJVEstimation(estimation.rows(), estimation.cols());
      edges[i].resizeJVObservation(observation.rows(), observation.cols());
      edges[i].fixed = edge->operator[](i)->fixed;
      edges[i].setGradShapeAndOffset(accumulated_grad_shape, offset[i]);
    }

    const auto &measurement = edge->_measurement;
    jetMeasurement.resize(measurement.rows(), measurement.cols());
    const auto &information = edge->_information;
    jetInformation.resize(information.rows(), information.cols());
  } else if (nameHash != hash_of_input_edge) {
    return false;
  }

  for (int i = 0; i < vertex_num_in_edge; ++i)
    edges[i].push_back(edge->operator[](i));
  edgesPtr.push_back(edge);

  const auto &measurement = edge->_measurement;
  for (int i = 0; i < measurement.size(); ++i) {
    jetMeasurement(i).appendJet(measurement(i));
  }

  const auto &information = edge->_information;
  for (int i = 0; i < information.size(); ++i) {
    jetInformation(i).appendJet(information(i));
  }
  return true;
}

template <typename T>
void EdgeVector<T>::eraseVertex(const BaseVertex<T> &vertex) {
  for (int i = 0; i < edgesPtr.size(); ++i)
    if (edgesPtr[i]->existVertex(vertex)) {
      for (int j = 0; j < edges.size(); ++j) {
        edges[j].erase(i);
      }
      edgesPtr.erase(edgesPtr.begin() + i);
      i--;
      auto rows = jetMeasurement.rows(), cols = jetMeasurement.cols();
      for (int j = 0; j < rows; ++j) {
        for (int k = 0; k < cols; ++k) {
          jetMeasurement(j, k).erase(i);
        }
      }

      rows = jetInformation.rows(), cols = jetInformation.cols();
      for (int j = 0; j < rows; ++j) {
        for (int k = 0; k < cols; ++k) {
          jetInformation(j, k).erase(i);
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
  decideEdgeKind();
  // TODO(Jie Ren): num is a global variable
  num.reset(new int[cameraVertexNum + pointVertexNum]);

  if (_option.useSchur) {
    schurAbsolutePosition.resize(2);
    for (auto &vs : schurAbsolutePosition) {
      vs.resize(MemoryPool::getWorldSize());
      for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
        vs[i].resize(MemoryPool::getItemNum(i));
      }
    }

    schurRelativePosition.resize(2);
    for (auto &vs : schurRelativePosition) {
      vs.resize(MemoryPool::getWorldSize());
      for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
        vs[i].resize(MemoryPool::getItemNum(i));
      }
    }
  } else {
    // TODO(Jie Ren): implement this
  }
  switch (_option.device) {
  case Device::CUDA: {
    PrepareUpdateDataCUDA();
    break;
  }
  default: {
    throw std::runtime_error("Not implemented.");
  }
  }
}

template <typename T> void EdgeVector<T>::allocateResourcePost() {
  switch (_option.device) {
  case Device::CUDA: {
    preparePositionAndRelationDataCUDA();
    break;
  }
  default: {
    throw std::runtime_error("Not implemented");
    break;
  }
  }
}

template <typename T> void EdgeVector<T>::deallocateResource() {
  for (auto &ptrs : schurCsrRowPtr)
    for (auto &ptr : ptrs)
      ptr.reset();

  switch (_option.device) {
  case Device::CUDA: {
    deallocateResourceCUDA();
    break;
  }
  default: {
    throw std::runtime_error("Not implemented");
    break;
  }
  }
}

template <typename T> void EdgeVector<T>::makeVertices() {
  if (_option.useSchur) {
    makeSchurVertices();
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T> void EdgeVector<T>::makeSchurVertices() {
  for (int vertex_kind_idx = 0; vertex_kind_idx < 2; ++vertex_kind_idx) {
    const auto &vertex_vector = edges[vertex_kind_idx];

    // TODO(Jie Ren): global setting
    const auto &vertices_set =
        verticesSetPtr->find(vertex_vector[0]->kind())->second;

    auto &relative_position_inner = schurRelativePosition[vertex_kind_idx];
    auto &absolute_position_inner = schurAbsolutePosition[vertex_kind_idx];

    const auto &vertex_vector_other = edges[1 ^ vertex_kind_idx];
    const auto other_kind = vertex_vector_other[0]->kind();

    // iterate element, fill data in Jet_estimation_ and prepare data for
    // make_H_and_g_without_Info_two_Vertices

    std::size_t total_vertex_idx{0};
    for (int i = 0; i < _option.deviceUsed.size(); ++i) {
      const auto &schur_H_entrance_other = schurHessianEntrance[i].ra[other_kind];
      omp_set_num_threads(16);
#pragma omp parallel for
      for (int j = 0; j < schurHessianEntrance[i].counter; ++j) {
        const auto &row =
            schur_H_entrance_other[vertex_vector_other[total_vertex_idx + j]
                                       ->absolutePosition];
        relative_position_inner[i][j] = std::distance(
            row.begin(), std::lower_bound(row.begin(), row.end(),
                                          vertex_vector[total_vertex_idx + j]));
        absolute_position_inner[i][j] =
            vertex_vector[total_vertex_idx + j]->absolutePosition;
      }
      total_vertex_idx += schurHessianEntrance[i].counter;

      schurCsrRowPtr[i][vertex_kind_idx] = std::move(
          const_cast<std::vector<SchurHessianEntrance<T>> &>(schurHessianEntrance)[i]
              .csrRowPtr[vertex_kind_idx]);
      // fill csrRowPtr_. next row's csrRowPtr_ = this row's csrRowPtr_ + this
      // row's non-zero element number.
      const unsigned int rows = vertex_vector[0]->getEstimation().rows();
      const unsigned int cols = vertex_vector[0]->getEstimation().cols();
      num[vertex_kind_idx] = vertices_set.size();

      schurEquationContainer[i].nnz[vertex_kind_idx] = schurHessianEntrance[i].nnzInE;
      schurEquationContainer[i].nnz[vertex_kind_idx + 2] =
          num[vertex_kind_idx] * rows * cols * rows * cols;
      schurEquationContainer[i].dim[vertex_kind_idx] = rows * cols;
    }
  }
}

template <typename T> JVD<T> EdgeVector<T>::forward() {
  edgesPtr[0]->_edge.bindEdgeVector(this);
  return edgesPtr[0]->forward();
}

template <typename T> void EdgeVector<T>::fitDevice() {
  if (_option.device == Device::CUDA)
    bindCUDAGradPtrs();

  for (int vertex_kind_idx = 0; vertex_kind_idx < edges.size();
       ++vertex_kind_idx) {
    const auto &vertex_vector = edges[vertex_kind_idx];
    // set _device
    auto &Jet_estimation = edges[vertex_kind_idx].getJVEstimation();
    auto &Jet_observation = edges[vertex_kind_idx].getJVObservation();

    auto rows = vertex_vector[0]->getEstimation().rows();
    auto cols = vertex_vector[0]->getEstimation().cols();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Jet_estimation(i, j).to(_option.device);
      }
    }

    rows = Jet_observation.rows();
    cols = Jet_observation.cols();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Jet_observation(i, j).to(_option.device);
      }
    }
  }
  auto rows = jetMeasurement.rows(), cols = jetMeasurement.cols();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      jetMeasurement(i, j).to(_option.device);
    }
  }

  rows = jetInformation.rows(), cols = jetInformation.cols();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      jetInformation(i, j).to(_option.device);
    }
  }
}

template <typename T>
void EdgeVector<T>::buildLinearSystemSchur(const JVD<T> &jetEstimation) {
  switch (_option.device) {
  case Device::CUDA: {
    buildLinearSystemSchurCUDA(jetEstimation);
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
}  // namespace MegBA
