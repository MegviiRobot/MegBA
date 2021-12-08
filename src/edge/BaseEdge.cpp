/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <utility>

#include <edge/BaseEdge.h>
#include <omp.h>

namespace MegBA {
template <typename T> void BaseEdge<T>::appendVertex(BaseVertex<T> &vertex) {
  parent::push_back(&vertex);
}

template <typename T>
bool BaseEdge<T>::existVertex(const BaseVertex<T> &vertex) const {
  return std::find(parent::begin(), parent::end(), &vertex) != parent::end();
}

template <typename T> void EdgeVector<T>::decideEdgeKind()  {
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
  case CPU_t:
    free(relativePositionCamera);
    free(relativePositionPoint);
    free(absolutePositionCamera);
    free(absolutePositionPoint);
    free(connectionNumPoint);
    break;
  case CUDA_t:
    clearCUDA();
    break;
  }
  relativePositionCamera = nullptr;
  relativePositionPoint = nullptr;
  absolutePositionCamera = nullptr;
  absolutePositionPoint = nullptr;
  connectionNumPoint = nullptr;
}

template <typename T> EdgeVector<T>::EdgeVector(const ProblemOption &option, const std::vector<SchurHEntrance<T>> &schurHEntrance)
    : _option{option}, schurHEntrance{schurHEntrance},
      schurCsrRowPtr(option.worldSize) {
  schurEquationContainer.reserve(option.worldSize);
  for (int i = 0; i < option.worldSize; ++i) {
    schurEquationContainer.emplace_back(option.device);
    schurPositionAndRelationContainer.emplace_back(option.device);
  }
}

template <typename T> void EdgeVector<T>::rollback() {
  if (_option.useSchur) {
    schurDaPtrs.swap(schurDaPtrsOld);
  } else {
    // TODO: implement this
  }
  rebindDaPtrs();
  backupDaPtrs();
}

template <typename T> bool EdgeVector<T>::tryPushBack(BaseEdge<T> &edge) {
  /*
         * Try to push the coming edge into the back of the EdgeVector, return true if success, false if failed.
   */

  // Check whether the EdgeVector is empty. If empty, init.
  const std::size_t hash_of_input_edge = typeid(edge).hash_code();
  const auto vertex_num_in_edge = edge.size();
  if (edges.empty()) {
    nameHash = hash_of_input_edge;
    edges.resize(vertex_num_in_edge);
    // TODO: not consider the situation that vertex is fixed
    unsigned int accumulated_grad_shape = 0;
    unsigned int offset[vertex_num_in_edge];
    bool same_vertex = true;
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      offset[i] = accumulated_grad_shape;
      accumulated_grad_shape += edge[i]->getGradShape();
      if (same_vertex) {
        cameraVertexNum += edge[i]->get_Fixed() ? 0 : 1;
        same_vertex &= edge[i]->kind() ==
                       edge[i == vertex_num_in_edge - 1 ? i : i + 1]->kind();
      } else {
        pointVertexNum += edge[i]->get_Fixed() ? 0 : 1;
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

    const auto &measurement = edge._measurement;
    jetMeasurement.resize(measurement.rows(), measurement.cols());
    const auto &information = edge._information;
    jetInformation.resize(information.rows(), information.cols());
  } else if (nameHash != hash_of_input_edge)
    return false;

  for (int i = 0; i < vertex_num_in_edge; ++i)
    edges[i].push_back(edge[i]);
  edgesPtr.push_back(&edge);

  const auto &measurement = edge._measurement;
  for (int i = 0; i < measurement.size(); ++i) {
      jetMeasurement(i).append_Jet(measurement(i));
  }

  const auto &information = edge._information;
  for (int i = 0; i < information.size(); ++i) {
    jetInformation(i).append_Jet(information(i));
  }
  return true;
};

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
  // TODO: num is a global variable
  num.reset(new int[cameraVertexNum + pointVertexNum]);

  if (_option.useSchur) {
    schurAbsolutePosition.resize(2);
    for (auto &vs : schurAbsolutePosition) {
      vs.resize(Memory_Pool::getWorldSize());
      for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
        vs[i].resize(Memory_Pool::getElmNum(i));
      }
    }

    schurRelativePosition.resize(2);
    for (auto &vs : schurRelativePosition) {
      vs.resize(Memory_Pool::getWorldSize());
      for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
        vs[i].resize(Memory_Pool::getElmNum(i));
      }
    }
  } else {
    // TODO: implement this
  }
  switch (_option.device) {
  case CUDA_t: {
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
  for (auto &ptrs : schurCsrRowPtr)
    for (auto &ptr : ptrs)
      ptr.reset();

  switch (_option.device) {
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
  if (_option.useSchur) {
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
        verticesSetPtr->find(vertex_vector[0]->kind())->second;

    auto &relative_position_inner = schurRelativePosition[vertex_kind_idx];
    auto &absolute_position_inner = schurAbsolutePosition[vertex_kind_idx];

    const auto &vertex_vector_other = edges[1 ^ vertex_kind_idx];
    const auto other_kind = vertex_vector_other[0]->kind();

    // iterate element, fill data in Jet_estimation_ and prepare data for make_H_and_g_without_Info_two_Vertices

    std::size_t total_vertex_idx{0};
    for (int i = 0; i < _option.worldSize; ++i) {
      const auto &schur_H_entrance_other = schurHEntrance[i].ra_[other_kind];
      omp_set_num_threads(16);
#pragma omp parallel for
      for (int j = 0; j < schurHEntrance[i].counter; ++j) {
        const auto &row =
            schur_H_entrance_other[vertex_vector_other[total_vertex_idx + j]
                                       ->absolutePosition];
        relative_position_inner[i][j] = std::distance(
            row.begin(), std::lower_bound(row.begin(), row.end(),
                                          vertex_vector[total_vertex_idx + j]));
        absolute_position_inner[i][j] =
            vertex_vector[total_vertex_idx + j]->absolutePosition;
      }
      total_vertex_idx += schurHEntrance[i].counter;

      schurCsrRowPtr[i][vertex_kind_idx] = std::move(
          const_cast<std::vector<SchurHEntrance<T>> &>(schurHEntrance)[i]
              .csrRowPtr_[vertex_kind_idx]);
      // fill csrRowPtr_. next row's csrRowPtr_ = this row's csrRowPtr_ + this row's non-zero element number.
      const unsigned int rows = vertex_vector[0]->get_Estimation().rows();
      const unsigned int cols = vertex_vector[0]->get_Estimation().cols();
      num[vertex_kind_idx] = vertices_set.size();

      schurEquationContainer[i].nnz[vertex_kind_idx] =
          schurHEntrance[i].nnz_in_E;
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
  if (_option.device == CUDA_t)
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
void EdgeVector<T>::buildLinearSystemSchur(JVD<T> &jetEstimation) {
  switch (_option.device) {
  case CUDA_t: {
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
}