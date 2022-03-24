/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <omp.h>

#include <utility>

#include "edge/base_edge.h"
#include "linear_system/base_linear_system.h"

namespace MegBA {
template <typename T>
void BaseEdge<T>::appendVertex(BaseVertex<T> *vertex) {
  m_data.push_back(vertex);
}

template <typename T>
bool BaseEdge<T>::existVertex(const BaseVertex<T> &vertex) const {
  return std::find(m_data.begin(), m_data.end(), &vertex) != m_data.end();
}

template <typename T>
void EdgeVector<T>::decideEdgeKind() {
  if (cameraVertexNum + pointVertexNum == 1)
    edgeKind = ONE;
  else if (cameraVertexNum == 1 && pointVertexNum == 1)
    edgeKind = ONE_CAMERA_ONE_POINT;
  else if (cameraVertexNum == 2 && pointVertexNum == 0)
    edgeKind = TWO_CAMERA;
  else
    edgeKind = MULTI;
}

template <typename T>
EdgeVector<T>::EdgeVector(const ProblemOption &option) : option{option} {}

template <typename T>
bool EdgeVector<T>::tryPushBack(BaseEdge<T> &edge) {
  /*
   * Try to push the coming edge into the back of the EdgeVector, return true if
   * success, false if failed.
   */

  // Check whether the EdgeVector is empty. If empty, init.
  const std::size_t hash_of_input_edge = typeid(edge).hash_code();
  const auto vertex_num_in_edge = edge.size();
  if (edges.empty()) {
    nameHash = hash_of_input_edge;
    edges.resize(vertex_num_in_edge);
    // TODO(Jie Ren): not consider the situation that vertex is fixed
    unsigned int accumulated_grad_shape = 0;
    unsigned int offset[vertex_num_in_edge];
    bool same_vertex = true;
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      offset[i] = accumulated_grad_shape;
      accumulated_grad_shape += edge[i]->getGradShape();
      if (same_vertex) {
        cameraVertexNum += edge[i]->fixed ? 0 : 1;
        same_vertex &= edge[i]->kind() ==
                       edge[i == vertex_num_in_edge - 1 ? i : i + 1]->kind();
      } else {
        pointVertexNum += edge[i]->fixed ? 0 : 1;
        assert(edge[i]->kind() ==
               edge[i == vertex_num_in_edge - 1 ? i : i + 1]->kind());
      }
    }
    for (int i = 0; i < vertex_num_in_edge; ++i) {
      const auto &estimation = edge[i]->getEstimation();
      const auto &observation = edge[i]->getObservation();
      edges[i].resizeJVEstimation(estimation.rows(), estimation.cols());
      edges[i].resizeJVObservation(observation.rows(), observation.cols());
      edges[i].fixed = edge[i]->fixed;
      edges[i].setGradShapeAndOffset(accumulated_grad_shape, offset[i]);
    }

    const auto &measurement = edge._getMeasurement();
    jetMeasurement.resize(measurement.rows(), measurement.cols());
    const auto &information = edge._getInformation();
    jetInformation.resize(information.rows(), information.cols());
  } else if (nameHash != hash_of_input_edge) {
    return false;
  }

  for (int i = 0; i < vertex_num_in_edge; ++i) edges[i].push_back(edge[i]);
  edgesPtr.push_back(&edge);

  const auto &measurement = edge._getMeasurement();
  for (int i = 0; i < measurement.size(); ++i) {
    jetMeasurement(i).appendJet(measurement(i));
  }

  const auto &information = edge._getInformation();
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

template <typename T>
unsigned int EdgeVector<T>::getGradShape() const {
  unsigned int Grad_Shape = 0;
  for (const auto &vertex_vector : edges)
    Grad_Shape += vertex_vector.getGradShape();
  return Grad_Shape;
}

template <typename T>
void EdgeVector<T>::allocateResource() {
  decideEdgeKind();
  switch (option.device) {
    case Device::CUDA:
      allocateResourceCUDA();
      break;
    default:
      throw std::runtime_error("Not implemented.");
  }
}

template <typename T>
void EdgeVector<T>::deallocateResource() {
  switch (option.device) {
    case Device::CUDA:
      deallocateResourceCUDA();
      break;
    default:
      throw std::runtime_error("Not implemented");
  }
}

template <typename T>
JVD<T> EdgeVector<T>::forward() const {
  edgesPtr[0]->bindEdgeVector(this);
  return edgesPtr[0]->forward();
}

template <typename T>
void EdgeVector<T>::fitDevice() {
  if (option.device == Device::CUDA) bindCUDAGradPtrs();

  for (int vertexKindIdx = 0; vertexKindIdx < edges.size(); ++vertexKindIdx) {
    const auto &vertexVector = edges[vertexKindIdx];
    // set _device
    auto &jetEstimation = edges[vertexKindIdx].getJVEstimation();
    auto &jvObservation = edges[vertexKindIdx].getJVObservation();

    auto rows = vertexVector[0]->getEstimation().rows();
    auto cols = vertexVector[0]->getEstimation().cols();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        jetEstimation(i, j).to(option.device);
      }
    }

    rows = jvObservation.rows();
    cols = jvObservation.cols();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        jvObservation(i, j).to(option.device);
      }
    }
  }
  auto rows = jetMeasurement.rows(), cols = jetMeasurement.cols();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      jetMeasurement(i, j).to(option.device);
    }
  }

  rows = jetInformation.rows(), cols = jetInformation.cols();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      jetInformation(i, j).to(option.device);
    }
  }
}

template <typename T>
void EdgeVector<T>::buildLinearSystem(
    const JVD<T> &jetEstimation,
    const BaseLinearSystem<T> &linearSystem) const {
  switch (option.device) {
    case Device::CUDA: {
      buildLinearSystemCUDA(jetEstimation, linearSystem);
      break;
    }
    default:
      throw std::runtime_error("Not Implemented.");
  }
}

template <typename T>
void EdgeVector<T>::buildPositionContainer(
    const std::vector<HessianEntrance<T>> &hessianEntrance) {
  const auto worldSize = MemoryPool::getWorldSize();
  positionContainers.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    positionContainers[i].absolutePosition.resize(edges.size());
    positionContainers[i].relativePosition.resize(edges.size());
    for (int j = 0; j < edges.size(); ++j) {
      positionContainers[i].absolutePosition[j] =
          (int *)malloc(MemoryPool::getItemNum(i) * sizeof(int));
      positionContainers[i].relativePosition[j] =
          (int *)malloc(MemoryPool::getItemNum(i) * sizeof(int));
    }
  }

  std::size_t totalVertexIdx{0};
  for (int i = 0; i < worldSize; ++i) {
    for (int j = 0; j < edges.size(); ++j) {
      const auto kind = edges[j][0]->kind();
      const auto &schurHEntranceOther = hessianEntrance[i].ra[1 ^ kind];
      omp_set_num_threads(16);
#pragma omp parallel for
      for (int k = 0; k < hessianEntrance[i].counter; ++k) {
        // TODO(Jie Ren): we assume there only exist two vertices in each edge,
        //  we need some structure to record the connection relationship
        const auto &row = schurHEntranceOther[edges[1 ^ j][totalVertexIdx + k]
                                                  ->absolutePosition];
        positionContainers[i].relativePosition[j][k] = std::distance(
            row.begin(), std::lower_bound(row.begin(), row.end(),
                                          edges[j][totalVertexIdx + k]));
        positionContainers[i].absolutePosition[j][k] =
            edges[j][totalVertexIdx + k]->absolutePosition;
      }
    }
    totalVertexIdx += hessianEntrance[i].counter;
    // fill csrRowPtr. next row's csrRowPtr = this row's csrRowPtr + this
    // row's non-zero element number.
  }
}

template class BaseEdge<float>;
template class BaseEdge<double>;

template class EdgeVector<float>;
template class EdgeVector<double>;
}  // namespace MegBA
