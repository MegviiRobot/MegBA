/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <Eigen/Core>
#include <map>
#include <vector>
#include <set>
#include <utility>
#include "common.h"

namespace MegBA {
template <typename T>
using JVD = Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using TD = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

enum VertexKind { CAMERA = 0, POINT = 1, NONE = 2 };

template <typename T> struct BaseVertex {
  BaseVertex() = default;

  template <typename Estimation> void setEstimation(Estimation &&estimation) {
    _estimation = std::forward<Estimation>(estimation);
  }

  template <typename Estimation> void setObservation(Estimation &&observation) {
    _observation = std::forward<Estimation>(observation);
  }

  const TD<T> &getEstimation() const { return _estimation; }

  TD<T> &getEstimation() { return _estimation; }

  const TD<T> &getObservation() const { return _observation; }

  TD<T> &getObservation() { return _observation; }

  unsigned int getGradShape() const {
    return fixed ? 0 : _estimation.rows() * _estimation.cols();
  }

  bool operator==(const BaseVertex<T> &vertex) const {
    return _estimation == vertex._estimation;
  }

  virtual VertexKind kind() { return NONE; }
  int absolutePosition{};
  bool fixed{false};

 private:
  TD<T> _estimation{};
  TD<T> _observation{};
};

template <typename T> struct CameraVertex : public BaseVertex<T> {
  CameraVertex() = default;

  VertexKind kind() final { return CAMERA; };
};

template <typename T> struct PointVertex : public BaseVertex<T> {
  PointVertex() = default;

  VertexKind kind() final { return POINT; }
};

template <typename T> class VertexVector : public std::vector<BaseVertex<T> *> {
  typedef std::vector<BaseVertex<T> *> parent;
  std::map<const BaseVertex<T> *, std::size_t> _vertexCounter;
  JVD<T> _jvEstimation;
  JVD<T> _jvObservation;
  int64_t _estimationRows;
  int64_t _estimationCols;
  int64_t _observationRows;
  int64_t _observationCols;

 public:
  bool fixed;

  void CPU() {
    for (int i = 0; i < _estimationRows; ++i)
      for (int j = 0; j < _estimationCols; ++j)
        _jvEstimation(i, j).CPU();

    for (int i = 0; i < _observationRows; ++i)
      for (int j = 0; j < _observationCols; ++j)
        _jvObservation(i, j).CPU();
  }

  void erase(std::size_t idx) {
    for (int i = 0; i < _estimationRows; ++i)
      for (int j = 0; j < _estimationCols; ++j)
        _jvEstimation(i, j).erase(idx);

    for (int i = 0; i < _observationRows; ++i)
      for (int j = 0; j < _observationCols; ++j)
        _jvObservation(i, j).erase(idx);

    auto vertex = (*this)[idx];
    auto find = _vertexCounter.find(vertex);
    if (find->second == 1)
      _vertexCounter.erase(find);
    else
      find->second--;

    parent::erase(parent::begin() + idx);
  }

  bool existVertex(const BaseVertex<T> *vertex) const {
    return _vertexCounter.find(vertex) != _vertexCounter.end();
  }

  void resizeJVEstimation(int64_t rows, int64_t cols) {
    _estimationRows = rows;
    _estimationCols = cols;
    _jvEstimation.resize(rows, cols);
  }

  void resizeJVObservation(int64_t rows, int64_t cols) {
    _observationRows = rows;
    _observationCols = cols;
    _jvObservation.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        _jvObservation(i, j).set_Grad_Shape(0);
      }
    }
  }

  void setGradShapeAndOffset(unsigned int N, unsigned int offset) {
    if (fixed)
      offset = -_estimationRows * _estimationCols - 1;
    for (int i = 0; i < _estimationRows; ++i) {
      for (int j = 0; j < _estimationCols; ++j) {
        _jvEstimation(i, j).set_Grad_Shape(fixed ? 0 : N);
        _jvEstimation(i, j).setGradPosition(
            fixed ? -1 : offset + i * _estimationCols + j);
      }
    }
  }

  void push_back(BaseVertex<T> *vertex) {
    parent::push_back(vertex);

    auto find = _vertexCounter.find(vertex);
    if (find == _vertexCounter.end())
      _vertexCounter.insert(std::make_pair(vertex, 1));
    else
      find->second++;

    const auto &estimation = vertex->getEstimation();
    for (int i = 0; i < _estimationRows; ++i) {
      for (int j = 0; j < _estimationCols; ++j) {
        _jvEstimation(i, j).appendJet(estimation(i, j));
      }
    }

    const auto &observation = vertex->getObservation();
    for (int i = 0; i < _observationRows; ++i) {
      for (int j = 0; j < _observationCols; ++j) {
        _jvObservation(i, j).appendJet(observation(i, j));
      }
    }
  }

  auto &getJVEstimation() { return _jvEstimation; }

  const auto &getJVEstimation() const { return _jvEstimation; }

  auto getGradShape() const {
    return fixed ? 0 : _estimationRows * _estimationCols;
  }

  auto &getJVObservation() { return _jvObservation; }

  const auto &getJVObservation() const { return _jvObservation; }
};

template <typename T> class EdgeWrapper;

template <typename T> class VertexWrapper {
  friend EdgeWrapper<T>;

  void bindJVEstimation(const JVD<T> &jvEstimation) {
    _jvEstimation = &jvEstimation;
  }

  void bindJVObservation(const JVD<T> &jvObservation) {
    _jvObservation = &jvObservation;
  }

  JVD<T> const *_jvEstimation{nullptr};

  JVD<T> const *_jvObservation{nullptr};

 public:
  JVD<T> const &getEstimation() const { return *_jvEstimation; }

  JVD<T> const &getObservation() const { return *_jvObservation; }
};

template <typename T>
class EdgeWrapper : public std::vector<VertexWrapper<T>> {
  typedef std::vector<VertexWrapper<T>> parent;
  JVD<T> const *_jvMeasurement{nullptr};

 public:
  JVD<T> const &getMeasurement() const { return *_jvMeasurement; }

  void bindEdgeVector(const EdgeVector<T> *ev) {
    _jvMeasurement = &ev->getMeasurement();
    parent::resize(ev->getVertexVectors().size());
    for (int i = 0; i < parent::size(); ++i) {
      (*this)[i].bindJVEstimation(ev->getEstimation(i));
      (*this)[i].bindJVObservation(ev->getObservation(i));
    }
  }
};
}  // namespace MegBA
