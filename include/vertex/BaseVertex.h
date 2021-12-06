#pragma once
#include "Common.h"
#include "Eigen/Core"
#include <set>
#include <map>
#include <utility>

namespace MegBA {
template <typename T> class BaseVertexWrapper;

template <typename T> class BaseEdgeWrapper;

template <typename T>
using JVD = Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using TD = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

enum VertexKind_t { CAMERA = 0, POINT = 1, NONE = 2 };

template <typename T> struct BaseVertex {
  BaseVertex() = default;

  virtual ~BaseVertex() = default;

  void set_Fixed(bool fixed) { fixed_ = fixed; };

  void set_Estimation(const TD<T> &estimation) { estimation_ = estimation; };

  void set_Estimation(TD<T> &&estimation) { estimation_ = std::move(estimation); };

  void set_Observation(const TD<T> &observation) { observation_ = observation; };

  void set_Observation(TD<T> &&observation) { observation_ = std::move(observation); };

  bool get_Fixed() { return fixed_; };

  const TD<T> &get_Estimation() const { return estimation_; };

  TD<T> &get_Estimation() { return estimation_; };

  const TD<T> &get_Observation() const { return observation_; };

  TD<T> &get_Observation() { return observation_; };

  bool has_Observation() { return observation_.rows() || observation_.cols(); };

  unsigned int get_Grad_Shape() const {
    return fixed_ ? 0 : estimation_.rows() * estimation_.cols();
  };

  bool operator==(const BaseVertex<T> &vertex) const {
    return estimation_ == vertex.estimation_;
  };

  virtual VertexKind_t kind() { return NONE; };
  int absolute_position = 0;

private:
  bool fixed_ = false;
  TD<T> estimation_;
  TD<T> observation_{};
};

template <typename T> struct CameraVertex : public BaseVertex<T> {
  CameraVertex() = default;

  VertexKind_t kind() final { return CAMERA; };
};

template <typename T> struct PointVertex : public BaseVertex<T> {
  PointVertex() = default;

  VertexKind_t kind() final { return POINT; };
};

template <typename T>
class Vertex_Vector : public std::vector<BaseVertex<T> *> {
  typedef std::vector<BaseVertex<T> *> parent;
  std::map<const BaseVertex<T> *, std::size_t> vertex_counter_;
  JVD<T> Jet_estimation_;
  JVD<T> Jet_observation_;
  long estimation_rows_;
  long estimation_cols_;
  long observation_rows_;
  long observation_cols_;
  bool fixed_;

public:
  void CPU() {
    for (int i = 0; i < estimation_rows_; ++i)
      for (int j = 0; j < estimation_cols_; ++j)
        Jet_estimation_(i, j).CPU();

    for (int i = 0; i < observation_rows_; ++i)
      for (int j = 0; j < observation_cols_; ++j)
        Jet_observation_(i, j).CPU();
  }

  void erase(std::size_t idx) {
    for (int i = 0; i < estimation_rows_; ++i)
      for (int j = 0; j < estimation_cols_; ++j)
        Jet_estimation_(i, j).erase(idx);

    for (int i = 0; i < observation_rows_; ++i)
      for (int j = 0; j < observation_cols_; ++j)
        Jet_observation_(i, j).erase(idx);

    auto vertex = (*this)[idx];
    auto find = vertex_counter_.find(vertex);
    if (find->second == 1)
      vertex_counter_.erase(find);
    else
      find->second--;

    parent::erase(parent::begin() + idx);
  };

  bool exist_Vertex(const BaseVertex<T> *vertex) const {
    return vertex_counter_.find(vertex) != vertex_counter_.end();
  };

  void resize_Jet_Estimation(long rows, long cols) {
    estimation_rows_ = rows;
    estimation_cols_ = cols;
    Jet_estimation_.resize(rows, cols);
  };

  void resize_Jet_Observation(long rows, long cols) {
    observation_rows_ = rows;
    observation_cols_ = cols;
    Jet_observation_.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Jet_observation_(i, j).set_Grad_Shape(0);
      }
    }
  };

  void set_Fixed(bool fixed) { fixed_ = fixed; }

  void set_Grad_Shape_and_Offset(unsigned int N, unsigned int offset) {
    if (fixed_)
      offset = -estimation_rows_ * estimation_cols_ - 1;
    for (int i = 0; i < estimation_rows_; ++i) {
      for (int j = 0; j < estimation_cols_; ++j) {
        Jet_estimation_(i, j).set_Grad_Shape(fixed_ ? 0 : N);
        Jet_estimation_(i, j).set_Grad_Position(
            fixed_ ? -1 : offset + i * estimation_cols_ + j);
      }
    }
  }

  void push_back(BaseVertex<T> *vertex) {
    parent::push_back(vertex);

    auto find = vertex_counter_.find(vertex);
    if (find == vertex_counter_.end())
      vertex_counter_.insert(std::make_pair(vertex, 1));
    else
      find->second++;

    const auto &estimation = vertex->get_Estimation();
    for (int i = 0; i < estimation_rows_; ++i) {
      for (int j = 0; j < estimation_cols_; ++j) {
        Jet_estimation_(i, j).append_Jet(estimation(i, j));
      }
    }

    const auto &observation = vertex->get_Observation();
    for (int i = 0; i < observation_rows_; ++i) {
      for (int j = 0; j < observation_cols_; ++j) {
        Jet_observation_(i, j).append_Jet(observation(i, j));
      }
    }
  };

  auto &get_Jet_Estimation() { return Jet_estimation_; }

  const auto &get_Jet_Estimation() const { return Jet_estimation_; }

  auto get_Estimation_Shape() const {
    return estimation_rows_ * estimation_cols_;
  }

  auto get_Grad_Shape() const {
    return fixed_ ? 0 : estimation_rows_ * estimation_cols_;
  }

  auto &get_Jet_Observation() { return Jet_observation_; }

  const auto &get_Jet_Observation() const { return Jet_observation_; }
};

template <typename T> class BaseVertexWrapper {

public:
  JVD<T> const &get_Estimation() const { return *Jet_estimation_; }

  JVD<T> const &get_Observation() const {
    return *Jet_observation_;
  }

private:
  friend BaseEdgeWrapper<T>;

  void bind_Jet_Estimation(const JVD<T> &Jet_estimation) {
    Jet_estimation_ = &Jet_estimation;
  }

  void bind_Jet_Observation(const JVD<T> &Jet_observation) {
    Jet_observation_ = &Jet_observation;
  }

  JVD<T> const *Jet_estimation_ = nullptr;

  JVD<T> const *Jet_observation_ = nullptr;
};

template <typename T>
class BaseEdgeWrapper : public std::vector<BaseVertexWrapper<T>> {
private:
  friend BaseEdge<T>;

  typedef std::vector<BaseVertexWrapper<T>> parent;

  JVD<T> const *Jet_measurement_ = nullptr;

  JVD<T> const &get_Measurement() const {
    return *Jet_measurement_;
  };

  void bind_Edge_Vector(const EdgeVector<T> *EV) {
    Jet_measurement_ = &EV->get_Measurement();
    parent::resize(EV->get_Edges().size());
    for (int i = 0; i < parent::size(); ++i) {
      (*this)[i].bind_Jet_Estimation(EV->get_Jet_Estimation_i(i));
      (*this)[i].bind_Jet_Observation(EV->get_Jet_Observation_i(i));
    }
  };
};
}
