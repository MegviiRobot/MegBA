#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "algo/lm_algo.h"
#include "edge/base_edge.h"
#include "geo/geo.cuh"
#include "linear_system/schur_LM_linear_system.h"
#include "problem/base_problem.h"
#include "solver/schur_pcg_solver.h"
#include "vertex/base_vertex.h"

template <typename T>
class BalEdgeAnalyticalDerivatives : public MegBA::BaseEdge<T> {
 public:
  MegBA::JVD<T> forward() override {
    using MappedJVD = Eigen::Map<const MegBA::geo::JVD<T>>;
    const auto& Vertices = this->getVertices();
    MappedJVD angle_axisd{&Vertices[0].getEstimation()(0, 0), 3, 1};
    MappedJVD t{&Vertices[0].getEstimation()(3, 0), 3, 1};
    MappedJVD intrinsics{&Vertices[0].getEstimation()(6, 0), 3, 1};

    const auto& point_xyz = Vertices[1].getEstimation();
    const auto& obs_uv = this->getMeasurement();
    MegBA::JVD<T>&& error = MegBA::geo::AnalyticalDerivativesKernelMatrix(
        angle_axisd, t, intrinsics, point_xyz, obs_uv);
    return error;
  }
};

namespace {
template <typename Derived>
bool writeVector(std::ostream& os, const Eigen::DenseBase<Derived>& b) {
  for (int i = 0; i < b.size(); i++) os << b(i) << " ";
  return os.good();
}

template <typename Derived>
bool readVector(std::istream& is, Eigen::DenseBase<Derived>& b) {
  for (int i = 0; i < b.size() && is.good(); i++) is >> b(i);
  return is.good() || is.eof();
}
}  // namespace

DEFINE_int32(world_size, 1, "World size");
DEFINE_string(path, "", "Path to your dataset");
DEFINE_int32(max_iter, 20, "LM solve iteration");
DEFINE_int32(solver_max_iter, 50, "Linear solver iteration");
DEFINE_double(solver_tol, 10., "The tolerance of the linear solver");
DEFINE_double(solver_refuse_ratio, 1., "The refuse ratio of the linear solver");
DEFINE_double(tau, 1., "Initial trust region");
DEFINE_double(epsilon1, 1., "Parameter of LM");
DEFINE_double(epsilon2, 1e-10, "Parameter of LM");

int main(int argc, char* argv[]) {
  std::string out_path;

  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "solving " << FLAGS_path << ", world_size: " << FLAGS_world_size
            << ", max iter: " << FLAGS_max_iter
            << ", solver_tol: " << FLAGS_solver_tol
            << ", solver_refuse_ratio: " << FLAGS_solver_refuse_ratio
            << ", solver_max_iter: " << FLAGS_solver_max_iter
            << ", tau: " << FLAGS_tau << ", epsilon1: " << FLAGS_epsilon1
            << ", epsilon2: " << FLAGS_epsilon2 << std::endl;
  typedef double T;

  std::ifstream fin(FLAGS_path);

  int num_cameras = 0, num_points = 0, num_observations = 0;
  fin >> num_cameras;
  fin >> num_points;
  fin >> num_observations;

  MegBA::ProblemOption problemOption{};
  problemOption.nItem = num_observations;
  problemOption.N = 12;
  for (int i = 0; i < FLAGS_world_size; ++i) {
    problemOption.deviceUsed.push_back(i);
  }
  MegBA::SolverOption solverOption{};
  solverOption.solverOptionPCG.maxIter = FLAGS_solver_max_iter;
  solverOption.solverOptionPCG.tol = FLAGS_solver_tol;
  solverOption.solverOptionPCG.refuseRatio = FLAGS_solver_refuse_ratio;
  MegBA::AlgoOption algoOption{};
  algoOption.algoOptionLM.maxIter = FLAGS_max_iter;
  algoOption.algoOptionLM.initialRegion = FLAGS_tau;
  algoOption.algoOptionLM.epsilon1 = FLAGS_epsilon1;
  algoOption.algoOptionLM.epsilon2 = FLAGS_epsilon2;
  std::unique_ptr<MegBA::BaseAlgo<T>> algo{
      new MegBA::LMAlgo<T>{problemOption, algoOption}};
  std::unique_ptr<MegBA::BaseSolver<T>> solver{
      new MegBA::SchurPCGSolver<T>{problemOption, solverOption}};
  std::unique_ptr<MegBA::BaseLinearSystem<T>> linearSystem{
      new MegBA::SchurLMLinearSystem<T>{problemOption, std::move(solver)}};
  MegBA::BaseProblem<T> problem{problemOption, std::move(algo),
                                std::move(linearSystem)};

  std::vector<std::tuple<int, int, Eigen::Matrix<T, 2, 1>>> edge;
  std::vector<std::tuple<int, Eigen::Matrix<T, 9, 1>>> camera_vertices;
  std::vector<std::tuple<int, Eigen::Matrix<T, 3, 1>>> point_vertices;

  int counter = 0;
  // read edges
  while (!fin.eof()) {
    if (counter < num_observations) {
      int idx1, idx2;  // 关联的两个顶点
      fin >> idx1 >> idx2;
      idx2 += num_cameras;
      Eigen::Matrix<T, 2, 1> observations;
      readVector(fin, observations);
      edge.emplace_back(idx1, idx2, std::move(observations));
    } else {
      break;
    }
    counter++;
  }
  // read vertex
  counter = 0;
  while (!fin.eof()) {
    int idx = counter;
    if (counter < num_cameras) {
      Eigen::Matrix<T, 9, 1> camera_Vector9;
      readVector(fin, camera_Vector9);
      camera_vertices.emplace_back(idx, std::move(camera_Vector9));
    } else {
      Eigen::Matrix<T, 3, 1> point_Vector3;
      readVector(fin, point_Vector3);
      point_vertices.emplace_back(idx, std::move(point_Vector3));
    }
    counter++;
    if (!fin.good()) break;
  }

  for (int n = 0; n < num_cameras; ++n) {
    problem.appendVertex(std::get<0>(camera_vertices[n]),
                         new MegBA::CameraVertex<T>());
    problem.getVertex(std::get<0>(camera_vertices[n]))
        .setEstimation(std::get<1>(std::move(camera_vertices[n])));
  }
  for (int n = 0; n < num_points; ++n) {
    problem.appendVertex(std::get<0>(point_vertices[n]),
                         new MegBA::PointVertex<T>());
    problem.getVertex(std::get<0>(point_vertices[n]))
        .setEstimation(std::get<1>(std::move(point_vertices[n])));
  }

  for (int j = 0; j < num_observations; ++j) {
    auto edgePtr = new BalEdgeAnalyticalDerivatives<T>;
    edgePtr->appendVertex(&problem.getVertex(std::get<0>(edge[j])));
    edgePtr->appendVertex(&problem.getVertex(std::get<1>(edge[j])));
    edgePtr->setMeasurement(std::get<2>(std::move(edge[j])));
    edgePtr->setInformation(Eigen::Matrix2d::Identity());
    problem.appendEdge(*edgePtr);
  }
  problem.solve();
  GFLAGS_NAMESPACE::ShutDownCommandLineFlags();
}
