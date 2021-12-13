#include <iostream>
#include "problem/base_problem.h"
#include "edge/base_edge.h"
#include "vertex/base_vertex.h"
#include <unordered_map>
#include <random>
#include <cusparse_v2.h>
#include "geo/geo.cuh"
#include <fstream>

template<typename T>
class BAL_Edge : public MegBA::BaseEdge<T> {
public:
  MegBA::JVD<T> forward() override {
        using MappedJVD = Eigen::Map<const MegBA::geo::JVD<T>>;
        const auto &Vertices = this->getVertices();
        MappedJVD angle_axisd{&Vertices[0].getEstimation()(0, 0), 3, 1};
        MappedJVD t{&Vertices[0].getEstimation()(3, 0), 3, 1};
        MappedJVD intrinsics{&Vertices[0].getEstimation()(6, 0), 3, 1};

        const auto &point_xyz = Vertices[1].getEstimation();
        const auto &obs_uv = this->getMeasurement();
        auto &&R = MegBA::geo::AngleAxisToRotationKernelMatrix(angle_axisd);
        Eigen::Matrix<MegBA::JetVector<T>, 3, 1> re_projection = R * point_xyz + t;
        re_projection = -re_projection / re_projection(2);
        // f, k1, k2 = intrinsics
        auto fr = MegBA::geo::RadialDistortion(re_projection, intrinsics);

        MegBA::JVD<T> error = fr * re_projection.head(2) - obs_uv;
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
}

int main(int argc, char *arcv[]) {
    std::string name;
    int iter, solver_max_iter, world_size;
    double solver_tol, solver_refuse_ratio, tau, epsilon1, epsilon2;
    std::string out_path;

    if (argc != 10) {
        throw std::runtime_error("not enough parameters");
    } else {
        for (int i = 1; i < argc; ++i) {
            std::string key;
            int idx{0};
            while (arcv[i][idx] != '=')
                key += arcv[i][idx++];
            idx++;
            char *p{&arcv[i][idx]};
            if (key == "--world_size")
                world_size = atoi(p);
            if (key == "--name")
                name = p;
            if (key == "--iter")
                iter = atoi(p);
            if (key == "--solver_tol")
                solver_tol = atof(p);
            if (key == "--solver_refuse_ratio")
                solver_refuse_ratio = atof(p);
            if (key == "--solver_max_iter")
                solver_max_iter = atoi(p);
            if (key == "--tau")
                tau = atof(p);
            if (key == "--epsilon1")
                epsilon1 = atof(p);
            if (key == "--epsilon2")
                epsilon2 = atof(p);
        }
    }
    std::cout
    << "solving " << name
    << ", world_size: " << world_size
    << ", solver iter: " << iter
    << ", solver_tol: " << solver_tol
    << ", solver_refuse_ratio: " << solver_refuse_ratio
    << ", solver_max_iter: " << solver_max_iter
    << ", tau: " << tau
    << ", epsilon1: " << epsilon1
    << ", epsilon2: " << epsilon2
    << std::endl;
    typedef float T;

    std::string path{"../../"};
    std::ifstream fin(path.append("/dataset/" + name));

    int num_cameras = 0, num_points = 0, num_observations = 0;
    fin >> num_cameras;
    fin >> num_points;
    fin >> num_observations;

    MegBA::ProblemOption option{};
    option.nEle = num_observations;
    option.N = 12;
    for (int i = 0; i < world_size; ++i) {
      option.deviceUsed.insert(i);
    }
    MegBA::BaseProblem<T> problem{option};

    std::vector<std::tuple<int, int, Eigen::Matrix<T, 2, 1>>> edge;
    std::vector<std::tuple<int, Eigen::Matrix<T, 9, 1>>> camera_vertices;
    std::vector<std::tuple<int, Eigen::Matrix<T, 3, 1>>> point_vertices;

    int counter = 0;
    // read edges
    while (!fin.eof()) {
        if (counter < num_observations) {
            int idx1, idx2;     // 关联的两个顶点
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
        problem.appendVertex(std::get<0>(camera_vertices[n]), new MegBA::CameraVertex<T>());

        //        Eigen::Matrix<T, 6, 1> camera;
        //        camera.head(3) = std::get<1>(camera_vertices[n]).head(3);
        //        camera.tail(3) = std::get<1>(camera_vertices[n]).segment(3, 3);
        problem.getVertex(std::get<0>(camera_vertices[n])).setEstimation(std::get<1>(std::move(camera_vertices[n])));
        //        problem.getVertex(std::get<0>(camera_vertices[n])).set_Observation(std::get<1>(camera_vertices[n]).tail(3));
        //        problem.getVertex(std::get<0>(camera_vertices[n])).set_Fixed(true);
    }
    for (int n = 0; n < num_points; ++n) {
        problem.appendVertex(std::get<0>(point_vertices[n]), new MegBA::PointVertex<T>());
        problem.getVertex(std::get<0>(point_vertices[n])).setEstimation(std::get<1>(std::move(point_vertices[n])));
    }

    for (int j = 0; j < num_observations; ++j) {
        auto edge_ptr = new BAL_Edge<T>;
        edge_ptr->appendVertex(&problem.getVertex(std::get<0>(edge[j])));
        edge_ptr->appendVertex(&problem.getVertex(std::get<1>(edge[j])));
        edge_ptr->setMeasurement(std::get<2>(std::move(edge[j])));
        problem.appendEdge(edge_ptr);
    }
    problem.solveLM(iter, solver_tol, solver_refuse_ratio, solver_max_iter, tau,
                    epsilon1, epsilon2);
}