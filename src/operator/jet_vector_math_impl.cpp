/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include "operator/jet_vector.h"
#include "operator/jet_vector_math_impl.h"
namespace MegBA {
namespace math {
namespace impl {
namespace {
namespace TT {
template<typename T>
struct JetVectorMultipliesJetVectorV {
  __host__ __device__
      T operator()(thrust::tuple<T, T, T, T> zip) {
    T fa, fv, ga, gv;
    thrust::tie(fa, fv, ga, gv) = zip;
    return fa * gv + fv * ga;
  }
};

template <typename T>
struct Inverse : public thrust::unary_function<T, T> {
  __host__ __device__
      T operator()(T x) { return T(1.) / x; }
};

template<typename T>
struct JetVectorDividesJetVectorV {
  __host__ __device__
      T operator()(thrust::tuple<T, T, T, T> zip) {
    T fa, fv, inv_ga, gv;
    thrust::tie(fa, fv, inv_ga, gv) = zip;
    return (fv - fa * inv_ga * gv) * inv_ga;
  }
};

template<typename T>
struct ScalarVectorDividesJetVectorV {
  __host__ __device__
      T operator()(thrust::tuple<T, T, T> zip) {
    T fa, inv_ga, gv;
    thrust::tie(fa, inv_ga, gv) = zip;
    return - fa * inv_ga * gv * inv_ga;
  }
};

template <typename T>
struct ScalarMinusJetVector : public thrust::unary_function<T, T> {
  T scalar;
  explicit ScalarMinusJetVector(T scalar) : scalar(scalar) {}
  __host__ __device__
      T operator()(T x) { return scalar - x; }
};

template <typename T>
struct ScalarDividesJetVectorA : public thrust::unary_function<T, T> {
  T scalar;
  explicit ScalarDividesJetVectorA(T scalar) : scalar(scalar) {}
  __host__ __device__
      T operator()(T x) { return scalar / x; }
};

template <typename T>
struct ScalarDividesJetVectorV : public thrust::binary_function<T, T, T> {
  T scalar;
  explicit ScalarDividesJetVectorV(T scalar) : scalar(scalar) {}
  __host__ __device__
      T operator()(T a, T v) { return -v * scalar / (a * a); }
};

template <typename T>
struct AbsMask : public thrust::unary_function<T, T> {
  __host__ __device__
      T operator()(T x) { return x > 0. ? T(1.) : T(-1.); }
};

template <typename T>
struct Sin : public thrust::unary_function<T, T> {
  __host__ __device__
      T operator()(T x) { return std::sin(x); }
};

template <typename T>
struct NegativeSinMultiplies : public thrust::binary_function<T, T, T> {
  __host__ __device__
      T operator()(T a, T v) { return -std::sin(a) * v; }
};

template <typename T>
struct Cos : public thrust::unary_function<T, T> {
  __host__ __device__
      T operator()(T x) { return std::cos(x); }
};

template <typename T>
struct CosMultiplies : public thrust::binary_function<T, T, T> {
  __host__ __device__
      T operator()(T a, T v) { return std::cos(a) * v; }
};

template <typename T>
struct Sqrt : public thrust::unary_function<T, T> {
  __host__ __device__
      T operator()(T x) { return std::sqrt(x); }
};

template <typename T>
struct SqrtJetVectorV : public thrust::binary_function<T, T, T> {
  __host__ __device__
      T operator()(T sqrted_a, T v) { return T(0.5) * v / sqrted_a; }
};
}  // namespace TT
}  // namespace

template <typename T>
void jetVectorAddJetVectorCPU(const MegBA::JetVector<T> &f,
                              const MegBA::JetVector<T> &g,
                              MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      g.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      thrust::plus<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::plus<T>());
}

template <typename T>
void jetVectorAddScalarVectorCPU(const MegBA::JetVector<T> &f,
                                 const MegBA::JetVector<T> &g,
                                 MegBA::JetVector<T> *out) {
  out->getCPUGrad() = f.getCPUGrad();

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::plus<T>());
}

template <typename T>
void scalarVectorAddScalarVectorCPU(const MegBA::JetVector<T> &f,
                                    const MegBA::JetVector<T> &g,
                                    MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::plus<T>());
}

template <typename T>
void vectorAddVectorCPU(const MegBA::JetVector<T> &f,
                        const MegBA::JetVector<T> &g,
                        MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      jetVectorAddJetVectorCPU(f, g, out);
    } else {
      jetVectorAddScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      jetVectorAddScalarVectorCPU(g, f, out);
    } else {
      scalarVectorAddScalarVectorCPU(f, g, out);
    }
  }
}
template void vectorAddVectorCPU<double>(const MegBA::JetVector<double> &f,
                                         const MegBA::JetVector<double> &g,
                                         MegBA::JetVector<double> *out);

template void vectorAddVectorCPU<float>(const MegBA::JetVector<float> &f,
                                        const MegBA::JetVector<float> &g,
                                        MegBA::JetVector<float> *out);

template <typename T>
void JetVector_minus_JetVector_CPU(const MegBA::JetVector<T> &f,
                                   const MegBA::JetVector<T> &g,
                                   MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      g.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      thrust::minus<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::minus<T>());
}

template <typename T>
void jetVectorMinusScalarVectorCPU(const MegBA::JetVector<T> &f,
                                   const MegBA::JetVector<T> &g,
                                   MegBA::JetVector<T> *out) {
  out->getCPUGrad() = f.getCPUGrad();

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::minus<T>());
}

template <typename T>
void scalarVectorMinusJetVectorCPU(const MegBA::JetVector<T> &f,
                                   const MegBA::JetVector<T> &g,
                                   MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(g.getCPUGrad()[i].begin(), g.getCPUGrad()[i].end(),
                      out->getCPUGrad()[i].begin(), thrust::negate<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::minus<T>());
}

template <typename T>
void scalarVectorMinusScalarVectorCPU(const MegBA::JetVector<T> &f,
                                      const MegBA::JetVector<T> &g,
                                      MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::minus<T>());
}

template <typename T>
void vectorMinusVectorCPU(const MegBA::JetVector<T> &f,
                          const MegBA::JetVector<T> &g,
                          MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      JetVector_minus_JetVector_CPU(f, g, out);
    } else {
      jetVectorMinusScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      scalarVectorMinusJetVectorCPU(f, g, out);
    } else {
      scalarVectorMinusScalarVectorCPU(f, g, out);
    }
  }
}
template void vectorMinusVectorCPU<double>(const MegBA::JetVector<double> &f,
                                           const MegBA::JetVector<double> &g,
                                           MegBA::JetVector<double> *out);

template void vectorMinusVectorCPU<float>(const MegBA::JetVector<float> &f,
                                          const MegBA::JetVector<float> &g,
                                          MegBA::JetVector<float> *out);

template <typename T>
void jetVectorMultipliesJetVectorCPU(const MegBA::JetVector<T> &f,
                                     const MegBA::JetVector<T> &g,
                                     MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          f.getCPURes().begin(), f.getCPUGrad()[i].begin(),
                          g.getCPURes().begin(), g.getCPUGrad()[i].begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          f.getCPURes().end(), f.getCPUGrad()[i].end(),
                          g.getCPURes().end(), g.getCPUGrad()[i].end())),
                      out->getCPUGrad()[i].begin(),
                      TT::JetVectorMultipliesJetVectorV<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::multiplies<T>());
}
template <typename T>
void jetVectorMultipliesScalarVectorCPU(const MegBA::JetVector<T> &f,
                                        const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      g.getCPURes().begin(), out->getCPUGrad()[i].begin(),
                      thrust::multiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::multiplies<T>());
}

template <typename T>
void scalarVectorMultipliesScalarVectorCPU(const MegBA::JetVector<T> &f,
                                           const MegBA::JetVector<T> &g,
                                           MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::multiplies<T>());
}

template <typename T>
void vectorMultipliesVectorCPU(const MegBA::JetVector<T> &f,
                               const MegBA::JetVector<T> &g,
                               MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      jetVectorMultipliesJetVectorCPU(f, g, out);
    } else {
      jetVectorMultipliesScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      jetVectorMultipliesScalarVectorCPU(g, f, out);
    } else {
      scalarVectorMultipliesScalarVectorCPU(f, g, out);
    }
  }
}
template void
vectorMultipliesVectorCPU<double>(const MegBA::JetVector<double> &f,
                                  const MegBA::JetVector<double> &g,
                                  MegBA::JetVector<double> *out);

template void vectorMultipliesVectorCPU<float>(const MegBA::JetVector<float> &f,
                                               const MegBA::JetVector<float> &g,
                                               MegBA::JetVector<float> *out);

template <typename T>
void jetVectorDividesJetVectorCPU(const MegBA::JetVector<T> &f,
                                  const MegBA::JetVector<T> &g,
                                  MegBA::JetVector<T> *out) {
  std::vector<T> inv_ga(f.getCPURes().size());
  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(), inv_ga.begin(),
                    TT::Inverse<T>());
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          f.getCPURes().begin(), f.getCPUGrad()[i].begin(),
                          inv_ga.begin(), g.getCPUGrad()[i].begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          f.getCPURes().end(), f.getCPUGrad()[i].end(),
                          inv_ga.end(), g.getCPUGrad()[i].end())),
                      out->getCPUGrad()[i].begin(),
                      TT::JetVectorDividesJetVectorV<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), inv_ga.begin(),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}

template <typename T>
void jetVectorDividesScalarVectorCPU(const MegBA::JetVector<T> &f,
                                     const MegBA::JetVector<T> &g,
                                     MegBA::JetVector<T> *out) {
  std::vector<T> inv_ga(f.getCPURes().size());
  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(), inv_ga.begin(),
                    TT::Inverse<T>());
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      inv_ga.begin(), out->getCPUGrad()[i].begin(),
                      thrust::multiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), inv_ga.begin(),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}

template <typename T>
void scalarVectorDividesJetVectorCPU(const MegBA::JetVector<T> &f,
                                     const MegBA::JetVector<T> &g,
                                     MegBA::JetVector<T> *out) {
  std::vector<T> inv_ga(f.getCPURes().size());
  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(), inv_ga.begin(),
                    TT::Inverse<T>());
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            f.getCPURes().begin(), inv_ga.begin(), g.getCPUGrad()[i].begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            f.getCPURes().end(), inv_ga.end(), g.getCPUGrad()[i].end())),
        out->getCPUGrad()[i].begin(),
        TT::ScalarVectorDividesJetVectorV<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), inv_ga.begin(),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}

template <typename T>
void scalarVectorDividesScalarVectorCPU(const MegBA::JetVector<T> &f,
                                        const MegBA::JetVector<T> &g,
                                        MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::divides<T>());
}

template <typename T>
void vectorDividesVectorCPU(const MegBA::JetVector<T> &f,
                            const MegBA::JetVector<T> &g,
                            MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      jetVectorDividesJetVectorCPU(f, g, out);
    } else {
      jetVectorDividesScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      scalarVectorDividesJetVectorCPU(f, g, out);
    } else {
      scalarVectorDividesScalarVectorCPU(f, g, out);
    }
  }
}
template void vectorDividesVectorCPU<double>(const MegBA::JetVector<double> &f,
                                             const MegBA::JetVector<double> &g,
                                             MegBA::JetVector<double> *out);

template void vectorDividesVectorCPU<float>(const MegBA::JetVector<float> &f,
                                            const MegBA::JetVector<float> &g,
                                            MegBA::JetVector<float> *out);

template <typename T>
void jetVectorAddScalarCPU(const MegBA::JetVector<T> &f, T g,
                           MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    thrust::make_constant_iterator(g), out->getCPURes().begin(),
                    thrust::plus<T>());
  for (unsigned int i = 0; i < out->getGradShape(); ++i)
    thrust::copy(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                 out->getCPUGrad()[i].begin());
}
template void jetVectorAddScalarCPU<double>(const MegBA::JetVector<double> &f,
                                            double g,
                                            MegBA::JetVector<double> *out);

template void jetVectorAddScalarCPU<float>(const MegBA::JetVector<float> &f,
                                           float g,
                                           MegBA::JetVector<float> *out);

template <typename T>
void jetVectorMinusScalarCPU(const MegBA::JetVector<T> &f, T g,
                             MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    thrust::make_constant_iterator(g), out->getCPURes().begin(),
                    thrust::minus<T>());

  for (unsigned int i = 0; i < out->getGradShape(); ++i)
    thrust::copy(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                 out->getCPUGrad()[i].begin());
}
template void jetVectorMinusScalarCPU<double>(const MegBA::JetVector<double> &f,
                                              double g,
                                              MegBA::JetVector<double> *out);

template void jetVectorMinusScalarCPU<float>(const MegBA::JetVector<float> &f,
                                             float g,
                                             MegBA::JetVector<float> *out);

template <typename T>
void jetVectorMultipliesScalarCPU(const MegBA::JetVector<T> &f, T g,
                                  MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      thrust::make_constant_iterator(g),
                      out->getCPUGrad()[i].begin(), thrust::multiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    thrust::make_constant_iterator(g), out->getCPURes().begin(),
                    thrust::multiplies<T>());
}
template void
jetVectorMultipliesScalarCPU<double>(const MegBA::JetVector<double> &f,
                                     double g, MegBA::JetVector<double> *out);

template void
jetVectorMultipliesScalarCPU<float>(const MegBA::JetVector<float> &f, float g,
                                    MegBA::JetVector<float> *out);

template <typename T>
void jetVectorDividesScalarCPU(const MegBA::JetVector<T> &f, T g,
                               MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      thrust::make_constant_iterator(T(1.) / g),
                      out->getCPUGrad()[i].begin(), thrust::multiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    thrust::make_constant_iterator(T(1.) / g),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}
template void
jetVectorDividesScalarCPU<double>(const MegBA::JetVector<double> &f, double g,
                                  MegBA::JetVector<double> *out);

template void jetVectorDividesScalarCPU<float>(const MegBA::JetVector<float> &f,
                                               float g,
                                               MegBA::JetVector<float> *out);

template <typename T>
void scalarMinusJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(g.getCPUGrad()[i].begin(), g.getCPUGrad()[i].end(),
                      out->getCPUGrad()[i].begin(), thrust::negate<T>());
  }

  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(),
                    out->getCPURes().begin(), TT::ScalarMinusJetVector<T>(f));
}
template void scalarMinusJetVectorCPU<double>(double f,
                                              const MegBA::JetVector<double> &g,
                                              MegBA::JetVector<double> *out);

template void scalarMinusJetVectorCPU<float>(float f,
                                             const MegBA::JetVector<float> &g,
                                             MegBA::JetVector<float> *out);

template <typename T>
void scalarDividesJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(g.getCPURes().begin(), g.getCPURes().end(),
                      g.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      TT::ScalarDividesJetVectorV<T>(f));
  }

  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(),
                    out->getCPURes().begin(),
                    TT::ScalarDividesJetVectorA<T>(f));
}
template void
scalarDividesJetVectorCPU<double>(double f, const MegBA::JetVector<double> &g,
                                  MegBA::JetVector<double> *out);

template void scalarDividesJetVectorCPU<float>(float f,
                                               const MegBA::JetVector<float> &g,
                                               MegBA::JetVector<float> *out);

template <typename T>
void absJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  std::vector<T> mask(f.getCPURes().size());
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), mask.begin(),
                    TT::AbsMask<T>());
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                      mask.begin(), out->getCPUGrad()[i].begin(),
                      thrust::multiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), mask.begin(),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}
template void absJetVectorCPU<double>(const MegBA::JetVector<double> &f,
                                      MegBA::JetVector<double> *out);

template void absJetVectorCPU<float>(const JetVector<float> &f,
                                     JetVector<float> *out);

template <typename T>
void cosJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                      f.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      TT::NegativeSinMultiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    out->getCPURes().begin(), TT::Cos<T>());
}
template void cosJetVectorCPU<double>(const MegBA::JetVector<double> &f,
                                      MegBA::JetVector<double> *out);

template void cosJetVectorCPU<float>(const JetVector<float> &f,
                                     JetVector<float> *out);

template <typename T>
void sinJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                      f.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      TT::CosMultiplies<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    out->getCPURes().begin(), TT::Sin<T>());
}
template void sinJetVectorCPU<double>(const MegBA::JetVector<double> &f,
                                      MegBA::JetVector<double> *out);

template void sinJetVectorCPU<float>(const MegBA::JetVector<float> &f,
                                     MegBA::JetVector<float> *out);

template <typename T>
void sqrtJetVectorCPU(const MegBA::JetVector<T> &f, MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    out->getCPURes().begin(), TT::Sqrt<T>());

  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(out->getCPURes().begin(), out->getCPURes().end(),
                      f.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      TT::SqrtJetVectorV<T>());
  }
}
template void sqrtJetVectorCPU<double>(const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> *out);

template void sqrtJetVectorCPU<float>(const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> *out);
}  // namespace impl
}  // namespace math
}  // namespace MegBA
