/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include "operator/jet_vector.h"
#include "operator/jet_vector_math_impl.h"

namespace MegBA {
namespace math {
namespace impl {
namespace {
namespace TT {
template <typename T>
struct JetVectorMulJetVectorV {
  __host__ __device__ T operator()(thrust::tuple<T, T, T, T> zip) {
    T fa, fv, ga, gv;
    thrust::tie(fa, fv, ga, gv) = zip;
    return fa * gv + fv * ga;
  }
};

template <typename T>
struct Inverse : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return T(1.) / x; }
};

template <typename T>
struct JetVectorDivJetVectorV {
  __host__ __device__ T operator()(thrust::tuple<T, T, T, T> zip) {
    T fa, fv, inv_ga, gv;
    thrust::tie(fa, fv, inv_ga, gv) = zip;
    return (fv - fa * inv_ga * gv) * inv_ga;
  }
};

template <typename T>
struct ScalarVectorDivJetVectorV {
  __host__ __device__ T operator()(thrust::tuple<T, T, T> zip) {
    T fa, inv_ga, gv;
    thrust::tie(fa, inv_ga, gv) = zip;
    return -fa * inv_ga * gv * inv_ga;
  }
};

template <typename T>
struct ScalarSubJetVector : public thrust::unary_function<T, T> {
  T scalar;
  explicit ScalarSubJetVector(T scalar) : scalar(scalar) {}
  __host__ __device__ T operator()(T x) { return scalar - x; }
};

template <typename T>
struct ScalarDivJetVectorA : public thrust::unary_function<T, T> {
  T scalar;
  explicit ScalarDivJetVectorA(T scalar) : scalar(scalar) {}
  __host__ __device__ T operator()(T x) { return scalar / x; }
};

template <typename T>
struct ScalarDivJetVectorV : public thrust::binary_function<T, T, T> {
  T scalar;
  explicit ScalarDivJetVectorV(T scalar) : scalar(scalar) {}
  __host__ __device__ T operator()(T a, T v) { return -v * scalar / (a * a); }
};

template <typename T>
struct AbsMask : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return x > 0. ? T(1.) : T(-1.); }
};

template <typename T>
struct Sin : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return std::sin(x); }
};

template <typename T>
struct NegativeSinMul : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T a, T v) { return -std::sin(a) * v; }
};

template <typename T>
struct Cos : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return std::cos(x); }
};

template <typename T>
struct CosMul : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T a, T v) { return std::cos(a) * v; }
};

template <typename T>
struct Sqrt : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return std::sqrt(x); }
};

template <typename T>
struct SqrtJetVectorV : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T sqrted_a, T v) {
    return T(0.5) * v / sqrted_a;
  }
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
void jetVectorSubScalarVectorCPU(const MegBA::JetVector<T> &f,
                                 const MegBA::JetVector<T> &g,
                                 MegBA::JetVector<T> *out) {
  out->getCPUGrad() = f.getCPUGrad();

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::minus<T>());
}

template <typename T>
void scalarVectorSubJetVectorCPU(const MegBA::JetVector<T> &f,
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
void scalarVectorSubScalarVectorCPU(const MegBA::JetVector<T> &f,
                                    const MegBA::JetVector<T> &g,
                                    MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::minus<T>());
}

template <typename T>
void vectorSubVectorCPU(const MegBA::JetVector<T> &f,
                        const MegBA::JetVector<T> &g,
                        MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      JetVector_minus_JetVector_CPU(f, g, out);
    } else {
      jetVectorSubScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      scalarVectorSubJetVectorCPU(f, g, out);
    } else {
      scalarVectorSubScalarVectorCPU(f, g, out);
    }
  }
}
template void vectorSubVectorCPU<double>(const MegBA::JetVector<double> &f,
                                         const MegBA::JetVector<double> &g,
                                         MegBA::JetVector<double> *out);

template void vectorSubVectorCPU<float>(const MegBA::JetVector<float> &f,
                                        const MegBA::JetVector<float> &g,
                                        MegBA::JetVector<float> *out);

template <typename T>
void jetVectorMulJetVectorCPU(const MegBA::JetVector<T> &f,
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
                      TT::JetVectorMulJetVectorV<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::multiplies<T>());
}
template <typename T>
void jetVectorMulScalarVectorCPU(const MegBA::JetVector<T> &f,
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
void scalarVectorMulScalarVectorCPU(const MegBA::JetVector<T> &f,
                                    const MegBA::JetVector<T> &g,
                                    MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::multiplies<T>());
}

template <typename T>
void vectorMulVectorCPU(const MegBA::JetVector<T> &f,
                        const MegBA::JetVector<T> &g,
                        MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      jetVectorMulJetVectorCPU(f, g, out);
    } else {
      jetVectorMulScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      jetVectorMulScalarVectorCPU(g, f, out);
    } else {
      scalarVectorMulScalarVectorCPU(f, g, out);
    }
  }
}
template void vectorMulVectorCPU<double>(const MegBA::JetVector<double> &f,
                                         const MegBA::JetVector<double> &g,
                                         MegBA::JetVector<double> *out);

template void vectorMulVectorCPU<float>(const MegBA::JetVector<float> &f,
                                        const MegBA::JetVector<float> &g,
                                        MegBA::JetVector<float> *out);

template <typename T>
void jetVectorDivJetVectorCPU(const MegBA::JetVector<T> &f,
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
                      TT::JetVectorDivJetVectorV<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), inv_ga.begin(),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}

template <typename T>
void jetVectorDivScalarVectorCPU(const MegBA::JetVector<T> &f,
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
void scalarVectorDivJetVectorCPU(const MegBA::JetVector<T> &f,
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
        out->getCPUGrad()[i].begin(), TT::ScalarVectorDivJetVectorV<T>());
  }

  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(), inv_ga.begin(),
                    out->getCPURes().begin(), thrust::multiplies<T>());
}

template <typename T>
void scalarVectorDivScalarVectorCPU(const MegBA::JetVector<T> &f,
                                    const MegBA::JetVector<T> &g,
                                    MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    g.getCPURes().begin(), out->getCPURes().begin(),
                    thrust::divides<T>());
}

template <typename T>
void vectorDivVectorCPU(const MegBA::JetVector<T> &f,
                        const MegBA::JetVector<T> &g,
                        MegBA::JetVector<T> *out) {
  if (f.getGradShape() != 0) {
    if (g.getGradShape() != 0) {
      jetVectorDivJetVectorCPU(f, g, out);
    } else {
      jetVectorDivScalarVectorCPU(f, g, out);
    }
  } else {
    if (g.getGradShape() != 0) {
      scalarVectorDivJetVectorCPU(f, g, out);
    } else {
      scalarVectorDivScalarVectorCPU(f, g, out);
    }
  }
}
template void vectorDivVectorCPU<double>(const MegBA::JetVector<double> &f,
                                         const MegBA::JetVector<double> &g,
                                         MegBA::JetVector<double> *out);

template void vectorDivVectorCPU<float>(const MegBA::JetVector<float> &f,
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
void jetVectorSubScalarCPU(const MegBA::JetVector<T> &f, T g,
                           MegBA::JetVector<T> *out) {
  thrust::transform(f.getCPURes().begin(), f.getCPURes().end(),
                    thrust::make_constant_iterator(g), out->getCPURes().begin(),
                    thrust::minus<T>());

  for (unsigned int i = 0; i < out->getGradShape(); ++i)
    thrust::copy(f.getCPUGrad()[i].begin(), f.getCPUGrad()[i].end(),
                 out->getCPUGrad()[i].begin());
}
template void jetVectorSubScalarCPU<double>(const MegBA::JetVector<double> &f,
                                            double g,
                                            MegBA::JetVector<double> *out);

template void jetVectorSubScalarCPU<float>(const MegBA::JetVector<float> &f,
                                           float g,
                                           MegBA::JetVector<float> *out);

template <typename T>
void jetVectorMulScalarCPU(const MegBA::JetVector<T> &f, T g,
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
template void jetVectorMulScalarCPU<double>(const MegBA::JetVector<double> &f,
                                            double g,
                                            MegBA::JetVector<double> *out);

template void jetVectorMulScalarCPU<float>(const MegBA::JetVector<float> &f,
                                           float g,
                                           MegBA::JetVector<float> *out);

template <typename T>
void jetVectorDivScalarCPU(const MegBA::JetVector<T> &f, T g,
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
template void jetVectorDivScalarCPU<double>(const MegBA::JetVector<double> &f,
                                            double g,
                                            MegBA::JetVector<double> *out);

template void jetVectorDivScalarCPU<float>(const MegBA::JetVector<float> &f,
                                           float g,
                                           MegBA::JetVector<float> *out);

template <typename T>
void scalarSubJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(g.getCPUGrad()[i].begin(), g.getCPUGrad()[i].end(),
                      out->getCPUGrad()[i].begin(), thrust::negate<T>());
  }

  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(),
                    out->getCPURes().begin(), TT::ScalarSubJetVector<T>(f));
}
template void scalarSubJetVectorCPU<double>(double f,
                                            const MegBA::JetVector<double> &g,
                                            MegBA::JetVector<double> *out);

template void scalarSubJetVectorCPU<float>(float f,
                                           const MegBA::JetVector<float> &g,
                                           MegBA::JetVector<float> *out);

template <typename T>
void scalarDivJetVectorCPU(T f, const JetVector<T> &g, JetVector<T> *out) {
  for (unsigned int i = 0; i < out->getGradShape(); ++i) {
    thrust::transform(g.getCPURes().begin(), g.getCPURes().end(),
                      g.getCPUGrad()[i].begin(), out->getCPUGrad()[i].begin(),
                      TT::ScalarDivJetVectorV<T>(f));
  }

  thrust::transform(g.getCPURes().begin(), g.getCPURes().end(),
                    out->getCPURes().begin(), TT::ScalarDivJetVectorA<T>(f));
}
template void scalarDivJetVectorCPU<double>(double f,
                                            const MegBA::JetVector<double> &g,
                                            MegBA::JetVector<double> *out);

template void scalarDivJetVectorCPU<float>(float f,
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
                      TT::NegativeSinMul<T>());
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
                      TT::CosMul<T>());
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
