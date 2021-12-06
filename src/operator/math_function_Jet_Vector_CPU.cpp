/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "thrust/copy.h"
#include "thrust/transform.h"
#include <thrust/iterator/constant_iterator.h>
#include "operator/JetVector.h"
#include "operator/Thrust_Transform.h"
#include "operator/math_function_Jet_Vector_CPU.h"
namespace MegBA {
    namespace math {
        namespace function {
            template<typename T>
            void JetVector_add_JetVector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                           MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                      g.get_CPU_Grad()[i].begin(),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::plus<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::plus<T>());
            }

            template<typename T>
            void JetVector_add_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                  MegBA::JetVector<T> &out) {
                out.get_CPU_Grad() = f.get_CPU_Grad();

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::plus<T>());
            }

            template<typename T>
            void Scalar_Vector_add_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::plus<T>());
            }

            template<typename T>
            void Vector_add_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                       MegBA::JetVector<T> &out) {
                if (f.get_Grad_Shape() != 0)
                    if (g.get_Grad_Shape() != 0)
                        JetVector_add_JetVector_CPU(f, g, out);
                    else
                        JetVector_add_Scalar_Vector_CPU(f, g, out);
                else
                    if (g.get_Grad_Shape() != 0)
                        JetVector_add_Scalar_Vector_CPU(g, f, out);
                    else
                        Scalar_Vector_add_Scalar_Vector_CPU(f, g, out);
            }
            template void Vector_add_Vector_CPU<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                                          MegBA::JetVector<double> &out);

            template void Vector_add_Vector_CPU<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                                         MegBA::JetVector<float> &out);

            template<typename T>
            void JetVector_minus_JetVector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                      g.get_CPU_Grad()[i].begin(),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::minus<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::minus<T>());
            }

            template<typename T>
            void JetVector_minus_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                               MegBA::JetVector<T> &out) {
                out.get_CPU_Grad() = f.get_CPU_Grad();

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::minus<T>());
            }

            template<typename T>
            void Scalar_Vector_minus_JetVector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                               MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(g.get_CPU_Grad()[i].begin(), g.get_CPU_Grad()[i].end(),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::negate<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::minus<T>());
            }

            template<typename T>
            void Scalar_Vector_minus_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                  MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::minus<T>());
            }

            template<typename T>
            void Vector_minus_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                         MegBA::JetVector<T> &out) {
                if (f.get_Grad_Shape() != 0)
                    if (g.get_Grad_Shape() != 0)
                        JetVector_minus_JetVector_CPU(f, g, out);
                    else
                        JetVector_minus_Scalar_Vector_CPU(f, g, out);
                else
                    if (g.get_Grad_Shape() != 0)
                        Scalar_Vector_minus_JetVector_CPU(f, g, out);
                    else
                        Scalar_Vector_minus_Scalar_Vector_CPU(f, g, out);
            }
            template void Vector_minus_Vector_CPU<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                                            MegBA::JetVector<double> &out);

            template void Vector_minus_Vector_CPU<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                                           MegBA::JetVector<float> &out);

            template<typename T>
            void
            JetVector_multiplies_JetVector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            thrust::make_zip_iterator(thrust::make_tuple(f.get_CPU_Res().begin(),
                                                                         f.get_CPU_Grad()[i].begin(),
                                                                         g.get_CPU_Res().begin(),
                                                                         g.get_CPU_Grad()[i].begin())),
                            thrust::make_zip_iterator(thrust::make_tuple(f.get_CPU_Res().end(),
                                                                         f.get_CPU_Grad()[i].end(),
                                                                         g.get_CPU_Res().end(),
                                                                         g.get_CPU_Grad()[i].end())),
                            out.get_CPU_Grad()[i].begin(),
                      MegBA::TT::JetVector_multiplies_JetVector_v<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }
            template<typename T>
            void
            JetVector_multiplies_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                      g.get_CPU_Res().begin(),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }

            template<typename T>
            void
            Scalar_Vector_multiplies_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }

            template<typename T>
            void Vector_multiplies_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                              MegBA::JetVector<T> &out) {
                if (f.get_Grad_Shape() != 0)
                    if (g.get_Grad_Shape() != 0)
                        JetVector_multiplies_JetVector_CPU(f, g, out);
                    else
                        JetVector_multiplies_Scalar_Vector_CPU(f, g, out);
                else
                    if (g.get_Grad_Shape() != 0)
                        JetVector_multiplies_Scalar_Vector_CPU(g, f, out);
                    else
                        Scalar_Vector_multiplies_Scalar_Vector_CPU(f, g, out);
            }
            template void Vector_multiplies_Vector_CPU<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Vector_multiplies_Vector_CPU<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                MegBA::JetVector<float> &out);

            template<typename T>
            void JetVector_divides_JetVector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                              MegBA::JetVector<T> &out) {
                std::vector<T> inv_ga(f.get_CPU_Res().size());
                thrust::transform(g.get_CPU_Res().begin(), g.get_CPU_Res().end(), inv_ga.begin(),
                                  MegBA::TT::inverse<T>());
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            thrust::make_zip_iterator(thrust::make_tuple(f.get_CPU_Res().begin(),
                                                                         f.get_CPU_Grad()[i].begin(),
                                                                         inv_ga.begin(),
                                                                         g.get_CPU_Grad()[i].begin())),
                            thrust::make_zip_iterator(thrust::make_tuple(f.get_CPU_Res().end(),
                                                                         f.get_CPU_Grad()[i].end(),
                                                                         inv_ga.end(),
                                                                         g.get_CPU_Grad()[i].end())),
                            out.get_CPU_Grad()[i].begin(),
                      MegBA::TT::JetVector_divides_JetVector_v<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  inv_ga.begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }

            template<typename T>
            void
            JetVector_divides_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> &out) {
                std::vector<T> inv_ga(f.get_CPU_Res().size());
                thrust::transform(g.get_CPU_Res().begin(), g.get_CPU_Res().end(), inv_ga.begin(),
                                  MegBA::TT::inverse<T>());
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                      inv_ga.begin(),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  inv_ga.begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }

            template<typename T>
            void Scalar_Vector_divides_JetVector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                                 MegBA::JetVector<T> &out) {
                std::vector<T> inv_ga(f.get_CPU_Res().size());
                thrust::transform(g.get_CPU_Res().begin(), g.get_CPU_Res().end(), inv_ga.begin(),
                                  MegBA::TT::inverse<T>());
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            thrust::make_zip_iterator(thrust::make_tuple(f.get_CPU_Res().begin(),
                                                                         inv_ga.begin(),
                                                                         g.get_CPU_Grad()[i].begin())),
                            thrust::make_zip_iterator(thrust::make_tuple(f.get_CPU_Res().end(),
                                                                      inv_ga.end(),
                                                                      g.get_CPU_Grad()[i].end())),
                                                                      out.get_CPU_Grad()[i].begin(),
                      MegBA::TT::Scalar_Vector_divides_JetVector_v<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  inv_ga.begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }

            template<typename T>
            void Scalar_Vector_divides_Scalar_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  g.get_CPU_Res().begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::divides<T>());
            }

            template<typename T>
            void Vector_divides_Vector_CPU(const MegBA::JetVector<T> &f, const MegBA::JetVector<T> &g,
                                           MegBA::JetVector<T> &out) {
                if (f.get_Grad_Shape() != 0)
                    if (g.get_Grad_Shape() != 0)
                        JetVector_divides_JetVector_CPU(f, g, out);
                    else
                        JetVector_divides_Scalar_Vector_CPU(f, g, out);
                else
                    if (g.get_Grad_Shape() != 0)
                        Scalar_Vector_divides_JetVector_CPU(f, g, out);
                    else
                        Scalar_Vector_divides_Scalar_Vector_CPU(f, g, out);
            }
            template void Vector_divides_Vector_CPU<double>(
                    const MegBA::JetVector<double> &f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Vector_divides_Vector_CPU<float>(
                    const MegBA::JetVector<float> &f, const MegBA::JetVector<float> &g,
                                             MegBA::JetVector<float> &out);

            template<typename T>
            void JetVector_add_Scalar_CPU(const MegBA::JetVector<T> &f, T g,
                                           MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  thrust::make_constant_iterator(g),
                                  out.get_CPU_Res().begin(),
                                  thrust::plus<T>());
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i)
                    thrust::copy(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                 out.get_CPU_Grad()[i].begin());
            }
            template void JetVector_add_Scalar_CPU<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_add_Scalar_CPU<float>(
                    const MegBA::JetVector<float> &f, float g,
                                             MegBA::JetVector<float> &out);

            template<typename T>
            void JetVector_minus_Scalar_CPU(const MegBA::JetVector<T> &f, T g,
                                             MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  thrust::make_constant_iterator(g),
                                  out.get_CPU_Res().begin(),
                                  thrust::minus<T>());

                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i)
                    thrust::copy(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                 out.get_CPU_Grad()[i].begin());
            }
            template void JetVector_minus_Scalar_CPU<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_minus_Scalar_CPU<float>(
                    const MegBA::JetVector<float> &f, float g,
                MegBA::JetVector<float> &out);

            template<typename T>
            void JetVector_multiplies_Scalar_CPU(const MegBA::JetVector<T> &f, T g,
                                                  MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                      thrust::make_constant_iterator(g),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  thrust::make_constant_iterator(g),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }
            template void JetVector_multiplies_Scalar_CPU<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_multiplies_Scalar_CPU<float>(
                    const MegBA::JetVector<float> &f, float g,
                MegBA::JetVector<float> &out);

            template<typename T>
            void JetVector_divides_Scalar_CPU(const MegBA::JetVector<T> &f, T g, MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                                      thrust::make_constant_iterator(T(1.) / g),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  thrust::make_constant_iterator(T(1.) / g),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }
            template void JetVector_divides_Scalar_CPU<double>(
                    const MegBA::JetVector<double> &f, double g,
                MegBA::JetVector<double> &out);

            template void JetVector_divides_Scalar_CPU<float>(
                    const MegBA::JetVector<float> &f, float g,
                MegBA::JetVector<float> &out);

            template<typename T>
            void Scalar_minus_JetVector_CPU(T f, const JetVector<T> &g,
                                             JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(g.get_CPU_Grad()[i].begin(), g.get_CPU_Grad()[i].end(),
                                      out.get_CPU_Grad()[i].begin(),
                                      thrust::negate<T>());
                }

                thrust::transform(g.get_CPU_Res().begin(), g.get_CPU_Res().end(),
                                  out.get_CPU_Res().begin(),
                                  TT::scalar_minus_JetVector<T>(f));
            }
            template void Scalar_minus_JetVector_CPU<double>(
                    double f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Scalar_minus_JetVector_CPU<float>(
                    float f, const MegBA::JetVector<float> &g,
                MegBA::JetVector<float> &out);

            template<typename T>
            void Scalar_divides_JetVector_CPU(T f, const JetVector<T> &g,
                                               JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(g.get_CPU_Res().begin(), g.get_CPU_Res().end(),
                                      g.get_CPU_Grad()[i].begin(),
                                      out.get_CPU_Grad()[i].begin(),
                                      TT::scalar_divides_JetVector_v<T>(f));
                }

                thrust::transform(g.get_CPU_Res().begin(), g.get_CPU_Res().end(),
                                  out.get_CPU_Res().begin(),
                                  TT::scalar_divides_JetVector_a<T>(f));
            }
            template void Scalar_divides_JetVector_CPU<double>(
                    double f, const MegBA::JetVector<double> &g,
                MegBA::JetVector<double> &out);

            template void Scalar_divides_JetVector_CPU<float>(
                    float f, const MegBA::JetVector<float> &g,
                MegBA::JetVector<float> &out);

            template<typename T>
            void abs_JetVector_CPU(const MegBA::JetVector<T> &f,
                                    MegBA::JetVector<T> &out) {
                std::vector<T> mask(f.get_CPU_Res().size());
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(), mask.begin(),
                                  MegBA::TT::abs_mask<T>());
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            f.get_CPU_Grad()[i].begin(), f.get_CPU_Grad()[i].end(),
                            mask.begin(),
                            out.get_CPU_Grad()[i].begin(),
                            thrust::multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  mask.begin(),
                                  out.get_CPU_Res().begin(),
                                  thrust::multiplies<T>());
            }
            template void abs_JetVector_CPU<double>(
                    const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> &out);

            template void abs_JetVector_CPU<float>(
                    const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> &out);

            template<typename T>
            void cos_JetVector_CPU(const MegBA::JetVector<T> &f,
                                    MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                            f.get_CPU_Grad()[i].begin(),
                            out.get_CPU_Grad()[i].begin(),
                            TT::negative_sin_multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  out.get_CPU_Res().begin(),
                                  TT::cos<T>());
            }
            template void cos_JetVector_CPU<double>(
                    const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> &out);

            template void cos_JetVector_CPU<float>(
                    const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> &out);

            template<typename T>
            void sin_JetVector_CPU(const MegBA::JetVector<T> &f,
                                    MegBA::JetVector<T> &out) {
                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                            f.get_CPU_Grad()[i].begin(),
                            out.get_CPU_Grad()[i].begin(),
                            TT::cos_multiplies<T>());
                }

                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  out.get_CPU_Res().begin(),
                                  TT::sin<T>());
            }
            template void sin_JetVector_CPU<double>(
                    const MegBA::JetVector<double> &f,
                                       MegBA::JetVector<double> &out);

            template void sin_JetVector_CPU<float>(
                    const MegBA::JetVector<float> &f,
                                      MegBA::JetVector<float> &out);

            template<typename T>
            void sqrt_JetVector_CPU(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out) {
                thrust::transform(f.get_CPU_Res().begin(), f.get_CPU_Res().end(),
                                  out.get_CPU_Res().begin(),
                                  TT::sqrt<T>());

                for (unsigned int i = 0; i < out.get_Grad_Shape(); ++i) {
                    thrust::transform(
                            out.get_CPU_Res().begin(), out.get_CPU_Res().end(),
                            f.get_CPU_Grad()[i].begin(),
                            out.get_CPU_Grad()[i].begin(),
                            TT::sqrt_JetVector_v<T>());
                }
            }
            template void sqrt_JetVector_CPU<double>(
                    const MegBA::JetVector<double> &f,
                                        MegBA::JetVector<double> &out);

            template void sqrt_JetVector_CPU<float>(
                    const MegBA::JetVector<float> &f,
                                       MegBA::JetVector<float> &out);
        }
    }
}  // namespace MegBA
