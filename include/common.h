/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#define SPECIALIZE_CLASS(className) \
template class className<double>;   \
template class className<float>

#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <set>

namespace MegBA {
enum Device { CPU, CUDA };

enum AlgoKind { BASE_ALGO, LM };

enum LinearSystemKind { BASE_LINEAR_SYSTEM, SCHUR };

enum SolverKind { BASE_SOLVER, PCG };

struct SolverOption{
  struct SolverOptionPCG {
    int maxIter{100};
    double tol{1e-1};
    double refuseRatio{1e0};
  } solverOptionPCG;
};

struct AlgoOption {
  struct AlgoOptionLM {
    int maxIter{20};
    double initialRegion{1e3};
    double epsilon1{1};
    double epsilon2{1e-10};
  } algoOptionLM;
};

struct ProblemOption {
  bool useSchur{true};
  Device device{Device::CUDA};
  std::set<int> deviceUsed{};
  int N{-1};
  int64_t nItem{-1};
  AlgoKind algoKind{LM};
  LinearSystemKind linearSystemKind{SCHUR};
  SolverKind solverKind{PCG};
};

struct AlgoStatus {
  struct AlgoStatusLM {
    double region;
    bool recoverDiag{false};
  } algoStatusLM;
};

template <typename T>
class JetVector;

template <typename T>
class BaseProblem;

template <typename T>
class BaseVertex;

template <typename T>
class BaseEdge;

template <typename T>
class EdgeVector;

template <typename T>
class BaseAlgo;

template <typename T>
class BaseSolver;

template <typename T>
class BaseLinearSystem;

template <typename T>
using JVD = Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic>;
}  // namespace MegBA

namespace Eigen {
namespace internal {
// template<typename T, int Rows_, int Cols_, int Options_, int MaxRows_, int
// MaxCols_> struct traits<Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_,
// MaxRows_, MaxCols_> >
//{
// private:
//   using Scalar_ = MegBA::JetVector<T>;
//   enum { size = internal::size_at_compile_time<Rows_,Cols_>::ret };
//   typedef typename find_best_packet<Scalar_,size>::type PacketScalar;
//   enum {
//     row_major_bit = Options_&RowMajor ? RowMajorBit : 0,
//     is_dynamic_size_storage = MaxRows_== Dynamic || MaxCols_== Dynamic,
//     max_size = is_dynamic_size_storage ? Dynamic : MaxRows_ * MaxCols_,
//     default_alignment = compute_default_alignment<Scalar_,max_size>::value,
//     actual_alignment = ((Options_&DontAlign)==0) ? default_alignment : 0,
//     required_alignment = unpacket_traits<PacketScalar>::alignment,
//     packet_access_bit = (packet_traits<Scalar_>::Vectorizable &&
//     (EIGEN_UNALIGNED_VECTORIZE || (actual_alignment>=required_alignment))) ?
//     PacketAccessBit : 0
//   };
//
// public:
//   typedef Scalar_ Scalar;
//   typedef Dense StorageKind;
//   typedef Eigen::Index StorageIndex;
//   typedef MatrixXpr XprKind;
//   enum {
//     RowsAtCompileTime = Rows_,
//     ColsAtCompileTime = Cols_,
//     MaxRowsAtCompileTime = MaxRows_,
//     MaxColsAtCompileTime = MaxCols_,
//     Flags = compute_matrix_flags<Scalar_, Rows_, Cols_, Options_, MaxRows_,
//     MaxCols_>::ret, Options = Options_, InnerStrideAtCompileTime = 1,
//     OuterStrideAtCompileTime = (Options&RowMajor) ? ColsAtCompileTime :
//     RowsAtCompileTime,
//
//     // FIXME, the following flag in only used to define NeedsToAlign in
//     PlainObjectBase EvaluatorFlags = LinearAccessBit | DirectAccessBit |
//     packet_access_bit | row_major_bit, Alignment = actual_alignment
//   };
// };

// template<typename T, int Rows_, int Cols_, int Options_, int MaxRows_, int
// MaxCols_, int MapOptions, typename StrideType> struct traits<Map<const
// Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
// MapOptions, StrideType> >
//     : public traits<const Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_,
//     MaxRows_, MaxCols_>>
//{
//   using PlainObjectType = const Matrix<MegBA::JetVector<T>, Rows_, Cols_,
//   Options_, MaxRows_, MaxCols_>; typedef traits<PlainObjectType> TraitsBase;
//   enum {
//     PlainObjectTypeInnerSize =
//     ((traits<PlainObjectType>::Flags&RowMajorBit)==RowMajorBit)
//                                    ? PlainObjectType::ColsAtCompileTime
//                                    : PlainObjectType::RowsAtCompileTime,
//
//     InnerStrideAtCompileTime = StrideType::InnerStrideAtCompileTime == 0
//                                    ?
//                                    int(PlainObjectType::InnerStrideAtCompileTime)
//                                    :
//                                    int(StrideType::InnerStrideAtCompileTime),
//     OuterStrideAtCompileTime = StrideType::OuterStrideAtCompileTime == 0
//                                    ? (InnerStrideAtCompileTime==Dynamic ||
//                                    PlainObjectTypeInnerSize==Dynamic
//                                           ? Dynamic
//                                           : int(InnerStrideAtCompileTime) *
//                                           int(PlainObjectTypeInnerSize))
//                                    :
//                                    int(StrideType::OuterStrideAtCompileTime),
//     Alignment = int(MapOptions)&int(AlignedMask),
//     Flags = int(TraitsBase::Flags & (~NestByRefBit))
//   };
// private:
//   enum { Options }; // Expressions don't have Options
// };
//
// template<typename PlainObjectType, int MapOptions, typename StrideType> class
// Map;
//
// template<typename T, int Rows_, int Cols_, int Options_, int MaxRows_, int
// MaxCols_, int MapOptions, typename StrideType> class Map<const
// Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
// MapOptions, StrideType>
//     : public MapBase<Map<const Matrix<MegBA::JetVector<T>, Rows_, Cols_,
//     Options_, MaxRows_, MaxCols_>, MapOptions, StrideType> >
//{
// public:
//
//   typedef MapBase<Map> Base;
//   EIGEN_DENSE_PUBLIC_INTERFACE(Map)
//
//   typedef const typename Base::PointerType PointerType;
//   typedef PointerType PointerArgType;
//   EIGEN_DEVICE_FUNC
//   inline PointerType cast_to_pointer_type(PointerArgType ptr) { return ptr; }
//
//   EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
//       inline Index innerStride() const
//   {
//     return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
//   }
//
//   EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
//       inline Index outerStride() const
//   {
//     return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
//            : internal::traits<Map>::OuterStrideAtCompileTime != Dynamic ?
//            Index(internal::traits<Map>::OuterStrideAtCompileTime) :
//            IsVectorAtCompileTime ? (this->size() * innerStride()) :
//            int(Flags)&RowMajorBit ? (this->cols() * innerStride())
//                                                                         :
//                                                                         (this->rows()
//                                                                         * innerStride());
//   }
//
//   /** Constructor in the fixed-size case.
//       *
//       * \param dataPtr pointer to the array to map
//       * \param stride optional Stride object, passing the strides.
//    */
//   EIGEN_DEVICE_FUNC
//   explicit inline Map(PointerArgType dataPtr, const StrideType& stride =
//   StrideType())
//       : Base(cast_to_pointer_type(dataPtr)), m_stride(stride)
//   {
//   }
//
//   /** Constructor in the dynamic-size vector case.
//       *
//       * \param dataPtr pointer to the array to map
//       * \param size the size of the vector expression
//       * \param stride optional Stride object, passing the strides.
//    */
//   EIGEN_DEVICE_FUNC
//   inline Map(PointerArgType dataPtr, Index size, const StrideType& stride =
//   StrideType())
//       : Base(cast_to_pointer_type(dataPtr), size), m_stride(stride)
//   {
//   }
//
//   /** Constructor in the dynamic-size matrix case.
//       *
//       * \param dataPtr pointer to the array to map
//       * \param rows the number of rows of the matrix expression
//       * \param cols the number of columns of the matrix expression
//       * \param stride optional Stride object, passing the strides.
//    */
//   EIGEN_DEVICE_FUNC
//   inline Map(PointerArgType dataPtr, Index rows, Index cols, const
//   StrideType& stride = StrideType())
//       : Base(cast_to_pointer_type(dataPtr), rows, cols), m_stride(stride)
//   {
//   }
//
//   EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
//
// protected:
//   StrideType m_stride;
// };
}  // namespace internal
}  // namespace Eigen
