/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <utility>
#include <iostream>
#include <cassert>
#include <functional>
#include <vector>
#include "common.h"
#include "jet_vector-inl.h"
#include "jet_vector_op-inl.h"
#include "resource/memory_pool.h"
#include "resource/handle_manager.h"

namespace MegBA {
template <typename T> class JetVector {
  // cuda functions
  void initAsCUDA(const JetVector<T> &f);
  void CPU2CUDA(const JetVector<T> &f);
  void CUDA2CUDA(const JetVector<T> &f);
  void CUDA2CPU(const JetVector<T> &f);

  unsigned int _N{0};
  unsigned int _nItem{0};
  Device _device = Device::CPU;
  std::vector<std::vector<T>> _gradHostVec{};
  std::vector<T *> _gradDevicePtr{};
  std::vector<T> _valueHostVec{};
  std::vector<T *> _valueDevicePtr{};
  int _gradPosition = -1;
  bool _pureScalarFlag = false;
  T _pureScalar = 0;

 public:
  JetVector() = default;

  explicit JetVector(T scalar) : _pureScalarFlag(true), _pureScalar(scalar) {}

  JetVector(const JetVector<T> &f)
      : _N{f._N}, _nItem{f._nItem}, _device{f._device},
        _valueHostVec{f._valueHostVec},
        _gradHostVec{f._gradHostVec} {
    switch (_device) {
      case Device::CPU:
        break;
      case Device::CUDA:
        CUDA2CUDA(f);
        break;
    }
  }

  JetVector(JetVector<T> &&f) noexcept
      : _N{f._N}, _nItem{f._nItem},
        _device{f._device}, _gradHostVec{std::move(f._gradHostVec)},
        _gradDevicePtr{std::move(f._gradDevicePtr)},
        _valueHostVec{std::move(f._valueHostVec)},
        _valueDevicePtr{std::move(f._valueDevicePtr)}, _gradPosition{std::move(f._gradPosition)},
        _pureScalarFlag{f._pureScalarFlag},
        _pureScalar{f._pureScalar} {
    f._N = 0;
    f._nItem = 0;
    f._gradPosition = -1;
  }

  template <typename F>
  JetVector(const JetVector<T> &init_template, F &&math_func) {
    initAs(init_template);
    math_func(this);
  }

  ~JetVector() { clear(); }

  void appendJet(T a, int n);
  void appendJet(T a);
  void clear();

  static const JetVector<T> &getInitTemplate(const JetVector<T> &f,
                                             const JetVector<T> &g) {
    return f._N > g._N ? f : g;
  }

  void initAs(const JetVector<T> &initTemplate);
  JetVector<T> &to(Device device);
  JetVector<T> &CPU();
  JetVector<T> &CUDA();
  bool IsEmpty();
  void set_Grad_Shape(unsigned int N);
  void erase(std::size_t idx) {
    assert(_device == Device::CPU || _gradPosition != -1 || _N == 0);
    _valueHostVec.erase(_valueHostVec.begin() + idx);
    _nItem--;
  }
  const unsigned int &getGradShape() const { return _N; }
  const unsigned int &getItemNum() const { return _nItem; }
  std::size_t getItemNum(int rank) const { return MemoryPool::getItemNum(rank); }
  int getGradPosition() const { return _gradPosition; }
  const Device &getDevice() const { return _device; }

  const std::vector<std::vector<T>> &getCPUGrad() const { return _gradHostVec; }
  std::vector<std::vector<T>> &getCPUGrad() { return _gradHostVec; }

  const std::vector<T> &getCPURes() const { return _valueHostVec; }
  std::vector<T> &getCPURes() { return _valueHostVec; }

  const std::vector<T *> &getCUDAGradPtr() const { return _gradDevicePtr; }
  const std::vector<T *> &getCUDAResPtr() const { return _valueDevicePtr; }

  void bindValueDevicePtr(std::vector<T *> &&valueDevicePtr) {
    _valueDevicePtr = std::move(valueDevicePtr); }

  void setGradPosition(int gradPosition);

  JetVector<T> &operator=(const JetVector<T> &f);

  JetVector<T> &operator=(JetVector<T> &&f) noexcept;

  JetVector<T> operator+(const JetVector<T> &g) const;

  JetVector<T> operator-(const JetVector<T> &g) const;

  JetVector<T> operator*(const JetVector<T> &g) const;

  JetVector<T> operator/(const JetVector<T> &g) const;

  JetVector<T> &operator+=(const JetVector<T> &g);

  JetVector<T> &operator-=(const JetVector<T> &g);

  JetVector<T> &operator*=(const JetVector<T> &g);

  JetVector<T> &operator/=(const JetVector<T> &g);

  JetVector<T> operator-() const;

  JetVector<T> operator+(T g) const;

  JetVector<T> operator-(T g) const;

  JetVector<T> operator*(T g) const;

  JetVector<T> operator/(T g) const;

  JetVector<T> &operator+=(T g);

  JetVector<T> &operator-=(T g);

  JetVector<T> &operator*=(T g);

  JetVector<T> &operator/=(T g);

  JetVector<T> scalarSubThis(T g) const;
  JetVector<T> scalarDivThis(T g) const;
};

template <typename T>
std::ostream &ostreamCUDA(std::ostream &s, const JetVector<T> &z);
}  // namespace MegBA

template <typename T>
std::ostream &operator<<(std::ostream &s, const MegBA::JetVector<T> &z);

//namespace Eigen {
//namespace internal {
//template <typename T, int Rows_, int Cols_, int Options_, int MaxRows_,
//          int MaxCols_>
//struct traits<
//    Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
// private:
//  using Scalar_ = MegBA::JetVector<T>;
//  enum { size = internal::size_at_compile_time<Rows_, Cols_>::ret };
//  typedef typename find_best_packet<Scalar_, size>::type PacketScalar;
//  enum {
//    row_major_bit = Options_ & RowMajor ? RowMajorBit : 0,
//    is_dynamic_size_storage = MaxRows_ == Dynamic || MaxCols_ == Dynamic,
//    max_size = is_dynamic_size_storage ? Dynamic : MaxRows_ * MaxCols_,
//    default_alignment = compute_default_alignment<Scalar_, max_size>::value,
//    actual_alignment = ((Options_ & DontAlign) == 0) ? default_alignment : 0,
//    required_alignment = unpacket_traits<PacketScalar>::alignment,
//    packet_access_bit = (packet_traits<Scalar_>::Vectorizable &&
//                         (EIGEN_UNALIGNED_VECTORIZE ||
//                          (actual_alignment >= required_alignment)))
//                            ? PacketAccessBit
//                            : 0
//  };
//
// public:
//  typedef Scalar_ Scalar;
//  typedef Dense StorageKind;
//  typedef Eigen::Index StorageIndex;
//  typedef MatrixXpr XprKind;
//  enum {
//    RowsAtCompileTime = Rows_,
//    ColsAtCompileTime = Cols_,
//    MaxRowsAtCompileTime = MaxRows_,
//    MaxColsAtCompileTime = MaxCols_,
//    Flags = compute_matrix_flags<Scalar_, Rows_, Cols_, Options_, MaxRows_,
//                                 MaxCols_>::ret,
//    Options = Options_,
//    InnerStrideAtCompileTime = 1,
//    OuterStrideAtCompileTime =
//        (Options & RowMajor) ? ColsAtCompileTime : RowsAtCompileTime,
//
//    // FIXME, the following flag in only used to define NeedsToAlign in
//    PlainObjectBase EvaluatorFlags =
//        LinearAccessBit | DirectAccessBit | packet_access_bit | row_major_bit,
//    Alignment = actual_alignment
//  };
//};
//
//template <typename T, int Rows_, int Cols_, int Options_, int MaxRows_,
//          int MaxCols_, int MapOptions, typename StrideType>
//struct traits<Map<const Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_,
//                               MaxRows_, MaxCols_>,
//                  MapOptions, StrideType>>
//    : public traits<const Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_,
//                                 MaxRows_, MaxCols_>> {
//  using PlainObjectType = const Matrix<MegBA::JetVector<T>, Rows_, Cols_,
//                                       Options_, MaxRows_, MaxCols_>;
//  typedef traits<PlainObjectType> TraitsBase;
//  enum {
//    PlainObjectTypeInnerSize =
//        ((traits<PlainObjectType>::Flags & RowMajorBit) == RowMajorBit)
//            ? PlainObjectType::ColsAtCompileTime
//            : PlainObjectType::RowsAtCompileTime,
//
//    InnerStrideAtCompileTime =
//        StrideType::InnerStrideAtCompileTime == 0
//            ? int(PlainObjectType::InnerStrideAtCompileTime)
//            : int(StrideType::InnerStrideAtCompileTime),
//    OuterStrideAtCompileTime =
//        StrideType::OuterStrideAtCompileTime == 0
//            ? (InnerStrideAtCompileTime == Dynamic ||
//                       PlainObjectTypeInnerSize == Dynamic
//                   ? Dynamic
//                   : int(InnerStrideAtCompileTime) *
//                         int(PlainObjectTypeInnerSize))
//            : int(StrideType::OuterStrideAtCompileTime),
//    Alignment = int(MapOptions) & int(AlignedMask),
//    Flags = int(TraitsBase::Flags & (~NestByRefBit))
//  };
//
// private:
//  enum { Options };  // Expressions don't have Options
//};
//
//template <typename PlainObjectType, int MapOptions, typename StrideType>
//class Map;
//
//template <typename T, int Rows_, int Cols_, int Options_, int MaxRows_,
//          int MaxCols_, int MapOptions, typename StrideType>
//class Map<const Matrix<MegBA::JetVector<T>, Rows_, Cols_, Options_, MaxRows_,
//                       MaxCols_>,
//          MapOptions, StrideType>
//    : public MapBase<Map<const Matrix<MegBA::JetVector<T>, Rows_, Cols_,
//                                      Options_, MaxRows_, MaxCols_>,
//                         MapOptions, StrideType>> {
// public:
//  typedef MapBase<Map> Base;
//  EIGEN_DENSE_PUBLIC_INTERFACE(Map)
//
//  typedef const typename Base::PointerType PointerType;
//  typedef PointerType PointerArgType;
//  EIGEN_DEVICE_FUNC
//  inline PointerType cast_to_pointer_type(PointerArgType ptr) { return ptr; }
//
//  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index innerStride() const {
//    return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
//  }
//
//  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index outerStride() const {
//    return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
//           : internal::traits<Map>::OuterStrideAtCompileTime != Dynamic
//               ? Index(internal::traits<Map>::OuterStrideAtCompileTime)
//               : IsVectorAtCompileTime    ? (this->size() * innerStride())
//                 : int(Flags) & RowMajorBit ? (this->cols() * innerStride())
//                                            : (this->rows() * innerStride());
//  }
//
//  /** Constructor in the fixed-size case.
//       *
//       * \param dataPtr pointer to the array to map
//       * \param stride optional Stride object, passing the strides.
//   */
//  EIGEN_DEVICE_FUNC
//  explicit inline Map(PointerArgType dataPtr,
//                      const StrideType& stride = StrideType())
//      : Base(cast_to_pointer_type(dataPtr)), m_stride(stride) {}
//
//  /** Constructor in the dynamic-size vector case.
//       *
//       * \param dataPtr pointer to the array to map
//       * \param size the size of the vector expression
//       * \param stride optional Stride object, passing the strides.
//   */
//  EIGEN_DEVICE_FUNC
//  inline Map(PointerArgType dataPtr, Index size,
//             const StrideType& stride = StrideType())
//      : Base(cast_to_pointer_type(dataPtr), size), m_stride(stride) {}
//
//  /** Constructor in the dynamic-size matrix case.
//       *
//       * \param dataPtr pointer to the array to map
//       * \param rows the number of rows of the matrix expression
//       * \param cols the number of columns of the matrix expression
//       * \param stride optional Stride object, passing the strides.
//   */
//  EIGEN_DEVICE_FUNC
//  inline Map(PointerArgType dataPtr, Index rows, Index cols,
//             const StrideType& stride = StrideType())
//      : Base(cast_to_pointer_type(dataPtr), rows, cols), m_stride(stride) {}
//
//  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
//
// protected:
//  StrideType m_stride;
//};
//}  // namespace internal
//}  // namespace Eigen
