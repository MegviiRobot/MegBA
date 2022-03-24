/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <Eigen/Core>

namespace Eigen {
namespace internal {
template <typename MegBA_t>
struct scalar_constant_op<MegBA::JetVector<MegBA_t>> {
  using Scalar = MegBA::JetVector<MegBA_t>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  scalar_constant_op(const scalar_constant_op &other)
      : m_other(other.m_other) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_constant_op(const Scalar &other)
      : m_other(other) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar &operator()() const {
    return m_other;
  }
  template <typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const PacketType packetOp() const {
    return internal::pset1<PacketType>(m_other);
  }
  const Scalar &m_other;
};

template <typename MegBA_t>
struct scalar_constant_op<const MegBA::JetVector<MegBA_t>> {
  using Scalar = const MegBA::JetVector<MegBA_t>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  scalar_constant_op(const scalar_constant_op &other)
      : m_other(other.m_other) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_constant_op(const Scalar &other)
      : m_other(other) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar &operator()() const {
    return m_other;
  }
  template <typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const PacketType packetOp() const {
    return internal::pset1<PacketType>(m_other);
  }
  const Scalar &m_other;
};

template <typename MegBA_t>
struct assign_op<MegBA::JetVector<MegBA_t>, MegBA::JetVector<MegBA_t>> {
  using DstScalar = MegBA::JetVector<MegBA_t>;
  using SrcScalar = MegBA::JetVector<MegBA_t>;

  EIGEN_EMPTY_STRUCT_CTOR(assign_op)
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void assignCoeff(
      DstScalar &a, const SrcScalar &b) {
    a = b;
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void assignCoeff(DstScalar &a,
                                                                SrcScalar &&b) {
    a = std::move(b);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE void assignCoeff(
      DstScalar &a, const SrcScalar &&b) {
    a = std::move(const_cast<SrcScalar &&>(b));
  }

  template <int Alignment, typename Packet>
  EIGEN_STRONG_INLINE void assignPacket(DstScalar *a, const Packet &b) const {
    internal::pstoret<DstScalar, Packet, Alignment>(a, b);
  }
};

template <typename MegBA_t, int... MatrixArgs>
struct is_lvalue<Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs>
struct is_lvalue<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs, int MapOptions,
          typename StrideType>
struct is_lvalue<Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                     MapOptions, StrideType>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs, int MapOptions,
          typename StrideType>
struct is_lvalue<
    const Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
              MapOptions, StrideType>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs, int BlockRows, int BlockCols,
          bool InnerPanel>
struct is_lvalue<Block<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                       BlockRows, BlockCols, InnerPanel>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs, int BlockRows, int BlockCols,
          bool InnerPanel>
struct is_lvalue<
    const Block<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                BlockRows, BlockCols, InnerPanel>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs, int MapOptions,
          typename StrideType, int BlockRows, int BlockCols, bool InnerPanel>
struct is_lvalue<
    Block<const Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                    MapOptions, StrideType>,
          BlockRows, BlockCols, InnerPanel>> {
  enum { value = true };
};

template <typename MegBA_t, int... MatrixArgs, int MapOptions,
          typename StrideType, int BlockRows, int BlockCols, bool InnerPanel>
struct is_lvalue<const Block<
    const Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
              MapOptions, StrideType>,
    BlockRows, BlockCols, InnerPanel>> {
  enum { value = true };
};

template <typename MegBA_t, typename NullaryOp, int... MatrixArgs>
struct is_lvalue<CwiseNullaryOp<
    NullaryOp, const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>>> {
  enum { value = true };
};
}  // namespace internal

template <typename MegBA_t, int... MatrixArgs, int MapOptions,
          typename StrideType>
class MapBase<Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                  MapOptions, StrideType>,
              ReadOnlyAccessors>
    : public internal::dense_xpr_base<
          Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
              MapOptions, StrideType>>::type {
  using Derived = Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                      MapOptions, StrideType>;

 public:
  typedef typename internal::dense_xpr_base<Derived>::type Base;
  enum {
    RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
    ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
    InnerStrideAtCompileTime =
        internal::traits<Derived>::InnerStrideAtCompileTime,
    SizeAtCompileTime = Base::SizeAtCompileTime
  };

  typedef typename internal::traits<Derived>::StorageKind StorageKind;
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type PacketScalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef const Scalar *PointerType;

  using Base::derived;
  using Base::Flags;
  using Base::IsRowMajor;
  using Base::IsVectorAtCompileTime;
  using Base::MaxColsAtCompileTime;
  using Base::MaxRowsAtCompileTime;
  using Base::MaxSizeAtCompileTime;

  using Base::coeff;
  using Base::coeffRef;
  using Base::cols;
  using Base::eval;
  using Base::lazyAssign;
  using Base::rows;
  using Base::size;

  using Base::colStride;
  using Base::innerStride;
  using Base::outerStride;
  using Base::rowStride;

  // bug 217 - compile error on ICC 11.1
  using Base::operator=;

  typedef typename Base::CoeffReturnType CoeffReturnType;

  /** \copydoc DenseBase::rows() */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index rows() const EIGEN_NOEXCEPT {
    return m_rows.value();
  }
  /** \copydoc DenseBase::cols() */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index cols() const EIGEN_NOEXCEPT {
    return m_cols.value();
  }

  /** Returns a pointer to the first coefficient of the matrix or vector.
   *
   * \note When addressing this data, make sure to honor the strides returned by
   * innerStride() and outerStride().
   *
   * \sa innerStride(), outerStride()
   */
  EIGEN_DEVICE_FUNC inline const Scalar *data() const { return m_data; }

  /** \copydoc PlainObjectBase::coeff(Index,Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeff(Index rowId, Index colId) const {
    return m_data[colId * colStride() + rowId * rowStride()];
  }

  /** \copydoc PlainObjectBase::coeff(Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeff(Index index) const {
    return m_data[index * innerStride()];
  }

  /** \copydoc PlainObjectBase::coeffRef(Index,Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeffRef(Index rowId, Index colId) const {
    return this->m_data[colId * colStride() + rowId * rowStride()];
  }

  /** \copydoc PlainObjectBase::coeffRef(Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeffRef(Index index) const {
    return this->m_data[index * innerStride()];
  }

  /** \internal */
  template <int LoadMode>
  inline PacketScalar packet(Index rowId, Index colId) const {
    return internal::ploadt<PacketScalar, LoadMode>(
        m_data + (colId * colStride() + rowId * rowStride()));
  }

  /** \internal */
  template <int LoadMode>
  inline PacketScalar packet(Index index) const {
    return internal::ploadt<PacketScalar, LoadMode>(m_data +
                                                    index * innerStride());
  }

  /** \internal Constructor for fixed size matrices or vectors */
  EIGEN_DEVICE_FUNC
  explicit inline MapBase(PointerType dataPtr)
      : m_data(dataPtr), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
    checkSanity<Derived>();
  }

  /** \internal Constructor for dynamically sized vectors */
  EIGEN_DEVICE_FUNC
  inline MapBase(PointerType dataPtr, Index vecSize)
      : m_data(dataPtr),
        m_rows(RowsAtCompileTime == Dynamic ? vecSize
                                            : Index(RowsAtCompileTime)),
        m_cols(ColsAtCompileTime == Dynamic ? vecSize
                                            : Index(ColsAtCompileTime)) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(vecSize >= 0);
    eigen_assert(dataPtr == 0 || SizeAtCompileTime == Dynamic ||
                 SizeAtCompileTime == vecSize);
    checkSanity<Derived>();
  }

  /** \internal Constructor for dynamically sized matrices */
  EIGEN_DEVICE_FUNC
  inline MapBase(PointerType dataPtr, Index rows, Index cols)
      : m_data(dataPtr), m_rows(rows), m_cols(cols) {
    eigen_assert((dataPtr == 0) ||
                 (rows >= 0 &&
                  (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) &&
                  cols >= 0 &&
                  (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)));
    checkSanity<Derived>();
  }

#ifdef EIGEN_MAPBASE_PLUGIN
#include EIGEN_MAPBASE_PLUGIN
#endif

 protected:
  EIGEN_DEFAULT_COPY_CONSTRUCTOR(MapBase)
  EIGEN_DEFAULT_EMPTY_CONSTRUCTOR_AND_DESTRUCTOR(MapBase)

  template <typename T>
  EIGEN_DEVICE_FUNC void checkSanity(
      typename internal::enable_if<(internal::traits<T>::Alignment > 0),
                                   void *>::type = 0) const {
#if EIGEN_MAX_ALIGN_BYTES > 0
    // innerStride() is not set yet when this function is called, so we
    // optimistically assume the lowest plausible value:
    const Index minInnerStride = InnerStrideAtCompileTime == Dynamic
                                     ? 1
                                     : Index(InnerStrideAtCompileTime);
    EIGEN_ONLY_USED_FOR_DEBUG(minInnerStride);
    eigen_assert((((internal::UIntPtr(m_data) %
                    internal::traits<Derived>::Alignment) == 0) ||
                  (cols() * rows() * minInnerStride * sizeof(Scalar)) <
                      internal::traits<Derived>::Alignment) &&
                 "data is not aligned");
#endif
  }

  template <typename T>
  EIGEN_DEVICE_FUNC void checkSanity(
      typename internal::enable_if<internal::traits<T>::Alignment == 0,
                                   void *>::type = 0) const {}

  PointerType m_data;
  const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_rows;
  const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_cols;
};

template <typename MegBA_t, int... MatrixArgs, int MapOptions,
          typename StrideType, int BlockRows, int BlockCols, bool InnerPanel>
class MapBase<
    Block<const Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                    MapOptions, StrideType>,
          BlockRows, BlockCols, InnerPanel>,
    ReadOnlyAccessors>
    : public internal::dense_xpr_base<Block<
          const Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                    MapOptions, StrideType>,
          BlockRows, BlockCols, InnerPanel>>::type {
  using Derived =
      Block<const Map<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                      MapOptions, StrideType>,
            BlockRows, BlockCols, InnerPanel>;

 public:
  typedef typename internal::dense_xpr_base<Derived>::type Base;
  enum {
    RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
    ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
    InnerStrideAtCompileTime =
        internal::traits<Derived>::InnerStrideAtCompileTime,
    SizeAtCompileTime = Base::SizeAtCompileTime
  };

  typedef typename internal::traits<Derived>::StorageKind StorageKind;
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type PacketScalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef const Scalar *PointerType;

  using Base::derived;
  using Base::Flags;
  using Base::IsRowMajor;
  using Base::IsVectorAtCompileTime;
  using Base::MaxColsAtCompileTime;
  using Base::MaxRowsAtCompileTime;
  using Base::MaxSizeAtCompileTime;

  using Base::coeff;
  using Base::coeffRef;
  using Base::cols;
  using Base::eval;
  using Base::lazyAssign;
  using Base::rows;
  using Base::size;

  using Base::colStride;
  using Base::innerStride;
  using Base::outerStride;
  using Base::rowStride;

  // bug 217 - compile error on ICC 11.1
  using Base::operator=;

  typedef typename Base::CoeffReturnType CoeffReturnType;

  /** \copydoc DenseBase::rows() */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index rows() const EIGEN_NOEXCEPT {
    return m_rows.value();
  }
  /** \copydoc DenseBase::cols() */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index cols() const EIGEN_NOEXCEPT {
    return m_cols.value();
  }

  /** Returns a pointer to the first coefficient of the matrix or vector.
   *
   * \note When addressing this data, make sure to honor the strides returned by
   * innerStride() and outerStride().
   *
   * \sa innerStride(), outerStride()
   */
  EIGEN_DEVICE_FUNC inline const Scalar *data() const { return m_data; }

  /** \copydoc PlainObjectBase::coeff(Index,Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeff(Index rowId, Index colId) const {
    return m_data[colId * colStride() + rowId * rowStride()];
  }

  /** \copydoc PlainObjectBase::coeff(Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeff(Index index) const {
    return m_data[index * innerStride()];
  }

  /** \copydoc PlainObjectBase::coeffRef(Index,Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeffRef(Index rowId, Index colId) const {
    return this->m_data[colId * colStride() + rowId * rowStride()];
  }

  /** \copydoc PlainObjectBase::coeffRef(Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeffRef(Index index) const {
    return this->m_data[index * innerStride()];
  }

  /** \internal */
  template <int LoadMode>
  inline PacketScalar packet(Index rowId, Index colId) const {
    return internal::ploadt<PacketScalar, LoadMode>(
        m_data + (colId * colStride() + rowId * rowStride()));
  }

  /** \internal */
  template <int LoadMode>
  inline PacketScalar packet(Index index) const {
    return internal::ploadt<PacketScalar, LoadMode>(m_data +
                                                    index * innerStride());
  }

  /** \internal Constructor for fixed size matrices or vectors */
  EIGEN_DEVICE_FUNC
  explicit inline MapBase(PointerType dataPtr)
      : m_data(dataPtr), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
    checkSanity<Derived>();
  }

  /** \internal Constructor for dynamically sized vectors */
  EIGEN_DEVICE_FUNC
  inline MapBase(PointerType dataPtr, Index vecSize)
      : m_data(dataPtr),
        m_rows(RowsAtCompileTime == Dynamic ? vecSize
                                            : Index(RowsAtCompileTime)),
        m_cols(ColsAtCompileTime == Dynamic ? vecSize
                                            : Index(ColsAtCompileTime)) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(vecSize >= 0);
    eigen_assert(dataPtr == 0 || SizeAtCompileTime == Dynamic ||
                 SizeAtCompileTime == vecSize);
    checkSanity<Derived>();
  }

  /** \internal Constructor for dynamically sized matrices */
  EIGEN_DEVICE_FUNC
  inline MapBase(PointerType dataPtr, Index rows, Index cols)
      : m_data(dataPtr), m_rows(rows), m_cols(cols) {
    eigen_assert((dataPtr == 0) ||
                 (rows >= 0 &&
                  (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) &&
                  cols >= 0 &&
                  (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)));
    checkSanity<Derived>();
  }

#ifdef EIGEN_MAPBASE_PLUGIN
#include EIGEN_MAPBASE_PLUGIN
#endif

 protected:
  EIGEN_DEFAULT_COPY_CONSTRUCTOR(MapBase)
  EIGEN_DEFAULT_EMPTY_CONSTRUCTOR_AND_DESTRUCTOR(MapBase)

  template <typename T>
  EIGEN_DEVICE_FUNC void checkSanity(
      typename internal::enable_if<(internal::traits<T>::Alignment > 0),
                                   void *>::type = 0) const {
#if EIGEN_MAX_ALIGN_BYTES > 0
    // innerStride() is not set yet when this function is called, so we
    // optimistically assume the lowest plausible value:
    const Index minInnerStride = InnerStrideAtCompileTime == Dynamic
                                     ? 1
                                     : Index(InnerStrideAtCompileTime);
    EIGEN_ONLY_USED_FOR_DEBUG(minInnerStride);
    eigen_assert((((internal::UIntPtr(m_data) %
                    internal::traits<Derived>::Alignment) == 0) ||
                  (cols() * rows() * minInnerStride * sizeof(Scalar)) <
                      internal::traits<Derived>::Alignment) &&
                 "data is not aligned");
#endif
  }

  template <typename T>
  EIGEN_DEVICE_FUNC void checkSanity(
      typename internal::enable_if<internal::traits<T>::Alignment == 0,
                                   void *>::type = 0) const {}

  PointerType m_data;
  const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_rows;
  const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_cols;
};

template <typename MegBA_t, int... MatrixArgs, int BlockRows, int BlockCols,
          bool InnerPanel>
class MapBase<Block<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                    BlockRows, BlockCols, InnerPanel>,
              ReadOnlyAccessors>
    : public internal::dense_xpr_base<
          Block<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                BlockRows, BlockCols, InnerPanel>>::type {
  using Derived = Block<const Matrix<MegBA::JetVector<MegBA_t>, MatrixArgs...>,
                        BlockRows, BlockCols, InnerPanel>;

 public:
  typedef typename internal::dense_xpr_base<Derived>::type Base;
  enum {
    RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
    ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
    InnerStrideAtCompileTime =
        internal::traits<Derived>::InnerStrideAtCompileTime,
    SizeAtCompileTime = Base::SizeAtCompileTime
  };

  typedef typename internal::traits<Derived>::StorageKind StorageKind;
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type PacketScalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef const Scalar *PointerType;

  using Base::derived;
  using Base::Flags;
  using Base::IsRowMajor;
  using Base::IsVectorAtCompileTime;
  using Base::MaxColsAtCompileTime;
  using Base::MaxRowsAtCompileTime;
  using Base::MaxSizeAtCompileTime;

  using Base::coeff;
  using Base::coeffRef;
  using Base::cols;
  using Base::eval;
  using Base::lazyAssign;
  using Base::rows;
  using Base::size;

  using Base::colStride;
  using Base::innerStride;
  using Base::outerStride;
  using Base::rowStride;

  // bug 217 - compile error on ICC 11.1
  using Base::operator=;

  typedef typename Base::CoeffReturnType CoeffReturnType;

  /** \copydoc DenseBase::rows() */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index rows() const EIGEN_NOEXCEPT {
    return m_rows.value();
  }
  /** \copydoc DenseBase::cols() */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index cols() const EIGEN_NOEXCEPT {
    return m_cols.value();
  }

  /** Returns a pointer to the first coefficient of the matrix or vector.
   *
   * \note When addressing this data, make sure to honor the strides returned by
   * innerStride() and outerStride().
   *
   * \sa innerStride(), outerStride()
   */
  EIGEN_DEVICE_FUNC inline const Scalar *data() const { return m_data; }

  /** \copydoc PlainObjectBase::coeff(Index,Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeff(Index rowId, Index colId) const {
    return m_data[colId * colStride() + rowId * rowStride()];
  }

  /** \copydoc PlainObjectBase::coeff(Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeff(Index index) const {
    return m_data[index * innerStride()];
  }

  /** \copydoc PlainObjectBase::coeffRef(Index,Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeffRef(Index rowId, Index colId) const {
    return this->m_data[colId * colStride() + rowId * rowStride()];
  }

  /** \copydoc PlainObjectBase::coeffRef(Index) const */
  EIGEN_DEVICE_FUNC
  inline const Scalar &coeffRef(Index index) const {
    return this->m_data[index * innerStride()];
  }

  /** \internal */
  template <int LoadMode>
  inline PacketScalar packet(Index rowId, Index colId) const {
    return internal::ploadt<PacketScalar, LoadMode>(
        m_data + (colId * colStride() + rowId * rowStride()));
  }

  /** \internal */
  template <int LoadMode>
  inline PacketScalar packet(Index index) const {
    return internal::ploadt<PacketScalar, LoadMode>(m_data +
                                                    index * innerStride());
  }

  /** \internal Constructor for fixed size matrices or vectors */
  EIGEN_DEVICE_FUNC
  explicit inline MapBase(PointerType dataPtr)
      : m_data(dataPtr), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
    checkSanity<Derived>();
  }

  /** \internal Constructor for dynamically sized vectors */
  EIGEN_DEVICE_FUNC
  inline MapBase(PointerType dataPtr, Index vecSize)
      : m_data(dataPtr),
        m_rows(RowsAtCompileTime == Dynamic ? vecSize
                                            : Index(RowsAtCompileTime)),
        m_cols(ColsAtCompileTime == Dynamic ? vecSize
                                            : Index(ColsAtCompileTime)) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(vecSize >= 0);
    eigen_assert(dataPtr == 0 || SizeAtCompileTime == Dynamic ||
                 SizeAtCompileTime == vecSize);
    checkSanity<Derived>();
  }

  /** \internal Constructor for dynamically sized matrices */
  EIGEN_DEVICE_FUNC
  inline MapBase(PointerType dataPtr, Index rows, Index cols)
      : m_data(dataPtr), m_rows(rows), m_cols(cols) {
    eigen_assert((dataPtr == 0) ||
                 (rows >= 0 &&
                  (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) &&
                  cols >= 0 &&
                  (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)));
    checkSanity<Derived>();
  }

#ifdef EIGEN_MAPBASE_PLUGIN
#include EIGEN_MAPBASE_PLUGIN
#endif

 protected:
  EIGEN_DEFAULT_COPY_CONSTRUCTOR(MapBase)
  EIGEN_DEFAULT_EMPTY_CONSTRUCTOR_AND_DESTRUCTOR(MapBase)

  template <typename T>
  EIGEN_DEVICE_FUNC void checkSanity(
      typename internal::enable_if<(internal::traits<T>::Alignment > 0),
                                   void *>::type = 0) const {
#if EIGEN_MAX_ALIGN_BYTES > 0
    // innerStride() is not set yet when this function is called, so we
    // optimistically assume the lowest plausible value:
    const Index minInnerStride = InnerStrideAtCompileTime == Dynamic
                                     ? 1
                                     : Index(InnerStrideAtCompileTime);
    EIGEN_ONLY_USED_FOR_DEBUG(minInnerStride);
    eigen_assert((((internal::UIntPtr(m_data) %
                    internal::traits<Derived>::Alignment) == 0) ||
                  (cols() * rows() * minInnerStride * sizeof(Scalar)) <
                      internal::traits<Derived>::Alignment) &&
                 "data is not aligned");
#endif
  }

  template <typename T>
  EIGEN_DEVICE_FUNC void checkSanity(
      typename internal::enable_if<internal::traits<T>::Alignment == 0,
                                   void *>::type = 0) const {}

  PointerType m_data;
  const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_rows;
  const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_cols;
};
}  // namespace Eigen
