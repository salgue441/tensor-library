#pragma once

#include <tf/core/types.hpp>
#include <tf/core/error.hpp>
#include <tf/core/config.hpp>

#include <cstddef>
#include <complex>

namespace tf
{
  namespace math
  {
    /**
     * @enum BlasOperation
     * @brief Enumeration for the BLAS operation type.
     */
    enum class BlasOperation
    {
      NoTrans = 0,
      Trans = 1,
      ConjTrans = 2
    };

    /**
     * @class Blas
     * @brief BLAS operations wrapper with optimized implementations
     *
     * @version 1.0.0
     */
    class Blas
    {
    public:
      /**
       * @brief Compute vector dot product
       *
       * @tparam T Data type
       * @param n Number of elements in the vectors
       * @param x Vector x with stride incx
       * @param incx Stride of vector x
       * @param y Vector y with stride incy
       * @param incy Stride of vector y
       * @return T Dot product of the vectors
       */
      template <typename T>
      static T dot(size_t n, const T *x, int incx, const T *y, int incy);

      /**
       * @brief Compute vector norm
       *
       * @param T Data type
       * @param n Number of elements in the vector
       * @param x Vector x with stride incx
       * @param incx Stride of vector x
       * @return T Norm of the vector
       */
      template <typename T>
      static T nrm2(size_t n, const T *x, int incx);

      /**
       * @brief Scale vector by a constant
       *
       * @tparam T Data type
       * @param n Number of elements in the vector
       * @param alpha Scaling factor
       * @param x Vector x with stride incx
       * @param incx Stride of vector x
       */
      template <typename T>
      static void scal(size_t n, T alpha, T *x, int incx);

      /**
       * @brief Vector-vector addition
       *
       * y = alpha * x + y
       *
       * @tparam T Data type
       * @param n Number of elements in the vectors
       * @param alpha Scaling factor for vector x
       * @param x Vector x with stride incx
       * @param incx Stride of vector x
       * @param y Vector y with stride incy
       * @param incy Stride of vector y
       */
      template <typename T>
      static void axpy(size_t n, T alpha, const T *x, int incx, T *y, int incy);

      /**
       * @brief Matrix-vector multiplication
       *
       * y = alpha * op(A) * x + beta * y
       *
       * @tparam T Data type
       * @param m Number of rows in A
       * @param n Number of columns in A
       * @param A Matrix A with leading dimension lda
       * @param lda Leading dimension of A
       * @param x Vector x with stride incx
       * @param incx Stride of vector x
       * @param beta Scaling factor for vector y
       * @param y Vector y with stride incy
       * @param incy Stride of vector y
       */
      template <typename T>
      static void gemv(size_t m, size_t n, T alpha, const T *A, size_t lda,
                       const T *x, int incx, T beta, T *y, int incy);

      /**
       * @brief Symmetric matrix-vector multiplication
       *
       * y = alpha * A * x + beta * y
       *
       * @tparam T Data type
       * @param uplo Upper or lower triangular matrix
       * @param A Symmetric matrix A with leading dimension lda
       * @param lda Leading dimension of A
       * @param x Vector x with stride incx
       * @param incx Stride of vector x
       * @param beta Scaling factor for vector y
       * @param y Vector y with stride incy
       * @param incy Stride of vector y
       */
      template <typename T>
      static void symv(char uplo, size_t n, T alpha, const T *A, size_t lda,
                       const T *x, int incx, T beta, T *y, int incy);

      /**
       * @brief Matrix-matrix multiplication
       *
       * C = alpha * op(A) * op(B) + beta * C
       *
       * @tparam T Data type
       * @param transa Transpose operation for A
       * @param transb Transpose operation for B
       * @param m Number of rows in C
       * @param n Number of columns in C
       * @param k Number of columns in A and rows in B
       * @param alpha Scaling factor for A and B
       * @param A Matrix A with leading dimension lda
       * @param lda Leading dimension of A
       * @param B Matrix B with leading dimension ldb
       * @param ldb Leading dimension of B
       * @param beta Scaling factor for C
       * @param C Matrix C with leading dimension ldc
       * @param ldc Leading dimension of C
       */
      template <typename T>
      static void gemm(BlasOperation transa, BlasOperation transb,
                       size_t m, size_t n, size_t k, T alpha,
                       const T *A, size_t lda, const T *B, size_t ldb,
                       T beta, T *C, size_t ldc);

      /**
       * @brief Symmetric matrix-matrix multiplication
       *
       * C = alpha * A * B + beta * C
       *
       * @tparam T Data type
       * @param side Side of the matrix A
       * @param uplo Upper or lower triangular matrix
       * @param m Number of rows in C
       * @param n Number of columns in C
       * @param alpha Scaling factor for A and B
       * @param A Symmetric matrix A with leading dimension lda
       * @param lda Leading dimension of A
       * @param B Matrix B with leading dimension ldb
       * @param ldb Leading dimension of B
       * @param beta Scaling factor for C
       * @param C Matrix C with leading dimension ldc
       * @param ldc Leading dimension of C
       */
      template <typename T>
      static void symm(char side, char uplo, size_t m, size_t n, T alpha,
                       const T *A, size_t lda, const T *B, size_t ldb,
                       T beta, T *C, size_t ldc);

    private:
      /**
       * @brief Helper function for different data types
       *
       * @tparam T Data type
       * @param m Number of rows
       * @param n Number of columns
       * @param k Number of columns in A and rows in B
       * @param lda Leading dimension of A
       * @param ldb Leading dimension of B
       * @param ldc Leading dimension of C
       * @param transa Transpose operation for A
       * @param transb Transpose operation for B
       */
      template <typename T>
      static void check_gemm_dims(size_t m, size_t n, size_t k,
                                  size_t lda, size_t ldb, size_t ldc,
                                  BlasOperation transa, BlasOperation transb);
    };

  } // namespace math
} // namespace tf