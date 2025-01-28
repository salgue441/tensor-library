#include <tf/math/blas.hpp>
#include <cmath>
#include <algorithm>

namespace tf
{
  namespace math
  {
    // Vector dot product
    template <>
    float Blas::dot<float>(size_t n, const float *x, int incx,
                           const float *y, int incy)
    {
      TF_CHECK(n > 0, std::invalid_argument, "Invalid vector size");

      float result = 0.0f;
      for (size_t i = 0; i < n; ++i)
        result += x[i * incx] * y[i * incy];

      return result;
    }

    template <>
    double Blas::dot<double>(size_t n, const double *x,
                             int incx, const double *y, int incy)
    {
      TF_CHECK(n > 0, std::invalid_argument, "Invalid vector size");

      double result = 0.0;
      for (size_t i = 0; i < n; ++i)
        result += x[i * incx] * y[i * incy];

      return result;
    }

    // Vector norm
    template <>
    float Blas::nrm2(size_t n, const float *x, int incx)
    {
      float scale = 0.0f;
      float ssq = 1.0f;

      for (size_t i = 0; i < n; ++i)
      {
        if (x[i * incx] != 0.0f)
        {
          float absxi = std::abs(x[i * incx]);
          if (scale < absxi)
          {
            float temp = scale / absxi;

            ssq = 1.0f + ssq * (temp * temp);
            scale = absxi;
          }
          else
          {
            float temp = absxi / scale;
            ssq += temp * temp;
          }
        }
      }

      return scale * std::sqrt(ssq);
    }

    template <>
    double Blas::nrm2<double>(size_t n, const double *x, int incx)
    {
      double scale = 0.0;
      double ssq = 1.0;

      for (size_t i = 0; i < n; ++i)
      {
        if (x[i * incx] != 0.0)
        {
          double absxi = std::abs(x[i * incx]);

          if (scale < absxi)
          {
            double temp = scale / absxi;
            ssq = 1.0 + ssq * (temp * temp);
            scale = absxi;
          }
          else
          {
            double temp = absxi / scale;
            ssq += temp * temp;
          }
        }
      }

      return scale * std::sqrt(ssq);
    }

    // Scale vector
    template <>
    void Blas::scal<float>(size_t n, float alpha, float *x, int incx)
    {
      for (size_t i = 0; i < n; ++i)
        x[i * incx] *= alpha;
    }

    template <>
    void Blas::scal<double>(size_t n, double alpha, double *x, int incx)
    {
      for (size_t i = 0; i < n; ++i)
        x[i * incx] *= alpha;
    }

    // Vector addition
    template <>
    void Blas::axpy<float>(size_t n, float alpha, const float *x, int incx, float *y, int incy)
    {
      for (size_t i = 0; i < n; ++i)
        y[i * incy] += alpha * x[i * incx];
    }

    template <>
    void Blas::axpy<double>(size_t n, double alpha, const double *x, int incx, double *y, int incy)
    {
      for (size_t i = 0; i < n; ++i)
        y[i * incy] += alpha * x[i * incx];
    }

    template <typename T>
    void Blas::check_gemm_dims(size_t m, size_t n, size_t k,
                               size_t lda, size_t ldb, size_t ldc,
                               BlasOperation transa, BlasOperation transb)
    {
      bool ta = transa != BlasOperation::NoTrans;
      bool tb = transb != BlasOperation::NoTrans;

      if (ta && lda < k)
        throw core::ShapeError("Invalid lda for transposed A");

      if (!ta && lda < m)
        throw core::ShapeError("Invalid lda for non-transposed A");

      if (tb && ldb < n)
        throw core::ShapeError("Invalid ldb for transposed B");

      if (!tb && ldb < k)
        throw core::ShapeError("Invalid ldb for non-transposed B");

      if (ldc < n)
        throw core::ShapeError("Invalid ldc");
    }
  } // namespace math
} // namespace tf