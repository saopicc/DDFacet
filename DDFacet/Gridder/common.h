#ifndef GRIDDER_COMMON_H
#define GRIDDER_COMMON_H

#include <cmath>
#include <complex>
#include <Python.h>
#include <arrayobject.h>

static constexpr double C=299792458.;
static constexpr double PI=3.141592653589793238462643383279502884197;

using fcmplx = std::complex<float>;
using dcmplx = std::complex<double>;

template<typename T> static inline T *arrPtr(PyArrayObject *arrayin)
  { return reinterpret_cast<T *>(PyArray_DATA(arrayin)); }

static inline int *p_int32(PyArrayObject *arrayin)
  { return arrPtr<int>(arrayin); }

static inline uint16_t *p_uint16(PyArrayObject *arrayin)
  { return arrPtr<uint16_t>(arrayin); }

static inline long int *p_int64(PyArrayObject *arrayin)
  { return arrPtr<long int>(arrayin); }

static inline double *p_float64(PyArrayObject *arrayin)
  { return arrPtr<double>(arrayin); }

static inline float *p_float32(PyArrayObject *arrayin)
  { return arrPtr<float>(arrayin); }

static inline fcmplx *p_complex64(PyArrayObject *arrayin)
  { return arrPtr<fcmplx>(arrayin); }

static inline bool *p_bool(PyArrayObject *arrayin)
  { return arrPtr<bool>(arrayin); }

#endif
