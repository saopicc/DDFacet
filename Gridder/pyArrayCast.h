/**
Useful casting functions to go from PyArrayObject pointers
to array objects for common types
*/


#include <stdbool.h>
#include <stdint.h>

inline int32_t *p_int32(PyArrayObject *arrayin)  {
  return (int32_t *) arrayin->data;  /* pointer to arrayin data as int32 */
}


inline double *p_float64(PyArrayObject *arrayin)  {
  return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

inline float *p_float32(PyArrayObject *arrayin)  {
  return (float *) arrayin->data;  /* pointer to arrayin data as float */
}


inline float complex *p_complex64(PyArrayObject *arrayin)  {
  return (float complex *) arrayin->data;  /* pointer to arrayin data as double */
}

inline double complex *p_complex128(PyArrayObject *arrayin)  {
  return (double complex *) arrayin->data;  /* pointer to arrayin data as double */
}

inline bool *p_bool(PyArrayObject *arrayin)  {
  return (bool *) arrayin->data;  /* pointer to arrayin data as double */
}