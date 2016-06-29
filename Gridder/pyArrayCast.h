/**
Useful casting functions to go from PyArrayObject pointers
to array objects for common types
*/
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

/* Get pointer to arrayin int32 data*/
inline int32_t *p_int32(PyArrayObject *arrayin)  {
  return (int32_t *) arrayin->data;  
}

/* Get pointer to arrayin float64 data */
inline double *p_float64(PyArrayObject *arrayin)  {
  return (double *) arrayin->data; 
}

/* Get pointer to arrayin float32 data */
inline float *p_float32(PyArrayObject *arrayin)  {
  return (float *) arrayin->data;  
}

/* Get pointer to arrayin complex64 data*/
inline float complex *p_complex64(PyArrayObject *arrayin)  {
  return (float complex *) arrayin->data;  
}

/* Get pointer to arrayin complex128 data*/
inline double complex *p_complex128(PyArrayObject *arrayin)  {
  return (double complex *) arrayin->data;  
}

/* Get pointer to arrayin bool data*/
inline bool *p_bool(PyArrayObject *arrayin)  {
  return (bool *) arrayin->data;  
}

/* Get pointer to arrayin char data*/
char *p_char(PyArrayObject *arrayin)  {
  return (char *) arrayin->data;  
}

/* Get pointer to arrayin int16 data*/
short int *p_int16(PyArrayObject *arrayin)  {
  return (int16_t *) arrayin->data;  
}

/* Get pointer to arrayin int data*/
int *I_ptr(PyArrayObject *arrayin)  {
  return (int *) arrayin->data;
}