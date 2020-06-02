/**
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <Python.h>
#include "arrayobject.h"
#include "common.h"
#include <omp.h>
#include <float.h>
#include <vector>
#include <algorithm>

using namespace std;

//int is int32 in python 2 only
#if PY_MAJOR_VERSION >= 3
  typedef signed int int_t;
  #define p_int32 (const signed int*) p_int32
#else
  typedef signed int int_t;
#endif

static PyObject *pySetOMPNumThreads(PyObject */*self*/, PyObject *args)
{
  int_t nthr;
  if (!PyArg_ParseTuple(args, "i", &nthr)) return NULL;
  omp_set_num_threads(nthr);
  Py_RETURN_NONE;
}

static PyObject *pySetOMPDynamicNumThreads(PyObject */*self*/, PyObject *args)
{
  int_t nthr;
  if (!PyArg_ParseTuple(args, "i", &nthr)) return NULL;
  omp_set_dynamic(nthr);
  Py_RETURN_NONE;
}

static PyObject *pyWhereMaxMask(PyObject */*self*/, PyObject *args)
{
  PyArrayObject *A, *Blocks, *Mask;
  PyObject *ObjAns;
  int_t doabs;
  if (!PyArg_ParseTuple(args, "O!O!O!Oi",
      &PyArray_Type,  &A,
      &PyArray_Type,  &Mask,
      &PyArray_Type,  &Blocks,
      &ObjAns,
      &doabs
      )) return NULL;

  PyArrayObject *Ans = (PyArrayObject *) PyArray_ContiguousFromObject(ObjAns, NPY_FLOAT32, 0, 4);

  int_t NX=int_t(PyArray_DIMS(A)[0]), NY=int_t(PyArray_DIMS(A)[1]);

  const bool *pMask=p_bool(Mask);
  const int_t *pBlocks=p_int32(Blocks);
  int_t nblocks=int(PyArray_DIMS(Blocks)[0]);
  vector<float> MaxBlock(nblocks-1);
  vector<int_t> xMaxBlock(nblocks-1), yMaxBlock(nblocks-1);

#pragma omp parallel for
  for (int_t iblock=0; iblock<nblocks-1; iblock++)
    {
    const float *a = p_float32(A);
    float ThisMax=-FLT_MAX;
    int_t ThisxMax=0, ThisyMax=0;
    for (int_t i_a=pBlocks[iblock]; i_a<min(pBlocks[iblock+1], NX); ++i_a)
      for (int_t j_a=0; j_a<NY; ++j_a)
        {
        int_t ii=i_a*NY+j_a;
        if (!pMask[ii])
          {
          float ThisVal = doabs ? abs<float>(a[ii]) : a[ii];
          if (ThisVal>ThisMax)
            { ThisxMax = i_a; ThisyMax = j_a; ThisMax = ThisVal; }
          }
        }

    MaxBlock[iblock]=ThisMax;
    xMaxBlock[iblock]=ThisxMax;
    yMaxBlock[iblock]=ThisyMax;
    }

  float Max=0;
  int_t xMax=0, yMax=0;
  for (long iblock=0; iblock<nblocks-1; iblock++)
    if (MaxBlock[iblock]>Max)
      {
      Max=MaxBlock[iblock];
      xMax=xMaxBlock[iblock];
      yMax=yMaxBlock[iblock];
      }

  float *ans = p_float32(Ans);
  ans[0]=(float)xMax;
  ans[1]=(float)yMax;
  ans[2]=(float)Max;
  return PyArray_Return(Ans);
  }

static PyObject *pyWhereMax(PyObject */*self*/, PyObject *args)
{
  PyArrayObject *A, *Blocks;
  PyObject *ObjAns;
  int_t doabs;

  if (!PyArg_ParseTuple(args, "O!O!Oi",
      &PyArray_Type,  &A,
      &PyArray_Type,  &Blocks,
      &ObjAns,
      &doabs
      )) return NULL;

  PyArrayObject *Ans = (PyArrayObject *) PyArray_ContiguousFromObject(ObjAns, NPY_FLOAT32, 0, 4);
  int_t NX=int(PyArray_DIMS(A)[0]), NY=int(PyArray_DIMS(A)[1]);

  const int_t *pBlocks=p_int32(Blocks);
  int_t nblocks=int(PyArray_DIMS(Blocks)[0]);
  vector<float> MaxBlock(nblocks-1);
  vector<int> xMaxBlock(nblocks-1), yMaxBlock(nblocks-1);

#pragma omp parallel for
  for (long iblock = 0; iblock < nblocks-1; iblock++){
    const float *a = p_float32(A);
    float ThisMax=-FLT_MAX;
    int_t ThisxMax=0, ThisyMax=0;
    for (int_t i_a=pBlocks[iblock]; i_a<min(pBlocks[iblock+1], NX); ++i_a)
      for (int_t j_a=0; j_a<NY; ++j_a)
        {
        int_t ii=i_a*NY+j_a;
        float ThisVal = doabs ? abs<float>(a[ii]) : a[ii];
        if (ThisVal > ThisMax)
          { ThisMax=ThisVal; ThisxMax=i_a; ThisyMax=j_a; }
        }

    MaxBlock[iblock]=ThisMax;
    xMaxBlock[iblock]=ThisxMax;
    yMaxBlock[iblock]=ThisyMax;
    }

  float Max=0;
  int_t xMax=0, yMax=0;
  for (long iblock = 0; iblock < nblocks-1; iblock++){
    if(MaxBlock[iblock]>Max){
      Max=MaxBlock[iblock];
      xMax=xMaxBlock[iblock];
      yMax=yMaxBlock[iblock];
    }
  }

  float *ans = p_float32(Ans);
  ans[0]=(float)xMax;
  ans[1]=(float)yMax;
  ans[2]=(float)Max;
  return PyArray_Return(Ans);
}

struct addeq
  {
  void operator()(float &a, float b) const
    { a+=b; }
  };
struct muleq
  {
  void operator()(float &a, float b) const
    { a*=b; }
  };
struct diveq
  {
  void operator()(float &a, float b) const
    { a/=b; }
  };

template<typename Op>static PyObject *pyOpArray(PyObject *args, Op op)
{
  PyObject *ObjA;
  PyArrayObject *A, *B, *Aedge, *Bedge, *Blocks;
  float factor;

  if (!PyArg_ParseTuple(args, "OO!O!O!fO!",
      &ObjA,
      &PyArray_Type,  &Aedge,
      &PyArray_Type,  &B,
      &PyArray_Type,  &Bedge,
      &factor,
      &PyArray_Type,  &Blocks
      )) return NULL;

  A = (PyArrayObject *)PyArray_ContiguousFromObject(ObjA, NPY_FLOAT32, 0, 4);

  int_t NYa=int(PyArray_DIMS(A)[1]);
  int_t NYb=int(PyArray_DIMS(B)[1]);

  const int_t *aedge = p_int32(Aedge);
  int_t a_x0=aedge[0], a_x1=aedge[1], a_y0=aedge[2], a_y1=aedge[3];

  const int_t *bedge = p_int32(Bedge);
  int_t b_x0=bedge[0], b_y0=bedge[2];

  const int_t *pBlocks=p_int32(Blocks);
  int_t nblocks=int(PyArray_DIMS(Blocks)[0]);
#pragma omp parallel for
  for (long iblock=0; iblock<nblocks-1; iblock++){
   float *a = p_float32(A);
    const float *b = p_float32(B);

    for (int_t i_a=pBlocks[iblock]; i_a<min(pBlocks[iblock+1], a_x1); i_a++)
      {
      int_t i_b = b_x0+i_a-a_x0;
      for (int_t j_a=a_y0, j_b=b_y0; j_a<a_y1; ++j_a, ++j_b)
        op(a[i_a*NYa+j_a],b[i_b*NYb+j_b]*factor);
      }
    }
  return PyArray_Return(A);
}

static PyObject *pyAddArray(PyObject */*self*/, PyObject *args)
{ return pyOpArray(args, addeq()); }

static PyObject *pyProdArray(PyObject */*self*/, PyObject *args)
{ return pyOpArray(args, muleq()); }

static PyObject *pyDivArray(PyObject */*self*/, PyObject *args)
{ return pyOpArray(args, diveq()); }

static PyMethodDef _pyArrays_testMethods[] = {
  {"pyAddArray", pyAddArray, METH_VARARGS, 0},
  {"pyProdArray", pyProdArray, METH_VARARGS, 0},
  {"pyDivArray", pyDivArray, METH_VARARGS, 0},
  {"pyWhereMax", pyWhereMax, METH_VARARGS, 0},
  {"pyWhereMaxMask", pyWhereMaxMask, METH_VARARGS, 0},
  {"pySetOMPNumThreads", pySetOMPNumThreads, METH_VARARGS, 0},
  {"pySetOMPDynamicNumThreads", pySetOMPDynamicNumThreads, METH_VARARGS, 0},
  {NULL, NULL, 0, 0}     /* Sentinel - marks the end of this structure */
};

extern "C" {

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef _mod = {
    PyModuleDef_HEAD_INIT,
    "_pyGridder3x",
    "_pyGridder3x",
    -1,  
    _pyArrays_testMethods,
    NULL,
    NULL,
    NULL,
    NULL
  };
  PyMODINIT_FUNC PyInit__pyArrays3x(void) {
    PyObject * m = PyModule_Create(&_mod);
    import_array();
    return m;
  }
#else
  void init_pyArrays27()
  {
    Py_InitModule("_pyArrays27", _pyArrays_testMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
  }
#endif
}
