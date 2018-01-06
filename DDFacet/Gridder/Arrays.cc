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

static PyObject *pySetOMPNumThreads(PyObject */*self*/, PyObject *args)
{
  int nthr;
  if (!PyArg_ParseTuple(args, "i", &nthr)) return NULL;
  omp_set_num_threads(nthr);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *pySetOMPDynamicNumThreads(PyObject */*self*/, PyObject *args)
{
  int nthr;
  if (!PyArg_ParseTuple(args, "i", &nthr)) return NULL;
  omp_set_dynamic(nthr);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *pyWhereMaxMask(PyObject */*self*/, PyObject *args)
{
  PyArrayObject *A, *Blocks, *Mask;
  PyObject *ObjAns;
  int doabs;
  if (!PyArg_ParseTuple(args, "O!O!O!Oi",
      &PyArray_Type,  &A,
      &PyArray_Type,  &Mask,
      &PyArray_Type,  &Blocks,
      &ObjAns,
      &doabs
      ))  return NULL;

  PyArrayObject *Ans = (PyArrayObject *) PyArray_ContiguousFromObject(ObjAns, PyArray_FLOAT32, 0, 4);

  int NX=(int)A->dimensions[0];
  int NY=(int)A->dimensions[1];

  const bool *pMask=p_bool(Mask);
  const int *pBlocks=p_int32(Blocks);
  int nblocks=(int)Blocks->dimensions[0];
  vector<float> MaxBlock(nblocks-1);
  vector<int> xMaxBlock(nblocks-1), yMaxBlock(nblocks-1);

#pragma omp parallel for
  for (int iblock = 0; iblock < nblocks-1; iblock++){
    int i0=pBlocks[iblock];
    int i1=min(pBlocks[iblock+1], NX);

    const float *a = p_float32(A);
    float ThisMax=-FLT_MAX;
    int ThisxMax=0;
    int ThisyMax=0;
    for (int i_a = i0; i_a < i1; i_a++)
      for (int j_a = 0; j_a < NY; j_a++)
        {
        int ii=i_a*NY+j_a;
        float ThisVal = pMask[ii] ? 0.f : a[ii];
        if(doabs) ThisVal = fabsf(ThisVal);

        ThisxMax= ((ThisVal > ThisMax) ? i_a : ThisxMax);
        ThisyMax= ((ThisVal > ThisMax) ? j_a : ThisyMax);
        ThisMax = ((ThisVal > ThisMax) ? ThisVal : ThisMax);
        }

      MaxBlock[iblock]=ThisMax;
      xMaxBlock[iblock]=ThisxMax;
      yMaxBlock[iblock]=ThisyMax;
    }

  float Max=0;
  int xMax=0;
  int yMax=0;
  for (long iblock = 0; iblock < nblocks-1; iblock++)
    if(MaxBlock[iblock]>Max){
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
  int doabs;

  if (!PyArg_ParseTuple(args, "O!O!Oi",
      &PyArray_Type,  &A,
      &PyArray_Type,  &Blocks,
      &ObjAns,
      &doabs
      ))  return NULL;

  PyArrayObject *Ans = (PyArrayObject *) PyArray_ContiguousFromObject(ObjAns, PyArray_FLOAT32, 0, 4);
  int NX=(int)A->dimensions[0];
  int NY=(int)A->dimensions[1];

  const int *pBlocks=p_int32(Blocks);
  int nblocks=(int)Blocks->dimensions[0];
  vector<float> MaxBlock(nblocks-1);
  vector<int> xMaxBlock(nblocks-1), yMaxBlock(nblocks-1);

#pragma omp parallel for
  for (long iblock = 0; iblock < nblocks-1; iblock++){
    int i0=pBlocks[iblock];
    int i1=min(pBlocks[iblock+1], NX);

    const float *a = p_float32(A);
    float ThisMax=-FLT_MAX;
    int ThisxMax=0;
    int ThisyMax=0;
    for (int i_a = i0; i_a < i1; i_a++)
      {
      for (int j_a = 0; j_a < NY; j_a++)
        {
        int ii=i_a*NY+j_a;
        float ThisVal=a[ii];
        if(doabs==1){
          ThisVal=fabsf(ThisVal);
        }
        if (ThisVal > ThisMax){
          ThisMax=ThisVal;
          ThisxMax=i_a;
          ThisyMax=j_a;
          }
        }
      }

    MaxBlock[iblock]=ThisMax;
    xMaxBlock[iblock]=ThisxMax;
    yMaxBlock[iblock]=ThisyMax;
    }

  float Max=0;
  int xMax=0;
  int yMax=0;
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

  A = (PyArrayObject *)PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);

  int NYa=int(A->dimensions[1]);
  int NYb=int(B->dimensions[1]);

  const int *aedge = p_int32(Aedge);
  int a_x0=aedge[0];
  int a_x1=aedge[1];
  int a_y0=aedge[2];
  int a_y1=aedge[3];

  const int *bedge = p_int32(Bedge);
  int b_x0=bedge[0];
  int b_y0=bedge[2];

  const int *pBlocks=p_int32(Blocks);
  int nblocks=(int)Blocks->dimensions[0];
#pragma omp parallel for
  for (long iblock = 0; iblock < nblocks-1; iblock++){
    int i0=pBlocks[iblock];
    int i1=min(pBlocks[iblock+1], a_x1);

    float *a = p_float32(A);
    const float *b = p_float32(B);

    for (int i_a=i0; i_a<i1; i_a++)
      {
      int di=i_a-a_x0;
      int i_b=b_x0+di;
      for (int j_a=a_y0; j_a<a_y1; j_a++)
        {
        int dj=j_a-a_y0;
        int j_b=b_y0+dj;
        op(a[i_a*NYa+j_a],b[i_b*NYb+j_b]*factor);
        }
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

/* ==== Set up the methods table ====================== */
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

void init_pyArrays() {
  (void) Py_InitModule("_pyArrays", _pyArrays_testMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}

}
