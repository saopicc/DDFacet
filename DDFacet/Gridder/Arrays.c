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
/* A file to test imorting C modules for handling arrays to Python */
#include <Python.h>
#include <math.h>
#include <time.h>
#include "arrayobject.h"
#include "Arrays.h"
#include "complex.h"
#include <omp.h>
#include <float.h>

clock_t start;

void initTime(){start=clock();}

void timeit(char* Name){
  clock_t diff;
  diff = clock() - start;
  start=clock();
  float msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("%s: %f\n",Name,msec);
}

double AppendTimeit(){
  clock_t diff;
  diff = clock() - start;
  double msec = diff * 1000000 / CLOCKS_PER_SEC;
  return msec;
}



/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef _pyArrays_testMethods[] = {
	{"pyAddArray", pyAddArray, METH_VARARGS},
	{"pyProdArray", pyProdArray, METH_VARARGS},
	{"pyDivArray", pyDivArray, METH_VARARGS},
	{"pyWhereMax", pyWhereMax, METH_VARARGS},
	{"pyWhereMaxMask", pyWhereMaxMask, METH_VARARGS},
	{"pySetOMPNumThreads", pySetOMPNumThreads, METH_VARARGS},
	{"pySetOMPDynamicNumThreads", pySetOMPDynamicNumThreads, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_pyArrays()  {
  (void) Py_InitModule("_pyArrays", _pyArrays_testMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *pySetOMPNumThreads(PyObject *self, PyObject *args)
{
  int nthr;

  if (!PyArg_ParseTuple(args, "i", &nthr))  return NULL;

  omp_set_num_threads(nthr);

  //  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *pySetOMPDynamicNumThreads(PyObject *self, PyObject *args)
{
  int nthr;

  if (!PyArg_ParseTuple(args, "i", &nthr))  return NULL;

  omp_set_dynamic(nthr);

  //  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *pyWhereMaxMask(PyObject *self, PyObject *args)
{
  PyArrayObject *A, *Blocks,*Ans,*Mask;
  PyObject *ObjAns;
  int doabs;

  if (!PyArg_ParseTuple(args, "O!O!O!Oi", 
			&PyArray_Type,  &A,
			&PyArray_Type,  &Mask,
			&PyArray_Type,  &Blocks,
			&ObjAns,
			&doabs
			))  return NULL;
  
  //  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  
  Ans = (PyArrayObject *) PyArray_ContiguousFromObject(ObjAns, PyArray_FLOAT32, 0, 4);
  
  int nx,ny,NX,NY,np;
  
  NX=A->dimensions[0];
  NY=A->dimensions[1];
  //printf("dims %i %i\n",NX,NY);
  

  bool* pMask=p_bool(Mask);

  long iblock;
  int* pBlocks=p_int32(Blocks);
  int nblocks=Blocks->dimensions[0];
  float *MaxBlock;
  int *xMaxBlock;
  int *yMaxBlock;
  MaxBlock=malloc((nblocks-1)*sizeof(float));
  xMaxBlock=malloc((nblocks-1)*sizeof(int));
  yMaxBlock=malloc((nblocks-1)*sizeof(int));

  {
#pragma omp parallel for private(iblock)
    for (iblock = 0; iblock < nblocks-1; iblock++){
      int i0=pBlocks[iblock];
      int i1=pBlocks[iblock+1];
      if(i1>=NX){i1=NX;};
      
      //printf("- block %i->%i\n",i0,i1);
      
      float* a = p_float32(A);
      int i_a;
      int j_a;
      float ThisMax=-FLT_MAX;
      int ThisxMax=0;
      int ThisyMax=0;
      int ThisIndex=0;
      float ThisVal;
      for (i_a = i0; i_a < i1; i_a++)
      	{
      	  for (j_a = 0; j_a < NY; j_a++)
      	    {
	      

      	      int ii=i_a*NY+j_a;
	      ThisVal=a[ii];
	      ThisVal=((pMask[ii]==1) ? 0 : ThisVal);

	      if(doabs){ThisVal = ((ThisVal > 0) ? ThisVal : -ThisVal);};


	      ThisxMax= ((ThisVal > ThisMax) ? i_a : ThisxMax);
	      ThisyMax= ((ThisVal > ThisMax) ? j_a : ThisyMax);
	      ThisMax = ((ThisVal > ThisMax) ? ThisVal : ThisMax);

      	      /* int ii=i_a*NY+j_a; */
	      /* if(pMask[ii]==1){ */
	      /* 	//printf("Skipping (%i,%i)\n",i_a,j_a); */
	      /* 	continue; */
	      /* }; */
	      /* ThisVal=a[ii]; */
	      /* if(doabs==1){ */
	      /* 	ThisVal=fabs(ThisVal); */
	      /* } */

	      /* //printf("%f, %f \n",a[ii],ThisVal); */
	      /* if (ThisVal > ThisMax){ */
	      /* 	ThisMax=ThisVal; */
	      /* 	ThisxMax=i_a; */
	      /* 	ThisyMax=j_a; */
	      /* } */


	      

      	      /* printf("%i %i %i\n",i_a,j_a,ii); */
	      /* printf("%i %i %f\n",ThisxMax,ThisyMax,ThisMax); */
      	    }
      	}

      MaxBlock[iblock]=ThisMax;
      xMaxBlock[iblock]=ThisxMax;
      yMaxBlock[iblock]=ThisyMax;
      //printf("maxc loop: %i %i %f\n",xMaxBlock[iblock],yMaxBlock[iblock],MaxBlock[iblock]);
      //printf("maxc loop2: %i %i %f\n",ThisxMax,ThisyMax,ThisMax);

    }
  }
  
  float Max=0;
  int xMax=0;
  int yMax=0;
  float* ans = p_float32(Ans);
  for (iblock = 0; iblock < nblocks-1; iblock++){
    if(MaxBlock[iblock]>Max){
      Max=MaxBlock[iblock];
      xMax=xMaxBlock[iblock];
      yMax=yMaxBlock[iblock];
    }
  }

  //printf("maxc: %i %i %f\n",xMax,yMax,Max);
  ans[0]=(float)xMax;
  ans[1]=(float)yMax;
  ans[2]=(float)Max;
  free(MaxBlock);
  free(xMaxBlock);
  free(yMaxBlock);
  return PyArray_Return(Ans);

}

static PyObject *pyWhereMax(PyObject *self, PyObject *args)
{
  PyArrayObject *A, *Blocks,*Ans;
  PyObject *ObjAns;
  int doabs;

  if (!PyArg_ParseTuple(args, "O!O!Oi", 
			&PyArray_Type,  &A,
			&PyArray_Type,  &Blocks,
			&ObjAns,
			&doabs
			))  return NULL;
  
  //  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  
  Ans = (PyArrayObject *) PyArray_ContiguousFromObject(ObjAns, PyArray_FLOAT32, 0, 4);
  
  int nx,ny,NX,NY,np;
  
  NX=A->dimensions[0];
  NY=A->dimensions[1];
  //printf("dims %i %i\n",NX,NY);
  



  long iblock;
  int* pBlocks=p_int32(Blocks);
  int nblocks=Blocks->dimensions[0];
  float *MaxBlock;
  int *xMaxBlock;
  int *yMaxBlock;
  MaxBlock=malloc((nblocks-1)*sizeof(float));
  xMaxBlock=malloc((nblocks-1)*sizeof(int));
  yMaxBlock=malloc((nblocks-1)*sizeof(int));

  {
#pragma omp parallel for private(iblock)
    for (iblock = 0; iblock < nblocks-1; iblock++){
      int i0=pBlocks[iblock];
      int i1=pBlocks[iblock+1];
      if(i1>=NX){i1=NX;};
      
      //printf("- block %i->%i\n",i0,i1);
      
      float* a = p_float32(A);
      int i_a,j_a;
      float ThisMax=-FLT_MAX;
      int ThisxMax=0;
      int ThisyMax=0;
      float ThisVal;
      for (i_a = i0; i_a < i1; i_a++)
      	{
      	  for (j_a = 0; j_a < NY; j_a++)
      	    {
      	      int ii=i_a*NY+j_a;
      	      /* ThisMax  = ((a[ii] > ThisMax) ? a[ii] : ThisMax); */
      	      /* ThisxMax = ((a[ii] > ThisMax) ? i_a : ThisxMax); */
      	      /* ThisyMax = ((a[ii] > ThisMax) ? j_a : ThisyMax); */
	      ThisVal=a[ii];
	      if(doabs==1){
		ThisVal=fabs(ThisVal);
	      }
	      //printf("%f, %f \n",a[ii],ThisVal);
	      if (ThisVal > ThisMax){
		ThisMax=ThisVal;
		ThisxMax=i_a;
		ThisyMax=j_a;
	      }
	      

      	      /* printf("%i %i %i\n",i_a,j_a,ii); */
	      /* printf("%i %i %f\n",ThisxMax,ThisyMax,ThisMax); */
      	    }
      	}

      MaxBlock[iblock]=ThisMax;
      xMaxBlock[iblock]=ThisxMax;
      yMaxBlock[iblock]=ThisyMax;
      //printf("maxc loop: %i %i %f\n",xMaxBlock[iblock],yMaxBlock[iblock],MaxBlock[iblock]);
      //printf("maxc loop2: %i %i %f\n",ThisxMax,ThisyMax,ThisMax);

    }
  }
  
  float Max=0;
  int xMax=0;
  int yMax=0;
  float* ans = p_float32(Ans);
  for (iblock = 0; iblock < nblocks-1; iblock++){
    if(MaxBlock[iblock]>Max){
      Max=MaxBlock[iblock];
      xMax=xMaxBlock[iblock];
      yMax=yMaxBlock[iblock];
    }
  }

  //printf("maxc: %i %i %f\n",xMax,yMax,Max);
  ans[0]=(float)xMax;
  ans[1]=(float)yMax;
  ans[2]=(float)Max;
  free(MaxBlock);
  free(xMaxBlock);
  free(yMaxBlock);
  return PyArray_Return(Ans);

}


static PyObject *pyAddArray(PyObject *self, PyObject *args)
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
			))  return NULL;
  
  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  
  
  int nx,ny,NX,NY,np;
  
  NX=A->dimensions[0];
  int NYa=A->dimensions[1];
  int NYb=B->dimensions[1];
  //printf("dims %i %i\n",NX,NY);
  
  int * aedge = p_int32(Aedge);
  int a_x0=aedge[0];
  int a_x1=aedge[1];
  int a_y0=aedge[2];
  int a_y1=aedge[3];

  int * bedge = p_int32(Bedge);
  int b_x0=bedge[0];
  int b_x1=bedge[1];
  int b_y0=bedge[2];
  int b_y1=bedge[3];



  long iblock;
  int* pBlocks=p_int32(Blocks);
  int nblocks=Blocks->dimensions[0];

/* for (i = 0; i < nx; i++) */
/*   { */
/* for (j = 0; j < ny; j++) */
/*   { */
/* a[i*ny+j] += b[i*ny+j];// * 2;//(factor); */
/* } */
/* } */

  {
#pragma omp parallel for private(iblock)
    for (iblock = 0; iblock < nblocks-1; iblock++){
      int i0=pBlocks[iblock];
      int i1=pBlocks[iblock+1];
      if(i1>=a_x1){i1=a_x1;};
      
      //printf("- block %i->%i\n",i0,i1);
      
      float* a = p_float32(A);
      float* b = p_float32(B);
      int i_a,j_a;

      for (i_a = i0; i_a < i1; i_a++)
	{
	  int di=i_a-a_x0;
	  int i_b=b_x0+di;
	  for (j_a = a_y0; j_a < a_y1; j_a++)
	    {
	      int dj=j_a-a_y0;
	      int j_b=b_y0+dj;
	      //printf("a[%i,%i] = b[%i,%i] * %f\n",i_a,j_a,i_b,j_b,factor); 
	      float bef=a[i_a*NYa+j_a];
	      a[i_a*NYa+j_a] += b[i_b*NYb+j_b]*(factor);
	      float aft=a[i_a*NYa+j_a];

	      //printf("    %f->%f\n",bef,aft);
	      
	    }
	}
      
    }
  }


/* float* a = p_float32(A); */

/* for (i = 0; i < nx; i++) */
/*   { */
/* for (j = 0; j < ny; j++) */
/*   { */
/* printf("%f\n",a[i*ny+j]); */
/* } */
/* } */
  

  return PyArray_Return(A);//,PyArray_Return(np_grid);

}







static PyObject *pyProdArray(PyObject *self, PyObject *args)
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
			))  return NULL;
  
  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  
  
  int nx,ny,NX,NY,np;
  
  NX=A->dimensions[0];
  int NYa=A->dimensions[1];
  int NYb=B->dimensions[1];
  //printf("dims %i %i\n",NX,NY);
  
  int * aedge = p_int32(Aedge);
  int a_x0=aedge[0];
  int a_x1=aedge[1];
  int a_y0=aedge[2];
  int a_y1=aedge[3];

  int * bedge = p_int32(Bedge);
  int b_x0=bedge[0];
  int b_x1=bedge[1];
  int b_y0=bedge[2];
  int b_y1=bedge[3];



  long iblock;
  int* pBlocks=p_int32(Blocks);
  int nblocks=Blocks->dimensions[0];

/* for (i = 0; i < nx; i++) */
/*   { */
/* for (j = 0; j < ny; j++) */
/*   { */
/* a[i*ny+j] += b[i*ny+j];// * 2;//(factor); */
/* } */
/* } */

  {
#pragma omp parallel for private(iblock)
    for (iblock = 0; iblock < nblocks-1; iblock++){
      int i0=pBlocks[iblock];
      int i1=pBlocks[iblock+1];
      if(i1>=a_x1){i1=a_x1;};
      
      //printf("- block %i->%i\n",i0,i1);
      
      float* a = p_float32(A);
      float* b = p_float32(B);
      int i_a,j_a;

      for (i_a = i0; i_a < i1; i_a++)
	{
	  int di=i_a-a_x0;
	  int i_b=b_x0+di;
	  for (j_a = a_y0; j_a < a_y1; j_a++)
	    {
	      int dj=j_a-a_y0;
	      int j_b=b_y0+dj;
	      //printf("a[%i,%i] = b[%i,%i] * %f\n",i_a,j_a,i_b,j_b,factor); 
	      //float bef=a[i_a*NYa+j_a];
	      a[i_a*NYa+j_a] *= b[i_b*NYb+j_b]*(factor);
	      //float aft=a[i_a*NYa+j_a];

	      //printf("    %f->%f\n",bef,aft);
	      
	    }
	}
      
    }
  }


/* float* a = p_float32(A); */

/* for (i = 0; i < nx; i++) */
/*   { */
/* for (j = 0; j < ny; j++) */
/*   { */
/* printf("%f\n",a[i*ny+j]); */
/* } */
/* } */
  

  return PyArray_Return(A);//,PyArray_Return(np_grid);

}


static PyObject *pyDivArray(PyObject *self, PyObject *args)
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
			))  return NULL;
  
  A = (PyArrayObject *) PyArray_ContiguousFromObject(ObjA, PyArray_FLOAT32, 0, 4);
  
  
  int nx,ny,NX,NY,np;
  
  NX=A->dimensions[0];
  int NYa=A->dimensions[1];
  int NYb=B->dimensions[1];
  //printf("dims %i %i\n",NX,NY);
  
  int * aedge = p_int32(Aedge);
  int a_x0=aedge[0];
  int a_x1=aedge[1];
  int a_y0=aedge[2];
  int a_y1=aedge[3];

  int * bedge = p_int32(Bedge);
  int b_x0=bedge[0];
  int b_x1=bedge[1];
  int b_y0=bedge[2];
  int b_y1=bedge[3];



  long iblock;
  int* pBlocks=p_int32(Blocks);
  int nblocks=Blocks->dimensions[0];

/* for (i = 0; i < nx; i++) */
/*   { */
/* for (j = 0; j < ny; j++) */
/*   { */
/* a[i*ny+j] += b[i*ny+j];// * 2;//(factor); */
/* } */
/* } */

  {
#pragma omp parallel for private(iblock)
    for (iblock = 0; iblock < nblocks-1; iblock++){
      int i0=pBlocks[iblock];
      int i1=pBlocks[iblock+1];
      if(i1>=a_x1){i1=a_x1;};
      
      //printf("- block %i->%i\n",i0,i1);
      
      float* a = p_float32(A);
      float* b = p_float32(B);
      int i_a,j_a;

      for (i_a = i0; i_a < i1; i_a++)
	{
	  int di=i_a-a_x0;
	  int i_b=b_x0+di;
	  for (j_a = a_y0; j_a < a_y1; j_a++)
	    {
	      int dj=j_a-a_y0;
	      int j_b=b_y0+dj;
	      //printf("a[%i,%i] = b[%i,%i] * %f\n",i_a,j_a,i_b,j_b,factor); 
	      //float bef=a[i_a*NYa+j_a];
	      a[i_a*NYa+j_a] /= b[i_b*NYb+j_b]*(factor);
	      //float aft=a[i_a*NYa+j_a];

	      //printf("    %f->%f\n",bef,aft);
	      
	    }
	}
      
    }
  }


/* float* a = p_float32(A); */

/* for (i = 0; i < nx; i++) */
/*   { */
/* for (j = 0; j < ny; j++) */
/*   { */
/* printf("%f\n",a[i*ny+j]); */
/* } */
/* } */
  

  return PyArray_Return(A);//,PyArray_Return(np_grid);

}







