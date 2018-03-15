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

/* Header to test of C modules for arrays for Python: C_test.c */
#include "complex.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <time.h>





/* ==== Prototypes =================================== */

// .... Python callable Vector functions ..................


/* .... C vector utility functions ..................*/
//PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int *pyvector_to_Carrayptrs2(PyArrayObject *arrayin);
//===========================================
double complex *GetCp(PyArrayObject *arrayin);
double complex *Complex_pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int *Int_pyvector_to_Carrayptrs(PyArrayObject *arrayin);
//int  not_doublevector(PyArrayObject *vec);
int nint(double n){
  //  double x=n+0.5;
  //printf("%f+0.5= %f\n",n,x);
  return floor(n+0.5);};

/* .... Python callable Matrix functions ..................*/

int *I_ptr(PyArrayObject *arrayin)  {
	return (int *) arrayin->data;
}


int *p_int32(PyArrayObject *arrayin)  {
  return (int *) arrayin->data;  /* pointer to arrayin data as double */
}

long int *p_int64(PyArrayObject *arrayin)  {
  return (long int *) arrayin->data;  /* pointer to arrayin data as double */
}


double *p_float64(PyArrayObject *arrayin)  {
  return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

float *p_float32(PyArrayObject *arrayin)  {
  return (float *) arrayin->data;  /* pointer to arrayin data as double */
}


float complex *p_complex64(PyArrayObject *arrayin)  {
  return (float complex *) arrayin->data;  /* pointer to arrayin data as double */
}

double complex *p_complex128(PyArrayObject *arrayin)  {
  return (double complex *) arrayin->data;  /* pointer to arrayin data as double */
}

bool *p_bool(PyArrayObject *arrayin)  {
  return (bool *) arrayin->data;  /* pointer to arrayin data as double */
}

char *p_char(PyArrayObject *arrayin)  {
  return (char *) arrayin->data;  /* pointer to arrayin data as double */
}

short int *p_int16(PyArrayObject *arrayin)  {
  return (short int *) arrayin->data;  /* pointer to arrayin data as double */
}

static PyObject *pyTestMatrix(PyObject *self, PyObject *args);
void MatInv(float complex *A, float complex* B, int H );
//void MatDot(float complex *A, float complex* B, float complex* Out );



static PyObject *pyGridderPoints(PyObject *self, PyObject *args);

static PyObject *pyAccumulateWeightsOntoGrid(PyObject *self, PyObject *args);


static PyObject *pyGridderWPol(PyObject *self, PyObject *args);
static PyObject *pyAddArray(PyObject *self, PyObject *args);
static PyObject *pyWhereMax(PyObject *self, PyObject *args);

//double PI=3.141592653589793238462643383279502884197;
float C=299792458.;
float PI=3.141592653589793238462643383279502884197;


float GiveDecorrelationFactor(int FSmear, int TSmear,
			      float l0, float m0,
			      double* uvwPtr,
			      double* uvw_dt_Ptr,
			      float nu,
			      float Dnu, 
			      float DT){
  //float PI=3.141592653589793238462643383279502884197;
  //float C=2.99792458e8;

  float n0=sqrt(1.-l0*l0-m0*m0)-1.;
  float DecorrFactor=1.;
  float phase=0;
  float phi=0;
  float df;
  phase=(uvwPtr[0])*l0;
  phase+=(uvwPtr[1])*m0;
  phase+=(uvwPtr[2])*n0;

  if(FSmear==1){
    phi=PI*(Dnu/C)*phase;
    //printf("%f %f %f %f = %f\n",PI,Dnu,C,phase,phi);
    if(phi!=0.){
      df=(float)(sin((double)phi)/((double)phi));
//      if( df<0 )
//        df=0;
      DecorrFactor*=df;
    };
  };

  float du,dv,dw;
  float dphase;
  if(TSmear==1){
    
    du=uvw_dt_Ptr[0]*l0;
    dv=uvw_dt_Ptr[1]*m0;
    dw=uvw_dt_Ptr[2]*n0;
    dphase=(du+dv+dw)*DT;
    phi=PI*(nu/C)*dphase;
    if(phi!=0.){
      df=(sin(phi)/(phi));
//      if( df<0 )
//        df=0;
      DecorrFactor*=df;
    };
  };
  return DecorrFactor;
}

static PyObject *pyDeGridderWPol(PyObject *self, PyObject *args);

int FullScalarMode;
int ScalarJones;
int ScalarVis;

