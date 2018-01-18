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



static PyObject *pyTestMatrix(PyObject *self, PyObject *args);
void MatInv(float complex *A, float complex* B, int H );
void MatDot(float complex *A, float complex* B, float complex* Out );



static PyObject *pyGridderPoints(PyObject *self, PyObject *args);




static PyObject *pyGridderWPol(PyObject *self, PyObject *args);
static PyObject *pyAddArray(PyObject *self, PyObject *args);
static PyObject *pyWhereMax(PyObject *self, PyObject *args);
float C=299792458.;



void gridderWPol(PyArrayObject *np_grid,
	      PyArrayObject *vis,
	      PyArrayObject *uvw,
	      PyArrayObject *flags,
	      PyArrayObject *weigths,
	      PyArrayObject *sumwt,
	      int dopsf,
	      PyObject *Lcfs,
	      PyObject *LcfsConj,
	      PyArrayObject *Winfos,
	      PyArrayObject *increment,
	      PyArrayObject *freqs,
	      PyObject *Lmaps, 
	      PyObject *LJones);

static PyObject *pyDeGridderWPol(PyObject *self, PyObject *args);

void DeGridderWPol(PyArrayObject *np_grid,
	      PyArrayObject *vis,
	      PyArrayObject *uvw,
	      PyArrayObject *flags,
		   //PyArrayObject *rows,
	      PyArrayObject *sumwt,
	      int dopsf,
	      PyObject *Lcfs,
	      PyObject *LcfsConj,
	      PyArrayObject *Winfos,
	      PyArrayObject *increment,
	      PyArrayObject *freqs,
	      PyObject *Lmaps, 
	      PyObject *LJones, 
	      PyObject *LSmear);

void ScaleJones(float complex* J0, float AlphaScaleJones){
  float complex z0;
  int ThisPol;
  for(ThisPol =0; ThisPol<4;ThisPol++){
    if(cabs(J0[ThisPol])!=0.){
      z0=J0[ThisPol]/cabs(J0[ThisPol]);
      J0[ThisPol]=(1.-AlphaScaleJones)*z0+AlphaScaleJones*J0[ThisPol];
      //J0[ThisPol]=z0+AlphaScaleJones*(J0[ThisPol]-z0);
    }
  }
}


double PI=3.141592653589793238462643383279502884197;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


void MatInv(float complex *A, float complex* B, int H ){
  float complex a,b,c,d,ff;

  if(H==0){
      a=A[0];
      b=A[1];
      c=A[2];
      d=A[3];}
  else{
    a=conj(A[0]);
    b=conj(A[2]);
    c=conj(A[1]);
    d=conj(A[3]);
  }  
  ff=1./((a*d-c*b));
  B[0]=ff*d;
  B[1]=-ff*b;
  B[2]=-ff*c;
  B[3]=ff*a;
}

void MatH(float complex *A, float complex* B){
  float complex a,b,c,d;

  a=conj(A[0]);
  b=conj(A[2]);
  c=conj(A[1]);
  d=conj(A[3]);
  B[0]=a;
  B[1]=b;
  B[2]=c;
  B[3]=d;
}

void MatDot(float complex *A, float complex* B, float complex* Out){
  float complex a0,b0,c0,d0;
  float complex a1,b1,c1,d1;

  a0=A[0];
  b0=A[1];
  c0=A[2];
  d0=A[3];
  
  a1=B[0];
  b1=B[1];
  c1=B[2];
  d1=B[3];
  
  Out[0]=a0*a1+b0*c1;
  Out[1]=a0*b1+b0*d1;
  Out[2]=c0*a1+d0*c1;
  Out[3]=c0*b1+d0*d1;

}

static PyObject *pyTestMatrix(PyObject *self, PyObject *args)
{
  PyArrayObject *Anp;

  if (!PyArg_ParseTuple(args, "O!",
			&PyArray_Type,  &Anp
			)
      )  return NULL;

  float complex* A  = p_complex64(Anp);
  float complex B[4];
  MatInv(A,B,1);
  int i;
  printf("inverse of input matrix:\n");
  for (i=0; i<4; i++){
    printf("%i: (%f,%f)\n",i,(float)creal(B[i]),(float)cimag(B[i]));
  };
   
  printf("\ndot product A.A^-1:\n");
  float complex Out[4];
  MatDot(A,B,Out);
  for (i=0; i<4; i++){
    printf("%i: (%f,%f)\n",i,(float)creal(Out[i]),(float)cimag(Out[i]));
  };

  printf("\n A^H:\n");
  MatH(A,B);
  for (i=0; i<4; i++){
    printf("%i: (%f,%f)\n",i,(float)creal(B[i]),(float)cimag(B[i]));
  };
  


  return Py_None;

}

int FullScalarMode;
int ScalarJones;
int ScalarVis;

void NormJones(float complex* J0, int ApplyAmp, int ApplyPhase, int DoScaleJones, double *uvwPtr, float WaveLengthMean, float CalibError){
  int ThisPol;
  int nPol=4;
  if(FullScalarMode){nPol=1;}
  if(ApplyAmp==0){
    for(ThisPol =0; ThisPol<nPol;ThisPol++){
      if(cabs(J0[ThisPol])!=0.){
	J0[ThisPol]/=cabs(J0[ThisPol]);
      }
    }
  }
	
  if(ApplyPhase==0){
    for(ThisPol =0; ThisPol<nPol;ThisPol++){
      J0[ThisPol]=cabs(J0[ThisPol]);
    }
  }
	
  if(DoScaleJones==1){
    float U2=uvwPtr[0]*uvwPtr[0];
    float V2=uvwPtr[1]*uvwPtr[1];
    float R2=(U2+V2)/(WaveLengthMean*WaveLengthMean);
    float CalibError2=CalibError*CalibError;
    float AlphaScaleJones=exp(-2.*PI*CalibError2*R2);
    ScaleJones(J0,AlphaScaleJones);
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

void GiveJones(float complex *ptrJonesMatrices, int *JonesDims, float *ptrCoefs, int i_t, int i_ant0, int i_dir, int Mode, float complex *Jout){
  int nd_Jones,na_Jones,nch_Jones;
  nd_Jones=JonesDims[1];
  na_Jones=JonesDims[2];
  nch_Jones=JonesDims[3];
  
  int ipol,idir;
  if(Mode==0){
    int offJ0=i_t*nd_Jones*na_Jones*nch_Jones*4
      +i_dir*na_Jones*nch_Jones*4
      +i_ant0*nch_Jones*4;
    for(ipol=0; ipol<4; ipol++){
      Jout[ipol]=*(ptrJonesMatrices+offJ0+ipol);
    }
  }

  if(Mode==1){
    for(idir=0; idir<nd_Jones; idir++){
      int offJ0=i_t*nd_Jones*na_Jones*nch_Jones*4
	+idir*na_Jones*nch_Jones*4
	+i_ant0*nch_Jones*4;
      for(ipol=0; ipol<4; ipol++){
	Jout[ipol]+=ptrCoefs[idir]*(*(ptrJonesMatrices+offJ0+ipol));
	
	//printf("%i, %f, %f, %f\n",ipol,ptrCoefs[idir],creal(Jout[ipol]),cimag(Jout[ipol]));
      }
      
    }
  }
}

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
  phase=(uvwPtr[0])*l0;
  phase+=(uvwPtr[1])*m0;
  phase+=(uvwPtr[2])*n0;

  if(FSmear==1){
    phi=PI*(Dnu/C)*phase;
    //printf("%f %f %f %f = %f\n",PI,Dnu,C,phase,phi);
    if(phi!=0.){
      DecorrFactor*=(float)(sin((double)phi)/((double)phi));
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
      DecorrFactor*=(sin(phi)/(phi));
    };
  };
  return DecorrFactor;
}
