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
	      PyObject *LJones);

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


double PI=3.14159265359;

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
