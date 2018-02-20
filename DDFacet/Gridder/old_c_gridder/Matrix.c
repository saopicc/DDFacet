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

#include "complex.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "arrayobject.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <time.h>

void printMat(float complex *A){
  printf("================================================================================================\n");
  printf("%20f + (1j)%20f | %20f + (1j)%20f\n",creal(A[0]),cimag(A[0]),creal(A[1]),cimag(A[1]));
  printf("%20f + (1j)%20f | %20f + (1j)%20f\n",creal(A[2]),cimag(A[2]),creal(A[3]),cimag(A[3]));
}

void Unity(float complex *A){
  A[0]=1.;
  A[1]=0.;
  A[2]=0.;
  A[3]=1.;
}

void Zero(float complex *A){
  A[0]=0.;
  A[1]=0.;
  A[2]=0.;
  A[3]=0.;
}


void MatInv(float complex *A, float complex* B, int H ){


  if(FullScalarMode)
    {
      B[0]=1./A[0]; 
    }
  else
    {
    float complex ff;
    ff=1./((A[0]*A[3]-A[2]*A[1]));
    B[0]=ff*A[3];
    B[1]=-ff*A[1];
    B[2]=-ff*A[2];
    B[3]=ff*A[0];
  }
}

/* void MatInv(float complex *A, float complex* B, int H ){ */
/*   float complex a,b,c,d,ff; */

/*   if(H==0){ */
/*       a=A[0]; */
/*       b=A[1]; */
/*       c=A[2]; */
/*       d=A[3];} */
/*   else{ */
/*     a=conj(A[0]); */
/*     b=conj(A[2]); */
/*     c=conj(A[1]); */
/*     d=conj(A[3]); */
/*   }   */
/*   ff=1./((a*d-c*b)); */
/*   B[0]=ff*d; */
/*   B[1]=-ff*b; */
/*   B[2]=-ff*c; */
/*   B[3]=ff*a; */
/* } */

void MatH(float complex *A, float complex* B){

      B[0]=conj(A[0]);
      B[1]=conj(A[2]);
      B[2]=conj(A[1]);
      B[3]=conj(A[3]);

  /* if(FullScalarMode) */
  /*   { */
  /*     B[0]=conj(A[0]); */
  /*   } */
  /* else */
  /*   { */
  /*     B[0]=conj(A[0]); */
  /*     B[1]=conj(A[2]); */
  /*     B[2]=conj(A[1]); */
  /*     B[3]=conj(A[3]); */
  /*   } */
}

void MatT(float complex *A, float complex* B){

      B[0]=(A[0]);
      B[1]=(A[2]);
      B[2]=(A[1]);
      B[3]=(A[3]);
  /* if(FullScalarMode) */
  /*   { */
  /*     B[0]=(A[0]); */
  /*   } */
  /* else */
  /*   { */
  /*     B[0]=(A[0]); */
  /*     B[1]=(A[2]); */
  /*     B[2]=(A[1]); */
  /*     B[3]=(A[3]); */
  /*   } */
}

void MatConj(float complex *A, float complex* B){

      B[0]=conj(A[0]);
      B[1]=conj(A[1]);
      B[2]=conj(A[2]);
      B[3]=conj(A[3]);

  /* if(FullScalarMode) */
  /*   { */
  /*     B[0]=conj(A[0]); */
  /*   } */
  /* else */
  /*   { */
  /*     B[0]=conj(A[0]); */
  /*     B[1]=conj(A[1]); */
  /*     B[2]=conj(A[2]); */
  /*     B[3]=conj(A[3]); */
  /*   } */
}


/* void MatH(float complex *A, float complex* B){ */
/*   float complex a,b,c,d; */

/*   a=conj(A[0]); */
/*   b=conj(A[2]); */
/*   c=conj(A[1]); */
/*   d=conj(A[3]); */
/*   B[0]=a; */
/*   B[1]=b; */
/*   B[2]=c; */
/*   B[3]=d; */
/* } */

// TypeMat:
// 0: scalar
// 1: diag
// 2: full

void Mat_A_l_SumProd(float complex *Out, int TypeMatOut, float complex lambda){
  

  Out[0]*=lambda;
  Out[1]*=lambda;
  Out[2]*=lambda;
  Out[3]*=lambda;

  /* if(TypeMatOut==0){ */
  /*   Out[0]*=lambda; */
  /*   Out[3]=Out[0]; */
  /* }else if (TypeMatOut==1){ */
  /*   Out[0]*=lambda; */
  /*   Out[3]*=lambda; */
  /* }else if(TypeMatOut==2){ */
  /*   Out[0]*=lambda; */
  /*   Out[1]*=lambda; */
  /*   Out[2]*=lambda; */
  /*   Out[3]*=lambda; */
  /* } */
  
}


void Mat_A_Bl_Sum(float complex *Out, int TypeMatOut, float complex* B, int TypeMatB, float complex lambda){

  Out[0]+=B[0]*lambda;
  Out[1]+=B[1]*lambda;
  Out[2]+=B[2]*lambda;
  Out[3]+=B[3]*lambda;

  /* TypeMatOut=2; */
  /* TypeMatB=2; */

  /* if(TypeMatOut==0){ */
  /*   if (TypeMatB==0){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[3]=Out[0]; */
  /*   }else if (TypeMatB==1){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[3]+=B[3]*lambda; */
  /*   }else if(TypeMatB==2){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[1]+=B[1]*lambda; */
  /*     Out[2]+=B[2]*lambda; */
  /*     Out[3]+=B[3]*lambda; */
  /*   } */
  /* }else if (TypeMatOut==1){ */
  /*   if(TypeMatB==0){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[3]+=B[0]*lambda; */
  /*   }else if(TypeMatB==1){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[3]+=B[3]*lambda; */
  /*   }else if(TypeMatB==2){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[1]+=B[1]*lambda; */
  /*     Out[2]+=B[2]*lambda; */
  /*     Out[3]+=B[3]*lambda; */
  /*   } */
  /* }else if(TypeMatOut==2){ */
  /*   if(TypeMatB==0){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[3]+=B[0]*lambda; */
  /*   }else if(TypeMatB==1){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[3]+=B[3]*lambda; */
  /*   }else if(TypeMatB==2){ */
  /*     Out[0]+=B[0]*lambda; */
  /*     Out[1]+=B[1]*lambda; */
  /*     Out[2]+=B[2]*lambda; */
  /*     Out[3]+=B[3]*lambda; */
  /*   } */
  /* } */

}


float complex DotBuf[4];
int iPolBuf;
void MatDot(float complex *A, int TypeMatA, float complex* B, int TypeMatB, float complex* Out){


  /* Out[0]=A[0]*B[0]; */
  /* Out[3]=Out[0]; */
  DotBuf[0]=A[0]*B[0]+A[1]*B[2];
  DotBuf[1]=A[0]*B[1]+A[1]*B[3];
  DotBuf[2]=A[2]*B[0]+A[3]*B[2];
  DotBuf[3]=A[2]*B[1]+A[3]*B[3];

  Out[0]=DotBuf[0];
  Out[1]=DotBuf[1];
  Out[2]=DotBuf[2];
  Out[3]=DotBuf[3];

  

  /* if(TypeMatA==0){ */
  /*   if (TypeMatB==0){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[3]=Out[0]; */
  /*   }else if (TypeMatB==1){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[3]=A[0]*B[3]; */
  /*   }else if(TypeMatB==2){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[1]=A[0]*B[1]; */
  /*     Out[2]=A[0]*B[2]; */
  /*     Out[3]=A[0]*B[3]; */
  /*   } */
  /* }else if (TypeMatA==1){ */
  /*   if(TypeMatB==0){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[3]=A[3]*B[0]; */
  /*   }else if(TypeMatB==1){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[3]=A[3]*B[3]; */
  /*   }else if(TypeMatB==2){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[1]=A[0]*B[1]; */
  /*     Out[2]=A[3]*B[2]; */
  /*     Out[3]=A[3]*B[3]; */
  /*   } */
  /* }else if(TypeMatA==2){ */
  /*   if(TypeMatB==0){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[1]=A[1]*B[0]; */
  /*     Out[2]=A[2]*B[0]; */
  /*     Out[3]=A[3]*B[0]; */
  /*   }else if(TypeMatB==1){ */
  /*     Out[0]=A[0]*B[0]; */
  /*     Out[1]=A[1]*B[3]; */
  /*     Out[2]=A[2]*B[0]; */
  /*     Out[3]=A[3]*B[3]; */
  /*   }else if(TypeMatB==2){ */
  /*     Out[0]=A[0]*B[0]+A[1]*B[2]; */
  /*     Out[1]=A[0]*B[1]+A[1]*B[3]; */
  /*     Out[2]=A[2]*B[0]+A[3]*B[2]; */
  /*     Out[3]=A[2]*B[1]+A[3]*B[3]; */
  /*   } */
  /* } */
  
}



/* /\* void MatDot(float complex *A, float complex* B, float complex* Out){ *\/ */
/* /\*   float complex a0,b0,c0,d0; *\/ */
/* /\*   float complex a1,b1,c1,d1; *\/ */

/* /\*   a0=A[0]; *\/ */
/* /\*   b0=A[1]; *\/ */
/* /\*   c0=A[2]; *\/ */
/* /\*   d0=A[3]; *\/ */
  
/* /\*   a1=B[0]; *\/ */
/* /\*   b1=B[1]; *\/ */
/* /\*   c1=B[2]; *\/ */
/* /\*   d1=B[3]; *\/ */
  
/* /\*   Out[0]=a0*a1+b0*c1; *\/ */
/* /\*   Out[1]=a0*b1+b0*d1; *\/ */
/* /\*   Out[2]=c0*a1+d0*c1; *\/ */
/* /\*   Out[3]=c0*b1+d0*d1; *\/ */

/* /\* } *\/ */

static PyObject *pyTestMatrix(PyObject *self, PyObject *args)
{
  PyArrayObject *Anp,*Bnp;
  int TypeMatA;
  int TypeMatB;
  float lambda;
 
  if (!PyArg_ParseTuple(args, "O!O!iif",
			&PyArray_Type,  &Anp,
			&PyArray_Type,  &Bnp,
			&TypeMatA,
			&TypeMatB,
			&lambda
			)
      )  return NULL;

  float complex* A  = p_complex64(Anp);
  /* float complex B[4]; */
  /* MatInv(A,B,1); */
  int i;
  /* printf("inverse of input matrix:\n"); */
  /* for (i=0; i<4; i++){ */
  /*   printf("%i: (%f,%f)\n",i,(float)creal(B[i]),(float)cimag(B[i])); */
  /* }; */

   
  float complex* B  = p_complex64(Bnp);
  //printf("\ndot product A.A^-1:\n");
  float complex Out[4];
  MatDot(A,TypeMatA,B,TypeMatB,Out);
  //Mat_A_Bl_Sum(Out,TypeMatA,B,TypeMatB,(float complex) lambda);
  printf("%f %f",(float)creal(Out[0]),(float)cimag(Out[0]));
  printf(" | %f %f\n",(float)creal(Out[1]),(float)cimag(Out[1]));
  printf("%f %f",(float)creal(Out[2]),(float)cimag(Out[2]));
  printf(" | %f %f\n",(float)creal(Out[3]),(float)cimag(Out[3]));



  /* printf("\n A^H:\n"); */
  /* MatH(A,B); */
  /* for (i=0; i<4; i++){ */
  /*   printf("%i: (%f,%f)\n",i,(float)creal(B[i]),(float)cimag(B[i])); */
  /* }; */
  


  return Py_None;

}
