/* Header to test of C modules for arrays for Python: C_test.c */
#include "complex.h"
#include <math.h>

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

static PyObject *pyGridderWPol(PyObject *self, PyObject *args);

void gridderWPol(PyArrayObject *np_grid,
	      PyArrayObject *vis,
	      PyArrayObject *uvw,
	      PyArrayObject *flags,
	      PyArrayObject *rows,
	      PyArrayObject *sumwt,
	      int dopsf,
	      PyObject *Lcfs,
	      PyObject *LcfsConj,
	      PyArrayObject *Winfos,
	      PyArrayObject *increment,
	      PyArrayObject *freqs,
	      PyObject *Lmaps);

/* .... C matrix utility functions ..................*/
//PyArrayObject *pymatrix(PyObject *objin);
//double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
//double **ptrvector(long n);
/* void free_Carrayptrs(double **v); */
/* void free_Carrayptrs2(double *v); */
/* int  not_doublematrix(PyArrayObject *mat); */

/* .... Python callable integer 2D array functions ..................*/


//* * .... C 2D int array utility functions ..................*\/ */
/* PyArrayObject *pyint2Darray(PyObject *objin); */
/* int **pyint2Darray_to_Carrayptrs(PyArrayObject *arrayin); */
/* int **ptrintvector(long n); */
/* void free_Cint2Darrayptrs(int **v); */
/* int  not_int2Darray(PyArrayObject *mat); */
