/**
 * Gridding and degridding with time and frequency averaging and applying
 * (average) time/frequency dependent gains.
 */
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <Python.h>
#include <omp.h>

#include "Stokes.h"
#include "Tools.h"
#include "JonesServer.h"
#include "Timer.h"
#include "arrayobject.h"
#include "pyArrayCast.h"
#include "Constants.h"


/* .... C vector utility functions ..................*/
//PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int *pyvector_to_Carrayptrs2(PyArrayObject *arrayin);
//===========================================
double complex *GetCp(PyArrayObject *arrayin);
double complex *Complex_pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int *Int_pyvector_to_Carrayptrs(PyArrayObject *arrayin);


/* ==== Setup _pyGridderSmearPols python module functions ====================== */
static PyObject *pyGridderWPol(PyObject *self,
				PyObject *args);
static PyObject *pyDeGridderWPol(PyObject *self,
				  PyObject *args);

static PyMethodDef _pyGridderSmearPols_testMethods[] = {
    {"pyGridderWPol", pyGridderWPol, METH_VARARGS},
    {"pyDeGridderWPol", pyDeGridderWPol, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the _pyGridderSmearPols functions ====================== */
// Module name must be _pyGridderSmearPols in compile and linked
void init_pyGridderSmearPols()  {
    (void) Py_InitModule("_pyGridderSmearPols", _pyGridderSmearPols_testMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}

int nint(double n) {
    //  double x=n+0.5;
    //printf("%f+0.5= %f\n",n,x);
    return floor(n+0.5);
};

float GiveDecorrelationFactor(int FSmear, int TSmear,
                              float l0, float m0,
                              double* uvwPtr,
                              double* uvw_dt_Ptr,
                              float nu,
                              float Dnu,
                              float DT);

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
                 PyObject *LJones,
                 PyArrayObject *SmearMapping,
                 PyObject *LOptimisation,
                 PyObject *LSmear,
                 PyArrayObject *np_ChanMapping);


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
                   PyArrayObject *SmearMapping,
                   PyObject *LOptimisation,
                   PyObject *LSmear,
                   PyArrayObject *np_ChanMapping);

int FullScalarMode;
int ScalarJones;
int ScalarVis;
