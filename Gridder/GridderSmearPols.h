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
#include "cerror.h"
#include "Stokes.h"
#include "Tools.h"
#include "JonesServer.h"
#include "Timer.h"
#include "arrayobject.h"
#include "pyArrayCast.h"
#include "Constants.h"
#include "gridding_parameters.h"

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

void parse_python_objects(PyArrayObject *grid,
			  PyArrayObject *vis,
			  PyArrayObject *uvw,
			  PyArrayObject *flags,
			  PyArrayObject *weights,
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
			  PyObject *LSmearing,
			  PyArrayObject *np_ChanMapping,
			  PyArrayObject *data_corr_products,
			  PyArrayObject *output_stokes_products,
			  gridding_parameters * out);

void gridderWPol(gridding_parameters * params);

void DeGridderWPol(gridding_parameters *  params);

int FullScalarMode;
int ScalarJones;
int ScalarVis;
