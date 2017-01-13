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

#include "Python.h"
#include "arrayobject.h"
#include "Gridder1D.h"
#include <math.h>
#include "complex.h"

/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef _pyGridder_testMethods[] = {
	{"pyGridderWPol", pyGridderWPol, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_pyGridder1D()  {
	(void) Py_InitModule("_pyGridder1D", _pyGridder_testMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *pyGridderWPol(PyObject *self, PyObject *args)
{
  PyObject *ObjGridIn;
  PyArrayObject *np_grid, *vis, *uvw, *cfs, *flags, *rows, *sumwt, *increment, *freqs,*WInfos;

  PyObject *Lcfs;
  PyObject *Lmaps;
  PyObject *LcfsConj;
  int dopsf;

  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!iO!O!O!O!O!O!", 
			&ObjGridIn,
			&PyArray_Type,  &vis, 
			&PyArray_Type,  &uvw, 
			&PyArray_Type,  &flags, 
			&PyArray_Type,  &rows, 
			&PyArray_Type,  &sumwt, 
			&dopsf, 
			&PyList_Type, &Lcfs,
			&PyList_Type, &LcfsConj,
			&PyArray_Type,  &WInfos,
			&PyArray_Type,  &increment,
			&PyArray_Type,  &freqs,
			&PyList_Type, &Lmaps
			))  return NULL;
  int nx,ny,nz,nzz;

  np_grid = (PyArrayObject *) PyArray_ContiguousFromObject(ObjGridIn, PyArray_COMPLEX128, 0, 4);
  nx=np_grid->dimensions[0];
  ny=np_grid->dimensions[1];
  nz=np_grid->dimensions[2];
  nzz=np_grid->dimensions[3];
  //printf("grid dims (%i,%i,%i,%i)\n",nx,ny,nz,nzz);
  nx=vis->dimensions[0];
  ny=vis->dimensions[1];
  nz=vis->dimensions[2];
  //printf("vis  dims (%i,%i,%i)\n",nx,ny,nz);
  nx=uvw->dimensions[0];
  ny=uvw->dimensions[1];
  nz=uvw->dimensions[2];
  //printf("uvw  dims (%i,%i)\n",nx,ny);


  int i=0;
  for (i=0; i<10; i++) {
    //    *cfs = PyList_GetItem(Lcfs, i);
    cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, i), PyArray_COMPLEX128, 0, 2);
    int sh=cfs->dimensions[0];
    //printf("dim %i\n",sh);
  }

  //gridderW(np_grid, vis, uvw, flags, rows, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs);

  gridderWPol(np_grid, vis, uvw, flags, rows, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps);
  
  return PyArray_Return(np_grid);

}

void gridderWPol(PyArrayObject *grid,
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
		 PyObject *Lmaps)
  {
    // Get size of convolution functions.
    PyArrayObject *cfs;
    PyArrayObject *NpPolMap;
    NpPolMap = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 0), PyArray_INT32, 0, 4);
    int npolsMap=NpPolMap->dimensions[0];
    int* PolMap=I_ptr(NpPolMap);
    
    //printf("npols=%i %i\n",npolsMap,PolMap[3]);

    // Get size of grid.
    double* ptrWinfo = pyvector_to_Carrayptrs(Winfos);
    double WaveRefWave = ptrWinfo[0];
    double wmax = ptrWinfo[1];
    double NwPlanes = ptrWinfo[2];
    int OverS=floor(ptrWinfo[3]);


    //printf("WaveRef=%f, wmax=%f \n",WaveRefWave,wmax);
    int nGridX    = grid->dimensions[3];
    int nGridY    = grid->dimensions[2];
    int nGridPol  = grid->dimensions[1];
    int nGridChan = grid->dimensions[0];

    // Get visibility data size.
    int nVisPol   = flags->dimensions[2];
    int nVisChan  = flags->dimensions[1];
    int nrows     = uvw->dimensions[0];
    //printf("(nrows, nVisChan, nVisPol)=(%i, %i, %i)\n",nrows,nVisChan,nVisPol);


    // Get oversampling and support size.
    int sampx = OverS;//int (cfs.sampling[0]);
    int sampy = OverS;//int (cfs.sampling[1]);

    double* __restrict__ sumWtPtr = pyvector_to_Carrayptrs(sumwt);//->data;
    double complex psfValues[4];
    psfValues[0] = psfValues[1] = psfValues[2] = psfValues[3] = 1;

    //uint inxRowWCorr(0);

    double offset_p[2],uvwScale_p[2];

    offset_p[0]=nGridX/2;//(nGridX-1)/2.;
    offset_p[1]=nGridY/2;
    float fnGridX=nGridX;
    float fnGridY=nGridY;
    double *incr=pyvector_to_Carrayptrs(increment);
    double *Pfreqs=pyvector_to_Carrayptrs(freqs);
    uvwScale_p[0]=fnGridX*incr[0];
    uvwScale_p[1]=fnGridX*incr[1];
    //printf("uvscale=(%f %f)",uvwScale_p[0],uvwScale_p[1]);
    double C=2.99792458e8;
    int inx;
    // Loop over all visibility rows to process.
    for (inx=0; inx<nrows; inx++) {
      int irow = inx;//rows[inx];
      //printf("irow=%i/%i\n",irow,nrows);
      //const double*  __restrict__ uvwPtr   = GetDp(uvw) + irow*3;
      double*  __restrict__ uvwPtr   = pyvector_to_Carrayptrs(uvw) + irow*3;
      /* const Float*   __restrict__ imgWtPtr = imagingWeight_p.data() + */
      /*                                        irow * nVisChan; */

      //printf("u=%f",*uvwPtr);
      int visChan;
      for (visChan=0; visChan<nVisChan; ++visChan) {
        int gridChan = 0;//chanMap_p[visChan];
        int CFChan = 0;//ChanCFMap[visChan];
	double recipWvl = Pfreqs[visChan] / C;
	double ThisWaveLength=C/Pfreqs[visChan];
	//printf("visChan=%i \n",visChan);
	
	//W-projection
	double wcoord=uvwPtr[2];
	
	int iwplane = floor((NwPlanes-1)*abs(wcoord)*(WaveRefWave/ThisWaveLength)/wmax);
	int skipW=0;
	if(iwplane>NwPlanes-1){skipW=1;continue;};

	//int iwplane = floor((NwPlanes-1)*abs(wcoord)/wmax);

	//printf("wcoord=%f, iw=%i \n",wcoord,iwplane);

	if(wcoord>0){
	  cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX128, 0, 2);
	} else{
	  cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX128, 0, 2);
	}
	int nConvX = cfs->dimensions[0];
	int nConvY = cfs->dimensions[1];
	int supx = (nConvX/OverS-1)/2;
	int supy = (nConvY/OverS-1)/2;

	
	/* printf("%i %i %i\n",nConvX,sampx,supx); */
	/* assert(1==0); */



	//printf("wcoord=%f, iw=%i, nConvX=%i ,revert=%i\n",wcoord,iwplane,nConvX,revert);


	//	printf("\n");

	//chanMap_p[visChan]=0;

        if (gridChan >= 0  &&  gridChan < nGridChan) {

          // Determine the grid position from the UV coordinates in wavelengths.
	  double posx,posy;

	  //For Even/Odd take the -1 off
	  posx = uvwScale_p[0] * uvwPtr[0] * recipWvl + offset_p[0];//-1;
	  posy = 0;//uvwScale_p[1] * uvwPtr[1] * recipWvl + offset_p[1]-1;

	  //printf("u=%8.2f, v=%8.2f, uvsc=%f, recip=%f, offset_p=%f, %f %f\n",uvwPtr[0],uvwPtr[1],uvwScale_p[0],recipWvl,offset_p[0],fnGridX,incr[0]);
	  //printf("posx=%6.2f, posy=%6.2f\n",posx,posy);
          int locx = nint (posx);    // location in grid
          int locy = nint (posy);
	  //printf("locx=%i, locy=%i\n",locx,locy);
          double diffx = locx - posx;
          double diffy = locy - posy;
	  //printf("diffx=%f, diffy=%f\n",diffx,diffy);
          int offx = nint (diffx * sampx); // location in
          int offy = nint (diffy * sampy); // oversampling
	  //printf("offx=%i, offy=%i\n",offx,offy);
          offx += (nConvX-1)/2;
          offy += (nConvY-1)/2;
	  offy=0;
          // Scaling with frequency is not necessary (according to Cyril).
          double freqFact = 1;
          int fsampx = nint (sampx * freqFact);
          int fsampy = nint (sampy * freqFact);
          int fsupx  = nint (supx / freqFact);
          int fsupy  = nint (supy / freqFact);

          // Only use visibility point if the full support is within grid.
	  
	  //printf("offx=%i, offy=%i\n",offx,offy);
	  //assert(1==0);

          //if (locx-supx >= 0  &&  locx+supx < nGridX  &&
          //    locy-supy >= 0  &&  locy+supy < nGridY) {
          if (locx-supx >= 0  &&  locx+supx < nGridX) {

	    //printf("inside loc!");
            // Get pointer to data and flags for this channel.
            int doff = (irow * nVisChan + visChan) * nVisPol;
            const double complex* __restrict__ visPtr  = Complex_pyvector_to_Carrayptrs(vis)  + doff;
            int* __restrict__ flagPtr = pyvector_to_Carrayptrs2(flags) + doff;

            // Handle a visibility if not flagged.
	    int ipol;
            for (ipol=0; ipol<nVisPol; ++ipol) {

	      //printf("flag=%i [on pol %i]\n",flagPtr[ipol],ipol);

	      //printf(".. (row, chan, pol)=(%i, %i, %i): F=%i \n",inx,visChan,ipol,flagPtr[ipol]);
              if (flagPtr[ipol]==0) {
		//printf("take %i on pol %i\n",flagPtr[ipol],ipol);
		//printf("flag: %i",flagPtr[ipol]);
		double complex VisVal;
		if (dopsf==1) {
		  VisVal = 1.;
		}else{
		  VisVal =visPtr[ipol];
		}
		
		//printf(".. (row, chan, pol)=(%i, %i, %i), VisVal=(%f,%f) \n",inx,visChan,ipol,creal(VisVal),cimag(VisVal));
		
                // Map to grid polarization. Only use pol if needed.
                int gridPol = PolMap[ipol];//0;//polMap_p(ipol);
                if (gridPol >= 0  &&  gridPol < nGridPol) {

                  // Get the offset in the grid data array.
                  int goff = (gridChan*nGridPol + gridPol) * nGridX * nGridY;
                  // Loop over the scaled support.
		  int sy=0;
                  //for (sy=-fsupy; sy<=fsupy; ++sy) {
                    // Get the pointer in the grid for the first x in this y.
                    //double complex __restrict__ *gridPtr = grid.data() + goff + (locy+sy)*nGridX + locx-supx;
		    double complex *gridPtr = Complex_pyvector_to_Carrayptrs(grid) + goff + (locy+sy)*nGridX + locx-supx;
		    int ShiftOnGrid=(locy+sy)*nGridX + locx-supx;
		    // Fast version
                    const double complex* __restrict__ cf[1];
                    int cfoff = (offy + sy*fsampy)*nConvX + offx - fsupx*fsampx;
		    //printf("start off CF: (%3.3i,%3.3i) -> CFoff: %i \n",offx,offy,cfoff);
		    // //printf("(offy, sy,fsampy,nConvX,offx,fsupx,fsampx) = (%i, %i, %i, %i, %i, %i, %i) \n",offy, sy,fsampy,nConvX,offx,fsupx,fsampx);

		    //cf[0] = (*cfs.vdata)[CFChan][0][0].data() + cfoff;
 		    cf[0] = Complex_pyvector_to_Carrayptrs(cfs) + cfoff;
		    int sx;
                    for (sx=-fsupx; sx<=fsupx; ++sx) {
		      //printf("\n"); 
		      //printf("CF= (%f, %f) @(x,y)=(%3.3i,%3.3i) \n",creal(*cf[0]),cimag(*cf[0]),sx,sy);
		      //printf(".. Grid pol=%i, gridin=(%f, %f), goff=%i, ShiftOnGrid=%i\n",gridPol,creal(*gridPtr),cimag(*gridPtr),goff,ShiftOnGrid);
                      // Loop over polarizations to correct for leakage.
                      //complex polSum(0,0);
  		      //polSum *= *imgWtPtr;
		      //printf(".. Vis Chan=%i, pol=%i, VisVal=(%f,%f) ",visChan,ipol,creal(VisVal),cimag(VisVal));
                      //*gridPtr++ += VisVal * *cf[0];
                      *gridPtr += VisVal * *cf[0];
		      //printf(" ... gridout=(%f, %f) \n",creal(*gridPtr),cimag(*gridPtr));
                      *gridPtr++;
		      cf[0] += fsampx;

		      /* polSum += VisVal * *cf[0]; */
		      /* cf[0] += fsampx; */
  		      /* polSum *= *imgWtPtr; */
                      /* *gridPtr++ += polSum; */
                    }

		    //}
                  sumWtPtr[gridPol+gridChan*nGridPol] += 1;//*imgWtPtr;
                } // end if gridPol
              } // end if !flagPtr
            } // end for ipol
          } // end if ongrid
        } // end if gridChan
        //imgWtPtr++;
      } // end for visChan
    } // end for inx
  }



double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
  return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

int *pyvector_to_Carrayptrs2(PyArrayObject *arrayin)  {
  return (int *) arrayin->data;  /* pointer to arrayin data as double */
}


double complex *Complex_pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
  return (double complex *) arrayin->data;  /* pointer to arrayin data as double */
}


