/* A file to test imorting C modules for handling arrays to Python */

#include <Python.h>
#include "arrayobject.h"
#include "Gridder.h"
#include <math.h>
#include "complex.h"
#include <time.h>

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
static PyMethodDef _pyGridder_testMethods[] = {
	{"pyGridderWPol", pyGridderWPol, METH_VARARGS},
	{"pyGridderPoints", pyGridderPoints, METH_VARARGS},
	{"pyDeGridderWPol", pyDeGridderWPol, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_pyGridder()  {
	(void) Py_InitModule("_pyGridder", _pyGridder_testMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}




static PyObject *pyGridderPoints(PyObject *self, PyObject *args)
{
  PyObject *ObjGridIn,*ObjWIn;
  PyArrayObject *np_grid, *np_w, *x, *y, *w;
  double R;

  if (!PyArg_ParseTuple(args, "OO!O!Od", 
			&ObjGridIn,
			&PyArray_Type,  &x, 
			&PyArray_Type,  &y, 
			&ObjWIn,
			//&PyArray_Type,  &w,
			&R
			))  return NULL;
  np_grid = (PyArrayObject *) PyArray_ContiguousFromObject(ObjGridIn, PyArray_FLOAT64, 0, 4);
  np_w = (PyArrayObject *) PyArray_ContiguousFromObject(ObjWIn, PyArray_FLOAT64, 0, 4);


  int nx,ny,np;
  nx=np_grid->dimensions[0];
  ny=np_grid->dimensions[1];
  double* grid = p_float64(np_grid);
  double* wp= p_float64(np_w);

  
  int* xp=I_ptr(x);
  int* yp=I_ptr(y);
  np=x->dimensions[0];
  double sumw=0;
  
  //printf("grid dims (%i,%i)\n",nx,ny);
  //printf("nvis dims (%i)\n",np);

  int i=0;
  for (i=0; i<np; i++) {
    grid[xp[i]+nx*yp[i]]+=wp[i];
    sumw+=wp[i];
    //printf("(x,y)=(%i,%i)\n",xp[i],yp[i]);
  }


  double Wk;
  double sumWk=0;
  for (i=0; i<np; i++) {
    Wk=grid[xp[i]+nx*yp[i]];
    sumWk+=Wk;
  }

  double fact=  (sumw/sumWk)*pow(5.*pow(10.,-R),2.);
  //printf("fact=(%f)\n",fact);


  for (i=0; i<np; i++) {
    Wk=grid[xp[i]+nx*yp[i]];
    wp[i]/=(1.+fact*Wk);
  }

  return PyArray_Return(np_w);//,PyArray_Return(np_grid);

}







static PyObject *pyGridderWPol(PyObject *self, PyObject *args)
{
  PyObject *ObjGridIn;
  PyArrayObject *np_grid, *vis, *uvw, *cfs, *flags, *weights, *sumwt, *increment, *freqs,*WInfos;

  PyObject *Lcfs;
  PyObject *LJones,*Lmaps;
  PyObject *LcfsConj;
  int dopsf;

  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!iO!O!O!O!O!O!O!", 
			&ObjGridIn,
			&PyArray_Type,  &vis, 
			&PyArray_Type,  &uvw, 
			&PyArray_Type,  &flags, 
			&PyArray_Type,  &weights,
			&PyArray_Type,  &sumwt, 
			&dopsf, 
			&PyList_Type, &Lcfs,
			&PyList_Type, &LcfsConj,
			&PyArray_Type,  &WInfos,
			&PyArray_Type,  &increment,
			&PyArray_Type,  &freqs,
			&PyList_Type, &Lmaps,
			&PyList_Type, &LJones
			))  return NULL;
  int nx,ny,nz,nzz;
  //initTime();
  np_grid = (PyArrayObject *) PyArray_ContiguousFromObject(ObjGridIn, PyArray_COMPLEX64, 0, 4);
  //timeit("declare np_grid");
  /* nx=np_grid->dimensions[0]; */
  /* ny=np_grid->dimensions[1]; */
  /* nz=np_grid->dimensions[2]; */
  /* nzz=np_grid->dimensions[3]; */
  /* printf("grid dims (%i,%i,%i,%i)\n",nx,ny,nz,nzz); */
  /* nx=vis->dimensions[0]; */
  /* ny=vis->dimensions[1]; */
  /* nz=vis->dimensions[2]; */
  /* printf("vis  dims (%i,%i,%i)\n",nx,ny,nz); */
  /* /\* nx=uvw->dimensions[0]; *\/ */
  /* /\* ny=uvw->dimensions[1]; *\/ */
  /* /\* nz=uvw->dimensions[2]; *\/ */
  /* /\* printf("uvw  dims (%i,%i)\n",nx,ny); *\/ */

  
  /* //bool * visPtr  = p_bool(vis); */
  /* bool * flagPtr  = p_bool(flags); */
  /* /\* printf("VV= (%f,%f)\n",creal(*visPtr),cimag(*visPtr)); *\/ */
  /* /\* printf("VV= (%f,%f)\n",crealf(*visPtr),cimagf(*visPtr)); *\/ */

  /* int x,y,z; */
  /* for(x=0; x<nx; x++){ */
  /* for(y=0; y<ny; y++){ */
  /* for(z=0; z<nz; z++){ */
  /*   /\* printf("\n"); *\/ */
  /*   /\* printf("flag [%i,%i,%i]= (%i)\n",x,y,z,(int)*flagPtr); *\/ */
  /*   /\* flagPtr++; *\/ */

  /*   int doff = (x * ny + y) * nz; */
  /*   printf("flag [%i,%i,%i]= (%i)\n",x,y,z,(int)flagPtr[doff+z]); */
  /*   int truth; */
  /*   truth=((int)flagPtr[doff+z]==1); */
  /*   printf("equal1? %i",truth); */
  /* }}} */

  gridderWPol(np_grid, vis, uvw, flags, weights, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones);
  //timeit("grid");
  
  return PyArray_Return(np_grid);

}

double PI=3.14159265359;


void gridderWPol(PyArrayObject *grid,
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
		 PyObject *Lmaps, PyObject *LJones)
  {
    // Get size of convolution functions.
    PyArrayObject *cfs;
    PyArrayObject *NpPolMap;
    NpPolMap = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 0), PyArray_INT32, 0, 4);

    PyArrayObject *NpFacetInfos;
    NpFacetInfos = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 1), PyArray_FLOAT64, 0, 4);

    double* ptrFacetInfos=p_float64(NpFacetInfos);
    double Cu=ptrFacetInfos[0];
    double Cv=ptrFacetInfos[1];
    double l0=ptrFacetInfos[2];
    double m0=ptrFacetInfos[3];
    double n0=sqrt(1-l0*l0-m0*m0)-1;


    double VarTimeGrid=0;
    int Nop=0;

    int npolsMap=NpPolMap->dimensions[0];
    int* PolMap=I_ptr(NpPolMap);
    
    //    printf("npols=%i %i\n",npolsMap,PolMap[3]);

    // Get size of grid.
    double* ptrWinfo = p_float64(Winfos);
    double WaveRefWave = ptrWinfo[0];
    double wmax = ptrWinfo[1];
    double NwPlanes = ptrWinfo[2];
    int OverS=floor(ptrWinfo[3]);


    //    printf("WaveRef=%f, wmax=%f \n",WaveRefWave,wmax);
    int nGridX    = grid->dimensions[3];
    int nGridY    = grid->dimensions[2];
    int nGridPol  = grid->dimensions[1];
    int nGridChan = grid->dimensions[0];

    // Get visibility data size.
    int nVisPol   = flags->dimensions[2];
    int nVisChan  = flags->dimensions[1];
    int nrows     = uvw->dimensions[0];
    //    printf("(nrows, nVisChan, nVisPol)=(%i, %i, %i)\n",nrows,nVisChan,nVisPol);


    // Get oversampling and support size.
    int sampx = OverS;//int (cfs.sampling[0]);
    int sampy = OverS;//int (cfs.sampling[1]);

    double* __restrict__ sumWtPtr = p_float64(sumwt);//->data;
    double complex psfValues[4];
    psfValues[0] = psfValues[1] = psfValues[2] = psfValues[3] = 1;

    //uint inxRowWCorr(0);

    double offset_p[2],uvwScale_p[2];

    offset_p[0]=nGridX/2;//(nGridX-1)/2.;
    offset_p[1]=nGridY/2;
    float fnGridX=nGridX;
    float fnGridY=nGridY;
    double *incr=p_float64(increment);
    double *Pfreqs=p_float64(freqs);
    uvwScale_p[0]=fnGridX*incr[0];
    uvwScale_p[1]=fnGridX*incr[1];
    //printf("uvscale=(%f %f)\n",uvwScale_p[0],uvwScale_p[1]);
    double C=2.99792458e8;
    int inx;
    // Loop over all visibility rows to process.

    for (inx=0; inx<nrows; inx++) {
      int irow = inx;//rows[inx];
      //printf("\n");
      //printf("irow=%i/%i\n",irow,nrows);
      //const double*  __restrict__ uvwPtr   = GetDp(uvw) + irow*3;
      double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;
      double*   imgWtPtr = p_float64(weights) +
	                                  irow  * nVisChan;

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
	  cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);
	} else{
	  cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
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

	  // Change coordinate and shift visibility to facet center
	  double complex UVNorm=2.*I*PI/ThisWaveLength;
	  double U=uvwPtr[0];
	  double V=uvwPtr[1];
	  double W=uvwPtr[2];
	  double complex corr=cexp(-UVNorm*(U*l0+V*m0+W*n0));
	  U+=W*Cu;
	  V+=W*Cv;

	  //	  printf("uvw = (%f, %f, %f)\n",U,V,W);


          // Determine the grid position from the UV coordinates in wavelengths.
	  double posx,posy;

	  //For Even/Odd take the -1 off
	  posx = uvwScale_p[0] * U * recipWvl + offset_p[0];//#-1;
	  posy = uvwScale_p[1] * V * recipWvl + offset_p[1];//-1;

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
          // Scaling with frequency is not necessary (according to Cyril).
          double freqFact = 1;
          int fsampx = nint (sampx * freqFact);
          int fsampy = nint (sampy * freqFact);
          int fsupx  = nint (supx / freqFact);
          int fsupy  = nint (supy / freqFact);

          // Only use visibility point if the full support is within grid.
	  
	  //printf("offx=%i, offy=%i\n",offx,offy);
	  //assert(1==0);

          if (locx-supx >= 0  &&  locx+supx < nGridX  &&
              locy-supy >= 0  &&  locy+supy < nGridY) {

	    //printf("inside loc!");
            // Get pointer to data and flags for this channel.
            int doff = (irow * nVisChan + visChan) * nVisPol;

            const float complex* __restrict__ visPtr  = p_complex64(vis)  + doff;
	    
	    //	    printf("First value: (%f,%f)\n",creal(*visPtr),cimag(*visPtr));

            bool* __restrict__ flagPtr = p_bool(flags) + doff;

            // Handle a visibility if not flagged.
	    int ipol;
            for (ipol=0; ipol<nVisPol; ++ipol) {

	      //printf("flag=%i [on pol %i, doff=%i]\n",(int)flagPtr[ipol],ipol,doff);

	      //printf(".. (row, chan, pol)=(%i, %i, %i): F=%i \n",inx,visChan,ipol,flagPtr[ipol]);
              if (((int)flagPtr[ipol])==0) {
		//printf("take %i on pol %i\n",flagPtr[ipol],ipol);
		//printf("flag: %i",flagPtr[ipol]);
		double complex VisVal;
		if (dopsf==1) {
		  VisVal = 1.;
		}else{
		  VisVal =visPtr[ipol];
		}
		VisVal*=(*imgWtPtr);
		VisVal*=corr;
		//		printf(".. (row, chan, pol)=(%i, %i, %i), VisVal=(%f,%f) \n",inx,visChan,ipol,creal(VisVal),cimag(VisVal));
		//printf(" \n");
		//printf("Vis: %f %f \n",creal(VisVal),cimag(VisVal));
		
                // Map to grid polarization. Only use pol if needed.
                int gridPol = PolMap[ipol];//0;//polMap_p(ipol);
                if (gridPol >= 0  &&  gridPol < nGridPol) {

  		  //cout<<"ipol: "<<ipol<<endl;
                  // Get the offset in the grid data array.
                  int goff = (gridChan*nGridPol + gridPol) * nGridX * nGridY;
                  // Loop over the scaled support.
		  int sy;

		  //initTime();
                  for (sy=-fsupy; sy<=fsupy; ++sy) {
                    // Get the pointer in the grid for the first x in this y.
                    //double complex __restrict__ *gridPtr = grid.data() + goff + (locy+sy)*nGridX + locx-supx;
		    float complex *gridPtr = p_complex64(grid) + goff + (locy+sy)*nGridX + locx-supx;
		    // Fast version
                    const float complex* __restrict__ cf[1];
                    int cfoff = (offy + sy*fsampy)*nConvX + offx - fsupx*fsampx;
		    //printf("start off CF: (%3.3i,%3.3i) -> CFoff: %i \n",offx,offy,cfoff);
		    //printf("(offy, sy,fsampy,nConvX,offx,fsupx,fsampx) = (%i, %i, %i, %i, %i, %i, %i) \n",offy, sy,fsampy,nConvX,offx,fsupx,fsampx);

		    //cf[0] = (*cfs.vdata)[CFChan][0][0].data() + cfoff;
 		    cf[0] = p_complex64(cfs) + cfoff;
		    int sx;
                    for (sx=-fsupx; sx<=fsupx; ++sx) {
		      //printf("(%3.3i,%3.3i) CF=(%f, %f) \n",sx,sy,creal(*cf[0]),cimag(*cf[0])); 
                      // Loop over polarizations to correct for leakage.
                      //complex polSum(0,0);
  		      //polSum *= *imgWtPtr;
		      //printf(".. Chan=%i, gridin=(%f, %f), VisVal=(%f,%f) ",visChan,creal(*gridPtr),cimag(*gridPtr),creal(VisVal),cimag(VisVal));
                      *gridPtr++ += VisVal * *cf[0];// * *imgWtPtr;
		      //printf(" ... gridout=(%f, %f) \n",creal(*gridPtr),cimag(*gridPtr));
		      cf[0] += fsampx;
		      //Nop+=1;

		      /* polSum += VisVal * *cf[0]; */
		      /* cf[0] += fsampx; */
  		      /* polSum *= *imgWtPtr; */
                      /* *gridPtr++ += polSum; */
                    }

                  }
		  //VarTimeGrid+=AppendTimeit();
                  sumWtPtr[gridPol+gridChan*nGridPol] += *imgWtPtr;
		  /* if{*imgWtPtr>2.}{ */
		  //printf(" [%i,%i,%f] ",inx,visChan,*imgWtPtr);
		  /* } */
                } // end if gridPol
              } // end if !flagPtr
            } // end for ipol
          } // end if ongrid
        } // end if gridChan
        imgWtPtr++;
      } // end for visChan
    } // end for inx
    //    printf(" timegrid %f %i \n",VarTimeGrid,Nop);
  }




////////////////////

static PyObject *pyDeGridderWPol(PyObject *self, PyObject *args)
{
  PyObject *ObjGridIn;
  PyObject *ObjVis;
  PyArrayObject *np_grid, *np_vis, *uvw, *cfs, *flags, *sumwt, *increment, *freqs,*WInfos;

  PyObject *Lcfs;
  PyObject *Lmaps;
  PyObject *LcfsConj;
  int dopsf;

  if (!PyArg_ParseTuple(args, "O!OO!O!O!iO!O!O!O!O!O!", 
			//&ObjGridIn,
			&PyArray_Type,  &np_grid,
			&ObjVis,//&PyArray_Type,  &vis, 
			&PyArray_Type,  &uvw, 
			&PyArray_Type,  &flags, 
			//&PyArray_Type,  &rows, 
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

  np_vis = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVis, PyArray_COMPLEX64, 0, 3);

  
  /* int nVisRow  = np_vis->dimensions[0]; */
  /* int nVisChan = np_vis->dimensions[1]; */
  /* int nVisPol  = np_vis->dimensions[2]; */
  /* int ix,iy,iz; */
  /* for(ix=0; ix<nVisRow; ix++){ */
  /*   for(iy=0; iy<nVisChan; iy++){ */
  /*     int doff = (ix * nVisChan + iy) * nVisPol; */
  /*     double complex* visPtr  = Complex_pyvector_to_Carrayptrs(np_vis)  + doff; */
  /*     for(iz=0; iz<nVisPol; iz++){ */
  /* 	//double complex Vis=visPtr[iz]; */
  /* 	printf("[%i,%i,%i] (%f , %f)      ",ix,iy,iz,creal(visPtr[iz]),cimag(visPtr[iz])); */
  /* 	visPtr[iz]=1.+(1.*I); */
  /* 	printf("[%i,%i,%i] (%f , %f)\n",ix,iy,iz,creal(visPtr[iz]),cimag(visPtr[iz])); */
  /*     }  */
  /*   }  */
  /* }  */


  DeGridderWPol(np_grid, np_vis, uvw, flags, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps);
  
  //return PyArray_Return(np_vis);

  return Py_None;

}





void DeGridderWPol(PyArrayObject *grid,
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
		   PyObject *Lmaps)
  {
    // Get size of convolution functions.
    PyArrayObject *cfs;
    PyArrayObject *NpPolMap;
    NpPolMap = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 0), PyArray_INT32, 0, 4);
    int npolsMap=NpPolMap->dimensions[0];
    int* PolMap=I_ptr(NpPolMap);
    
    PyArrayObject *NpFacetInfos;
    NpFacetInfos = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 1), PyArray_FLOAT64, 0, 4);

    PyArrayObject *NpRows;
    NpRows = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 2), PyArray_INT32, 0, 4);
    int* ptrRows=I_ptr(NpRows);
    int row0=ptrRows[0];
    int row1=ptrRows[1];
    
    double VarTimeDeGrid=0;
    int Nop=0;

    double* ptrFacetInfos=p_float64(NpFacetInfos);
    double Cu=ptrFacetInfos[0];
    double Cv=ptrFacetInfos[1];
    double l0=ptrFacetInfos[2];
    double m0=ptrFacetInfos[3];
    double n0=sqrt(1-l0*l0-m0*m0)-1;


    //printf("npols=%i %i\n",npolsMap,PolMap[3]);

    // Get size of grid.
    double* ptrWinfo = p_float64(Winfos);
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
    
    double* __restrict__ sumWtPtr = p_float64(sumwt);//->data;
    double complex psfValues[4];
    psfValues[0] = psfValues[1] = psfValues[2] = psfValues[3] = 1;

    //uint inxRowWCorr(0);

    double offset_p[2],uvwScale_p[2];

    offset_p[0]=nGridX/2;//(nGridX-1)/2.;
    offset_p[1]=nGridY/2;
    float fnGridX=nGridX;
    float fnGridY=nGridY;
    double *incr=p_float64(increment);
    double *Pfreqs=p_float64(freqs);
    uvwScale_p[0]=fnGridX*incr[0];
    uvwScale_p[1]=fnGridX*incr[1];
    //printf("uvscale=(%f %f)",uvwScale_p[0],uvwScale_p[1]);
    double C=2.99792458e8;
    int inx;


    double posx,posy;


    // Loop over all visibility rows to process.
    for (inx=row0; inx<row1; ++inx) {
      int irow = inx;

      //printf("row=%i/%i \n",irow,nrows);



      double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;
      // Loop over all channels in the visibility data.
      // Map the visibility channel to the grid channel.
      // Skip channel if data are not needed.
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
	  cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);
	} else{
	  cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
	}
	int nConvX = cfs->dimensions[0];
	int nConvY = cfs->dimensions[1];
	int supx = (nConvX/OverS-1)/2;
	int supy = (nConvY/OverS-1)/2;


	//cout<<"  chan="<<visChan<<"taking CF="<<CFChan<<endl;

	// !! dirty trick to select all channels
	//chanMap_p[visChan]=0;

        if (gridChan >= 0  &&  gridChan < nGridChan) {
          // Determine the grid position from the UV coordinates in wavelengths.

	  // Change coordinate and shift visibility to facet center
	  double complex UVNorm=2.*I*PI/ThisWaveLength;
	  double U=uvwPtr[0];
	  double V=uvwPtr[1];
	  double W=uvwPtr[2];
	  double complex corr=cexp(UVNorm*(U*l0+V*m0+W*n0));
	  U+=W*Cu;
	  V+=W*Cv;
	  //printf("uvw = (%f,%f,%f)\n",U,V,W);

	  double recipWvl = Pfreqs[visChan] / C;
	  //cout<<"vbs.freq_p[visChan]" <<vbs.freq_p[visChan] <<endl;
	  posx = uvwScale_p[0] * U * recipWvl + offset_p[0];//#-1;
	  posy = uvwScale_p[1] * V * recipWvl + offset_p[1];//-1;

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
          // Scaling with frequency is not necessary (according to Cyril).
          double freqFact = 1;
          int fsampx = nint (sampx * freqFact);
          int fsampy = nint (sampy * freqFact);
          int fsupx  = nint (supx / freqFact);
          int fsupy  = nint (supy / freqFact);

	  //



	  /* Weights_Lin_Interp[0]=(1.-diffx)*(1.-diffy); */
	  /* Weights_Lin_Interp[1]=(1.-diffx)*diffy; */
	  /* Weights_Lin_Interp[2]=diffx*(1.-diffy); */
	  /* Weights_Lin_Interp[3]=diffx*diffy; */


          // Only use visibility point if the full support is within grid.
          if (locx-supx >= 0  &&  locx+supx < nGridX  &&
              locy-supy >= 0  &&  locy+supy < nGridY) {
            ///            cout << "in grid"<<endl;
            // Get pointer to data and flags for this channel.
            int doff = (irow * nVisChan + visChan) * nVisPol;
            float complex* __restrict__ visPtr  = p_complex64(vis)  + doff;
            bool* __restrict__ flagPtr = p_bool(flags) + doff;

	    int ipol;

            // Handle a visibility if not flagged.
            /* for (ipol=0; ipol<nVisPol; ++ipol) { */
            /*   if (! flagPtr[ipol]) { */
	    /* 	visPtr[ipol] = Complex(0,0); */
            /*   } */
            /* } */

	    //for (Int w=0; w<4; ++w) {
	    //  Double weight_interp(Weights_Lin_Interp[w]);
            for (ipol=0; ipol<nVisPol; ++ipol) {
              if (((int)flagPtr[ipol])==0) {
                // Map to grid polarization. Only use pol if needed.
                int gridPol = PolMap[ipol];
                if (gridPol >= 0  &&  gridPol < nGridPol) {
                  /// Complex norm(0,0);
                  // Get the offset in the grid data array.

                  int goff = (gridChan*nGridPol + gridPol) * nGridX * nGridY;
                  // Loop over the scaled support.
		  int sy;
		  //initTime();
                  for (sy=-fsupy; sy<=fsupy; ++sy) {
                    // Get the pointer in the grid for the first x in this y.
		    float complex *gridPtr = p_complex64(grid) + goff + (locy+sy)*nGridX + locx-supx;
		    // Fast version
                    const float complex* __restrict__ cf[1];
                    int cfoff = (offy + sy*fsampy)*nConvX + offx - fsupx*fsampx;

                    // Get pointers to the first element to use in the 4
                    // convolution functions for this channel,pol.

		    // fast version


 		    cf[0] = p_complex64(cfs) + cfoff;
		    int sx;
                    for (sx=-fsupx; sx<=fsupx; ++sx) {
		      //outFile<<irow <<" "<<ipol<<" "<<posx<<" "<<posy<<" "<<(offx+ sx*fsampx-(nConvX-1.)/2.)/float(fsampx)<<" "<<(offy + sy*fsampy-(nConvX-1)/2.)/float(fsampy)
		      //<<" "<<real(*gridPtr * *cf[0])<<" "<<imag(*gridPtr * *cf[0])<<" "<<real(*gridPtr)<<" "<<imag(*gridPtr)<<endl;
		      visPtr[ipol] += *gridPtr  * *cf[0] *corr;;//* factor;
		      cf[0] += fsampx;
                      gridPtr++;
		      //Nop+=1;
		      //printf("(%f, %f)\n",creal(visPtr[ipol]),cimag(visPtr[ipol]));
                    }

		    // // Full version
                    // const Complex* __restrict__ cf[4];
                    // Int cfoff = (offy + sy*fsampy)*nConvX + offx - fsupx*fsampx;
                    // for (int i=0; i<4; ++i) {
                    //   cf[i] = (*cfs.vdata)[gridChan][i][ipol].data() + cfoff;
                    // }
                    // for (Int sx=-fsupx; sx<=fsupx; ++sx) {
                    //   for (Int i=0; i<nVisPol; ++i) {
                    //     visPtr[i] += *gridPtr * *cf[i];
                    //     cf[i] += fsampx;
                    //   }
                    //   gridPtr++;
                    // }

                  }
		  //VarTimeDeGrid+=AppendTimeit();
                } // end if gridPol
              } // end if !flagPtr

	      //visPtr[ipol]*=corr;
            } // end for ipol
          } // end if ongrid
        } // end if gridChan
	//}
      } // end for visChan
    } // end for inx
    //assert(false);
    //printf(" timedegrid %f %i\n",VarTimeDeGrid,Nop);
  }
