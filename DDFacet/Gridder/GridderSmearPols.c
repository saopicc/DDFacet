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
#include <stdlib.h>
#include <stdint.h>
#include "arrayobject.h"
#include "GridderSmearPols.h"
#include "complex.h"
#include "Stokes.h"
#include <omp.h>
#include "Tools.h"
#include "JonesServer.c"
//#include <fcntl.h>           /* For O_* constants */
#include "Semaphores.h"



clock_t start;

void initTime(){start=clock();}

void timeit(char* Name){
  clock_t diff;
  diff = clock() - start;
  start=clock();
  float msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("%s: %f\n",Name,msec);
}



/* double AppendTimeit(){ */
/*   clock_t diff; */
/*   diff = clock() - start; */
/*   double msec = diff * 1000000 / CLOCKS_PER_SEC; */
/*   return msec; */
/* } */

void AddTimeit(struct timespec PreviousTime, long int *aTime){
  long int t0 = PreviousTime.tv_nsec+PreviousTime.tv_sec*1000000000;
  clock_gettime(CLOCK_MONOTONIC_RAW, &PreviousTime);
  (*aTime)+=(PreviousTime.tv_nsec+PreviousTime.tv_sec*1000000000-t0);
}



/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef _pyGridderSmearPols_testMethods[] = {
	{"pyGridderWPol", pyGridderWPol, METH_VARARGS},
	{"pyDeGridderWPol", pyDeGridderWPol, METH_VARARGS},
	{"pyTestMatrix", pyTestMatrix, METH_VARARGS},
	{"pySetSemaphores", pySetSemaphores, METH_VARARGS},
	{"pyDeleteSemaphore", pyDeleteSemaphore, METH_VARARGS},
	{"pyAccumulateWeightsOntoGrid", pyAccumulateWeightsOntoGrid, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_pyGridderSmearPols()  {
  (void) Py_InitModule("_pyGridderSmearPols", _pyGridderSmearPols_testMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}


static PyObject *pyAccumulateWeightsOntoGrid(PyObject *self, PyObject *args)
{
    PyArrayObject *grid, *weights, *index;

    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type,  &grid,
            &PyArray_Type,  &weights,
            &PyArray_Type,  &index
            ))
        return NULL;

    double * pgrid      = p_float64(grid);
    float * pweights    = p_float32(weights);
    long int * pindex   = p_int64(index);
    size_t n = weights->dimensions[0];
    size_t i;

    for( i=0; i<n; i++)
    {
        size_t igrid = pindex[i];
        float w = pweights[i];
        if( w!=0 )
        {
            sem_t * psem = GiveSemaphoreFromCell(igrid);
            sem_wait(psem);
            pgrid[igrid] += w;
            sem_post(psem);
        }
    }

    Py_INCREF(Py_None);
    return Py_None;

}

//////////////////////////////////////////////////////////////////////
#define READ_4CORR \
  VisMeas[0]=visPtrMeas[0];\
  VisMeas[1]=visPtrMeas[1];\
  VisMeas[2]=visPtrMeas[2];\
  VisMeas[3]=visPtrMeas[3];
  
#define READ_2CORR_PAD \
  VisMeas[0]=visPtrMeas[0];\
  VisMeas[1]=0;\
  VisMeas[2]=0;\
  VisMeas[3]=visPtrMeas[1]; 

#define MULACCUM_4CORR \
  Vis[0] += VisMeas[0]*Weight;\
  Vis[1] += VisMeas[1]*Weight;\
  Vis[2] += VisMeas[2]*Weight;\
  Vis[3] += VisMeas[3]*Weight;
  
#define MULACCUM_2CORR_UNPAD \
  Vis[0] += VisMeas[0]*Weight;\
  Vis[1] += VisMeas[3]*Weight;
  
#define gridder_factory(griddername, stokesconversion, readcorrs, savecorrs) \
void griddername(PyArrayObject *grid, \
		 PyArrayObject *vis, \
		 PyArrayObject *uvw, \
		 PyArrayObject *flags, \
		 PyArrayObject *weights, \
		 PyArrayObject *sumwt, \
		 int dopsf, \
		 PyObject *Lcfs, \
		 PyObject *LcfsConj, \
		 PyArrayObject *Winfos, \
		 PyArrayObject *increment, \
		 PyArrayObject *freqs, \
		 PyObject *Lmaps, \
		 PyObject *LJones, \
		 PyArrayObject *SmearMapping, \
		 PyArrayObject *Sparsification, \
		 PyObject *LOptimisation, \
		 PyObject *LSmearing, \
		 PyArrayObject *np_ChanMapping \
		 ) \
  { \
    /* Get size of convolution functions. */\
    int nrows     = uvw->dimensions[0]; \
    PyArrayObject *cfs; \
    PyArrayObject *NpPolMap; \
    NpPolMap = (PyArrayObject *) PyList_GetItem(Lmaps, 0); \
    \
    PyArrayObject *NpFacetInfos; \
    NpFacetInfos = (PyArrayObject *) PyList_GetItem(Lmaps, 1); \
    \
    \
    /*///////////////////////////////////////*/\
    int LengthSmearingList=PyList_Size(LSmearing);\
    float DT,Dnu,lmin_decorr,mmin_decorr;\
    double* uvw_dt_Ptr;\
    int DoSmearTime,DoSmearFreq;\
    int DoDecorr=(LengthSmearingList>0);\
    \
    int *p_ChanMapping=p_int32(np_ChanMapping);\
    \
    if(DoDecorr){\
      uvw_dt_Ptr = p_float64((PyArrayObject *) PyList_GetItem(LSmearing, 0));\
      \
      PyObject *_FDT= PyList_GetItem(LSmearing, 1);\
      DT=(float) (PyFloat_AsDouble(_FDT));\
      PyObject *_FDnu= PyList_GetItem(LSmearing, 2);\
      Dnu=(float) (PyFloat_AsDouble(_FDnu));\
      \
      PyObject *_DoSmearTime= PyList_GetItem(LSmearing, 3);\
      DoSmearTime=(int) (PyFloat_AsDouble(_DoSmearTime));\
      \
      PyObject *_DoSmearFreq= PyList_GetItem(LSmearing, 4);\
      DoSmearFreq=(int) (PyFloat_AsDouble(_DoSmearFreq));\
      \
      PyObject *_Flmin_decorr= PyList_GetItem(LSmearing, 5);\
      lmin_decorr=(float) (PyFloat_AsDouble(_Flmin_decorr));\
      PyObject *_Fmmin_decorr= PyList_GetItem(LSmearing, 6);\
      mmin_decorr=(float) (PyFloat_AsDouble(_Fmmin_decorr));\
      \
      \
    }\
    \
    double* ptrFacetInfos=p_float64(NpFacetInfos);\
    double Cu=ptrFacetInfos[0];\
    double Cv=ptrFacetInfos[1];\
    double l0=ptrFacetInfos[2];\
    double m0=ptrFacetInfos[3];\
    double n0=sqrt(1-l0*l0-m0*m0)-1;\
    \
    \
    double VarTimeGrid=0;\
    int Nop=0;\
    \
    int npolsMap=NpPolMap->dimensions[0];\
    /** PolMap=I_ptr(NpPolMap);*/\
    \
    /* Get size of grid. */\
    double* ptrWinfo = p_float64(Winfos);\
    double WaveRefWave = ptrWinfo[0];\
    double wmax = ptrWinfo[1];\
    double NwPlanes = ptrWinfo[2];\
    int OverS=floor(ptrWinfo[3]);\
    \
    int nGridX    = grid->dimensions[3];\
    int nGridY    = grid->dimensions[2];\
    int nGridPol  = grid->dimensions[1];\
    int nGridChan = grid->dimensions[0];\
    \
    /* Get visibility data size. */\
    int nVisCorr   = flags->dimensions[2];\
    int nVisChan  = flags->dimensions[1];\
    \
    /* Get oversampling and support size.*/\
    int sampx = OverS;/*int (cfs.sampling[0]);*/\
    int sampy = OverS;/*int (cfs.sampling[1]);*/\
    \
    double* __restrict__ sumWtPtr = p_float64(sumwt);/*/->data;*/\
    double complex psfValues[4];\
    psfValues[0] = psfValues[1] = psfValues[2] = psfValues[3] = 1;\
    \
    /*uint inxRowWCorr(0);*/\
    \
    double offset_p[2],uvwScale_p[2];\
    \
    offset_p[0]=nGridX/2;/*(nGridX-1)/2.;*/\
    offset_p[1]=nGridY/2;\
    float fnGridX=nGridX;\
    float fnGridY=nGridY;\
    double *incr=p_float64(increment);\
    double *Pfreqs=p_float64(freqs);\
    uvwScale_p[0]=fnGridX*incr[0];\
    uvwScale_p[1]=fnGridX*incr[1];\
    double C=2.99792458e8;\
    int inx;\
    /* Loop over all visibility rows to process. */\
    \
    /* ################### Prepare full scalar mode */\
    \
    PyObject *_JonesType  = PyList_GetItem(LOptimisation, 0);\
    int JonesType=(int) PyFloat_AsDouble(_JonesType);\
    PyObject *_ChanEquidistant  = PyList_GetItem(LOptimisation, 1);\
    int ChanEquidistant=(int) PyFloat_AsDouble(_ChanEquidistant);\
    \
    PyObject *_SkyType  = PyList_GetItem(LOptimisation, 2);\
    int SkyType=(int) PyFloat_AsDouble(_SkyType);\
    \
    PyObject *_PolMode  = PyList_GetItem(LOptimisation, 3);\
    int PolMode=(int) PyFloat_AsDouble(_PolMode);\
    int ipol;\
    \
    int * MappingBlock = p_int32(SmearMapping);\
    /* total size is in two words */\
    size_t NTotBlocks = MappingBlock[1];\
    NTotBlocks <<= 32;\
    NTotBlocks += MappingBlock[0];\
    int * NRowBlocks = MappingBlock+2;\
    int * StartRow = MappingBlock+2+NTotBlocks;\
    size_t iBlock;\
    \
    /* in sparsification mode, the Sparsification argument is an array of length NTotBlocks flags. */\
    /* Only blocks with a True flag will be gridded. */\
    const bool *sparsificationFlag = 0;\
    if( PyArray_Size((PyObject*)Sparsification) ){\
        if( PyArray_Size((PyObject*)Sparsification) != NTotBlocks ) {\
            PyErr_SetString(PyExc_TypeError, "sparsification argument must be an array of length NTotBlocks");\
            return;\
         }\
        sparsificationFlag = p_bool(Sparsification);\
    }\
    \
    int NMaxRow=0;\
    for(iBlock=0; iBlock<NTotBlocks; iBlock++){\
      if( sparsificationFlag && !sparsificationFlag[iBlock] )\
            continue;\
      int NRowThisBlock=NRowBlocks[iBlock]-2;\
      if(NRowThisBlock>NMaxRow){\
	NMaxRow=NRowThisBlock;\
      }\
    }\
    /* these are used for equidistant channels: one holds the phase term in channel 0, */\
    /* the other one holds the delta-phase across channels */\
    float complex *CurrentCorrTerm = calloc(1,(NMaxRow)*sizeof(float complex));\
    float complex *dCorrTerm = calloc(1,(NMaxRow)*sizeof(float complex));\
    /* and this indicates for which channel the CurrentCorrTerm is currently computed */\
    int * CurrentCorrChan = calloc(1,(NMaxRow)*sizeof(int));\
    int CurrentCorrRow0 = -1;\
    /* ######################################################## */\
    \
    double WaveLengthMean=0.;\
    double FreqMean0=0.;\
    \
    size_t visChan;\
    \
    float factorFreq=1;/*GiveFreqStep();*/\
    /*printf("factorFreq %f\n",factorFreq);*/\
    \
    for (visChan=0; visChan<nVisChan; ++visChan){\
      WaveLengthMean+=C/Pfreqs[visChan];\
      FreqMean0+=Pfreqs[visChan];\
    }\
    WaveLengthMean/=nVisChan;\
    FreqMean0/=nVisChan;\
    float FracFreqWidth=0;\
    if (nVisChan>1){\
      float DeltaFreq=(Pfreqs[nVisChan-1]-Pfreqs[0]);\
      FracFreqWidth=DeltaFreq/FreqMean0;\
    }\
    \
    /*////////////////////////////////////////////////////////////////////////////*/\
    initJonesServer(LJones,JonesType,WaveLengthMean);\
    /*////////////////////////////////////////////////////////////////////////////*/\
    \
    long int TimeShift[1]={0};\
    long int TimeApplyJones[1]={0};\
    long int TimeAverage[1]={0};\
    long int TimeJones[1]={0};\
    long int TimeGrid[1]={0};\
    long int TimeGetJones[1]={0};\
    long int TimeStuff[1]={0};\
    struct timespec PreviousTime;\
    \
    float complex Vis[4];\
    float complex VisMeas[4];\
    int ThisPol;\
    \
    float *ThisSumJonesChan=calloc(1,(nVisChan)*sizeof(float));\
    float *ThisSumSqWeightsChan=calloc(1,(nVisChan)*sizeof(float));\
    \
    for(iBlock=0; iBlock<NTotBlocks; iBlock++){\
      \
      int NRowThisBlock=NRowBlocks[iBlock]-2;\
      int chStart = StartRow[0];\
      int chEnd = StartRow[1];\
      int *Row = StartRow+2;\
      /* advance pointer to next blocklist*/\
      StartRow += NRowBlocks[iBlock];\
      if( sparsificationFlag && !sparsificationFlag[iBlock] )\
	continue;\
      \
      double Umean=0;\
      double Vmean=0;\
      double Wmean=0;\
      double FreqMean=0;\
      int NVisThisblock=0;\
      for(ThisPol =0; ThisPol<4;ThisPol++){\
	Vis[ThisPol]=0;\
	VisMeas[ThisPol]=0;\
      }\
      \
      double ThisWeight=0.;\
      float ThisSumJones=0.;\
      float ThisSumSqWeights=0.;\
      for(visChan=0; visChan<nVisChan; visChan++){\
	ThisSumJonesChan[visChan]=0;\
	ThisSumSqWeightsChan[visChan]=0;\
      }\
      \
      /* when moving to a new block of rows, init this to -1 so the code below knows to initialize*/\
      /* CurrentCorrTerm when the first channel of each row comes in*/\
      \
      if( Row[0] != CurrentCorrRow0 )\
      {\
	for (inx=0; inx<NRowThisBlock; inx++)\
	    CurrentCorrChan[inx] = -1;\
	CurrentCorrRow0 = Row[0];\
      }\
      double visChanMean=0.;\
      resetJonesServerCounter();\
      \
      for (inx=0; inx<NRowThisBlock; inx++) {\
	size_t irow = Row[inx];\
	if(irow>nrows){continue;}\
	double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;\
	WeightVaryJJ=1.;\
	\
	float DeCorrFactor=1.;\
	if(DoDecorr){\
	  int iRowMeanThisBlock=Row[NRowThisBlock/2];\
	  \
	  double*  __restrict__ uvwPtrMidRow   = p_float64(uvw) + iRowMeanThisBlock*3;\
	  double*  __restrict__ uvw_dt_PtrMidRow   = uvw_dt_Ptr + iRowMeanThisBlock*3;\
	  \
	  DeCorrFactor=GiveDecorrelationFactor(DoSmearFreq,DoSmearTime,\
					       (float)lmin_decorr, (float)mmin_decorr,\
					       uvwPtrMidRow,\
					       uvw_dt_PtrMidRow,\
					       (float)FreqMean0,\
					       (float)Dnu, \
					       (float)DT);\
	}\
	\
	for (visChan=chStart; visChan<chEnd; ++visChan) {\
	  size_t doff = (irow * nVisChan + visChan) * nVisCorr;\
	  bool* __restrict__ flagPtr = p_bool(flags) + doff;\
	  float*   imgWtPtr = p_float32(weights) + irow  * nVisChan + visChan;\
	  \
	  /* We can do that since all flags in 4-pols are equalised in ClassVisServer */\
	  if(flagPtr[0]==1){continue;}\
	  \
	  /*###################### Facetting #######################*/\
	  /* Change coordinate and shift visibility to facet center*/\
	  float U=(float)uvwPtr[0];\
	  float V=(float)uvwPtr[1];\
	  float W=(float)uvwPtr[2];\
	  /*AddTimeit(PreviousTime,TimeShift);*/\
	  /*#######################################################*/\
	  \
	  float complex corr;\
	  if(ChanEquidistant){\
	      /* init correlation term for first channel that it's not initialized in */\
	      if( CurrentCorrChan[inx] == -1 )\
	      {\
		float complex dotprod = -2.*I*PI*(U*l0+V*m0+W*n0)/C;\
		CurrentCorrTerm[inx] = cexp(Pfreqs[visChan]*dotprod);\
		dCorrTerm[inx]       = cexp((Pfreqs[1]-Pfreqs[0])*dotprod);\
		CurrentCorrChan[inx] = visChan;\
	      }\
	      /* else, wind the correlation term forward by as many channels as necessary */\
	      /* this modification allows us to support blocks that skip across channels */\
	      else\
	      {\
		while( CurrentCorrChan[inx] < visChan )\
		{\
		  CurrentCorrTerm[inx] *= dCorrTerm[inx];\
		  CurrentCorrChan[inx]++;\
		}\
	      }\
	      corr = CurrentCorrTerm[inx];\
	  }\
	  else{\
	      float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;\
	      corr=cexp(-UVNorm*(U*l0+V*m0+W*n0));\
	  }/* Not chan-equidistant*/\
	  \
	  int OneFlagged=0;\
	  int cond;\
	  \
	  if(DoApplyJones){\
	    updateJones(irow, visChan, uvwPtr, 1, 1);\
	  } /*endif DoApplyJones*/\
	  \
	  /*ThisBlockAllFlagged=0;*/\
	  /*AddTimeit(PreviousTime,TimeStuff);*/\
	  \
	  float complex* __restrict__ visPtrMeas  = p_complex64(vis)  + doff;\
	  \
	  if (dopsf==1) {\
	    VisMeas[0]= 1.;\
	    VisMeas[1]= 0.;\
	    VisMeas[2]= 0.;\
	    VisMeas[3]= 1.;\
	    corr=1.;\
	    if(DoApplyJones){\
	      /* first product seems superfluous, why multiply by identity? */\
	      MatDot(J0,JonesType,VisMeas,SkyType,VisMeas);\
	      MatDot(VisMeas,SkyType,J1H,JonesType,VisMeas);\
	    }\
	    if(DoDecorr){\
	      for(ThisPol =0; ThisPol<4;ThisPol++)\
		VisMeas[ThisPol]*=DeCorrFactor;\
	    }\
	  }else{\
	    readcorrs \
	  }\
	  float FWeight = (*imgWtPtr)*WeightVaryJJ; /**WeightVaryJJ;*/\
	  float complex Weight=(FWeight) * corr;\
	  float complex visPtr[4];\
	  if(DoApplyJones){\
	    MatDot(J0H,JonesType,VisMeas,SkyType,visPtr);\
	    MatDot(visPtr,SkyType,J1,JonesType,VisMeas);\
	    savecorrs \
	    \
	    /*Compute per channel and overall approximate matrix sqroot:*/\
	    float FWeightDecorr= FWeight*DeCorrFactor*DeCorrFactor; \
	    ThisSumJones+=BB*FWeightDecorr; \
	    ThisSumSqWeights+=FWeightDecorr;\
	    \
	    ThisSumJonesChan[visChan]+=BB*FWeightDecorr;\
	    ThisSumSqWeightsChan[visChan]+=FWeightDecorr;\
	  }else{\
	    savecorrs \
	  };/* Don't apply Jones*/\
	  \
	  U+=W*Cu;\
	  V+=W*Cv;\
	  /*###################### Averaging #######################*/\
	  Umean+=U;\
	  Vmean+=V;\
	  Wmean+=W;\
	  FreqMean+=factorFreq*(float)Pfreqs[visChan];\
	  ThisWeight+=(FWeight);\
	  \
	  visChanMean+=p_ChanMapping[visChan];\
	  \
	  NVisThisblock+=1.;/*(*imgWtPtr);*/\
	}/*endfor vischan*/\
      }/*endfor RowThisBlock*/\
      if(NVisThisblock==0){continue;}\
      Umean/=NVisThisblock;\
      Vmean/=NVisThisblock;\
      Wmean/=NVisThisblock;\
      FreqMean/=NVisThisblock;\
      \
      /*printf("visChanMean, NVisThisblock: %f %f\n",(float)visChanMean, (float)NVisThisblock);*/\
      visChanMean/=NVisThisblock;\
      int ThisGridChan=p_ChanMapping[chStart];\
      double diffChan=visChanMean-ThisGridChan;\
      if(fabs(diffChan)>1e-6)\
      {\
        printf("gridder: probably there is a problem in the BDA mapping: (ChanMean, ThisGridChan, diff)=(%lf, %i, %lf)\n",visChanMean,ThisGridChan,diffChan);\
        for (visChan=chStart; visChan<chEnd; ++visChan)\
            printf("%d ", ThisGridChan-p_ChanMapping[visChan]);\
        printf("\n");\
      }\
      \
      visChanMean=0.;\
      /* ################################################ */\
      /* ######## Convert correlations to stokes ######## */\
      stokesconversion\
      \
      /* ################################################ */\
      /* ############## Start Gridding visibility ####### */\
      int gridChan = p_ChanMapping[chStart];/*0;//chanMap_p[visChan];*/\
      \
      int CFChan = 0;/*ChanCFMap[visChan];*/\
      double recipWvl = FreqMean / C;\
      double ThisWaveLength=C/FreqMean;\
      \
      /* ############## W-projection #################### */\
      double wcoord=Wmean;\
      int iwplane = floor((NwPlanes-1)*fabs(wcoord)*(WaveRefWave/ThisWaveLength)/wmax+0.5);\
      int skipW=0;\
      if(iwplane>NwPlanes-1){\
	skipW=1;\
/*	printf("SIP\n");*/\
	continue;\
      };\
      \
      if(wcoord>0){\
      	cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);\
      } else{\
      	cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);\
      }\
      int nConvX = cfs->dimensions[0];\
      int nConvY = cfs->dimensions[1];\
      int supx = (nConvX/OverS-1)/2;\
      int supy = (nConvY/OverS-1)/2;\
      int SupportCF=nConvX/OverS;\
      /* ################################################ */\
      \
      if (gridChan >= 0  &&  gridChan < nGridChan) {\
      	double posx,posy;\
      	/*For Even/Odd take the -1 off*/\
      	posx = uvwScale_p[0] * Umean * recipWvl + offset_p[0];/*#-1;*/\
      	posy = uvwScale_p[1] * Vmean * recipWvl + offset_p[1];/*-1;*/\
      \
      	int locx = nint (posx);    /* location in grid*/\
      	int locy = nint (posy);\
      	/*printf("locx=%i, locy=%i\n",locx,locy);*/\
      	double diffx = locx - posx;\
      	double diffy = locy - posy;\
      	/*printf("diffx=%f, diffy=%f\n",diffx,diffy);*/\
      \
      	int offx = nint (diffx * sampx); /* location in*/\
      	int offy = nint (diffy * sampy); /* oversampling*/\
      	/*printf("offx=%i, offy=%i\n",offx,offy);*/\
      	offx += (nConvX-1)/2;\
      	offy += (nConvY-1)/2;\
      	/* Scaling with frequency is not necessary (according to Cyril). */\
      	double freqFact = 1;\
      	int fsampx = nint (sampx * freqFact);\
      	int fsampy = nint (sampy * freqFact);\
      	int fsupx  = nint (supx / freqFact);\
      	int fsupy  = nint (supy / freqFact);\
      \
      	/* Only use visibility point if the full support is within grid. */\
      \
      	/*printf("offx=%i, offy=%i\n",offx,offy);*/\
      	/*assert(1==0);*/\
      \
      	if (locx-supx >= 0  &&  locx+supx < nGridX  &&\
      	    locy-supy >= 0  &&  locy+supy < nGridY) {\
      \
      	  int ipol;\
      	  for (ipol=0; ipol<nVisPol; ++ipol) {\
      	    float complex VisVal;\
      	    /* if (dopsf==1) { */\
      	    /*   VisVal = 1.; */\
      	    /* }else{ */\
      	    /*   VisVal =stokes_vis[ipol]; */\
      	    /* } */\
	    VisVal =stokes_vis[ipol];\
	    /*printf("VisVal=(%f,%f), factor=(%f)\n",creal(VisVal),cimag(VisVal),factorFreq);*/\
      	    /*VisVal*=ThisWeight;*/\
	    \
	    /*if(ThisBlockAllFlagged==0){VisVal = 0.;}*/\
	    \
      	    /* Map to grid polarization. Only use pol if needed.*/\
      	    int gridPol = PolMap[ipol];\
      	    if (gridPol >= 0  &&  gridPol < nGridPol) {\
      	      size_t goff = (gridChan*nGridPol + gridPol) * nGridX * nGridY;\
      	      int sy;\
      	      float complex* __restrict__ gridPtr;\
      	      const float complex* __restrict__ cf0;\
      	      int io=(offy - fsupy*fsampy);\
      	      int jo=(offx - fsupx*fsampx);\
      	      int cfoff = io * OverS * SupportCF*SupportCF + jo * SupportCF*SupportCF;\
      	      cf0 =  p_complex64(cfs) + cfoff;\
      	      for (sy=-fsupy; sy<=fsupy; ++sy) {\
      		gridPtr =  p_complex64(grid) + goff + (locy+sy)*nGridX + locx-supx;\
      		int sx;\
      		for (sx=-fsupx; sx<=fsupx; ++sx) {\
      		  /*printf("gird=(%f,%f), vis=(%f,%f), cf=(%f,%f)\n",creal((*gridPtr)),cimag((*gridPtr)),creal(VisVal),cimag(VisVal),creal(*cf0),cimag(*cf0));*/\
      		  *gridPtr++ += VisVal * *cf0;\
      		  cf0 ++;\
      		}\
		\
      	      }\
      	      sumWtPtr[gridPol+gridChan*nGridPol] += ThisWeight;\
	      if(DoApplyJones){\
	      	ptrSumJones[gridChan]+=ThisSumJones;\
	      	ptrSumJones[gridChan+nGridChan]+=ThisSumSqWeights;\
		\
		for(visChan=0; visChan<nVisChan; visChan++){\
		  ptrSumJonesChan[visChan]+=ThisSumJonesChan[visChan];\
		  ptrSumJonesChan[nVisChan+visChan]+=ThisSumSqWeightsChan[visChan];\
		}\
	      \
	      }\
	    \
      	    } /* end if gridPol */\
      	  } /* end for ipol */\
      	} /* end if ongrid */\
      } /* end if gridChan */\
      /*AddTimeit(PreviousTime,TimeGrid);*/\
      \
    } /*end for Block*/\
    \
    \
    /* /\* printf("Times:\n"); *\/ */\
    /* double tottime=*TimeShift+*TimeApplyJones+*TimeJones+*TimeGrid+*TimeAverage+*TimeGetJones+*TimeStuff; */\
    /* double tShift=100.*(((double)(*TimeShift))/tottime); */\
    /* double tApplyJones=100.*(((double)(*TimeApplyJones))/tottime); */\
    /* double tJones=100.*(((double)(*TimeJones))/tottime); */\
    /* double tGrid=100.*(((double)(*TimeGrid))/tottime); */\
    /* double tAverage=100.*(((double)(*TimeAverage))/tottime); */\
    /* double tGetJones=100.*(((double)(*TimeGetJones))/tottime); */\
    /* double tStuff=100.*(((double)(*TimeStuff))/tottime); */\
    \
    /* printf("TimeShift:      %5.2f\n",tShift); */\
    /* printf("TimeApplyJones: %5.2f\n",tApplyJones); */\
    /* printf("TimeJones:      %5.2f\n",tJones); */\
    /* printf("TimeGrid:       %5.2f\n",tGrid); */\
    /* printf("TimeAverage:    %5.2f\n",tAverage); */\
    /* printf("TimeGetJones:   %5.2f\n",tGetJones); */\
    /* printf("TimeStuff:      %5.2f\n",tStuff); */\
    \
    free(CurrentCorrTerm);\
    free(dCorrTerm);\
    free(CurrentCorrChan);\
    free(ThisSumJonesChan);\
    free(ThisSumSqWeightsChan);\
  } /* end */\

  
// Stamp out the many many many combinations:
gridder_factory(gridderWPol_I_FROM_XXXYYXYY, GMODE_STOKES_I_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_I_FROM_XXYY, GMODE_STOKES_I_FROM_XXYY, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_IQ_FROM_XXYY, GMODE_STOKES_IQ_FROM_XXYY, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_QI_FROM_XXYY, GMODE_STOKES_QI_FROM_XXYY, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_I_FROM_RRRLLRLL, GMODE_STOKES_I_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_I_FROM_RRLL, GMODE_STOKES_I_FROM_RRLL, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_IV_FROM_RRLL, GMODE_STOKES_IV_FROM_RRLL, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_VI_FROM_RRLL, GMODE_STOKES_VI_FROM_RRLL, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_Q_FROM_RRRLLRLL, GMODE_STOKES_Q_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_U_FROM_RRRLLRLL, GMODE_STOKES_U_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_V_FROM_XXXYYXYY, GMODE_STOKES_V_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_V_FROM_RRLL, GMODE_STOKES_V_FROM_RRLL, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_V_FROM_RRRLLRLL, GMODE_STOKES_V_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_Q_FROM_XXXYYXYY, GMODE_STOKES_Q_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_Q_FROM_XXYY, GMODE_STOKES_Q_FROM_XXYY, READ_2CORR_PAD, MULACCUM_2CORR_UNPAD)
gridder_factory(gridderWPol_U_FROM_XXXYYXYY, GMODE_STOKES_U_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQ_FROM_XXXYYXYY, GMODE_STOKES_IQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QI_FROM_XXXYYXYY, GMODE_STOKES_QI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IU_FROM_XXXYYXYY, GMODE_STOKES_IU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UI_FROM_XXXYYXYY, GMODE_STOKES_UI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IV_FROM_XXXYYXYY, GMODE_STOKES_IV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VI_FROM_XXXYYXYY, GMODE_STOKES_VI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQ_FROM_XXXYYXYY, GMODE_STOKES_UQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QU_FROM_XXXYYXYY, GMODE_STOKES_QU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QV_FROM_XXXYYXYY, GMODE_STOKES_QV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQ_FROM_XXXYYXYY, GMODE_STOKES_VQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UV_FROM_XXXYYXYY, GMODE_STOKES_UV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VU_FROM_XXXYYXYY, GMODE_STOKES_VU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQ_FROM_RRRLLRLL, GMODE_STOKES_IQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QI_FROM_RRRLLRLL, GMODE_STOKES_QI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IU_FROM_RRRLLRLL, GMODE_STOKES_IU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UI_FROM_RRRLLRLL, GMODE_STOKES_UI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IV_FROM_RRRLLRLL, GMODE_STOKES_IV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VI_FROM_RRRLLRLL, GMODE_STOKES_VI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQ_FROM_RRRLLRLL, GMODE_STOKES_UQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QU_FROM_RRRLLRLL, GMODE_STOKES_QU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QV_FROM_RRRLLRLL, GMODE_STOKES_QV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQ_FROM_RRRLLRLL, GMODE_STOKES_VQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UV_FROM_RRRLLRLL, GMODE_STOKES_UV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VU_FROM_RRRLLRLL, GMODE_STOKES_VU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQU_FROM_XXXYYXYY, GMODE_STOKES_IQU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUQ_FROM_XXXYYXYY, GMODE_STOKES_IUQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIQ_FROM_XXXYYXYY, GMODE_STOKES_UIQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQI_FROM_XXXYYXYY, GMODE_STOKES_UQI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUI_FROM_XXXYYXYY, GMODE_STOKES_QUI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIU_FROM_XXXYYXYY, GMODE_STOKES_QIU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQV_FROM_XXXYYXYY, GMODE_STOKES_IQV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVQ_FROM_XXXYYXYY, GMODE_STOKES_IVQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIQ_FROM_XXXYYXYY, GMODE_STOKES_VIQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQI_FROM_XXXYYXYY, GMODE_STOKES_VQI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVI_FROM_XXXYYXYY, GMODE_STOKES_QVI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIV_FROM_XXXYYXYY, GMODE_STOKES_QIV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUV_FROM_XXXYYXYY, GMODE_STOKES_IUV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVU_FROM_XXXYYXYY, GMODE_STOKES_IVU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIU_FROM_XXXYYXYY, GMODE_STOKES_VIU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUI_FROM_XXXYYXYY, GMODE_STOKES_VUI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVI_FROM_XXXYYXYY, GMODE_STOKES_UVI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIV_FROM_XXXYYXYY, GMODE_STOKES_UIV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUV_FROM_XXXYYXYY, GMODE_STOKES_QUV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVU_FROM_XXXYYXYY, GMODE_STOKES_QVU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQU_FROM_XXXYYXYY, GMODE_STOKES_VQU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUQ_FROM_XXXYYXYY, GMODE_STOKES_VUQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVQ_FROM_XXXYYXYY, GMODE_STOKES_UVQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQV_FROM_XXXYYXYY, GMODE_STOKES_UQV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQUV_FROM_XXXYYXYY, GMODE_STOKES_IQUV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQVU_FROM_XXXYYXYY, GMODE_STOKES_IQVU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUQV_FROM_XXXYYXYY, GMODE_STOKES_IUQV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUVQ_FROM_XXXYYXYY, GMODE_STOKES_IUVQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVQU_FROM_XXXYYXYY, GMODE_STOKES_IVQU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVUQ_FROM_XXXYYXYY, GMODE_STOKES_IVUQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIUV_FROM_XXXYYXYY, GMODE_STOKES_QIUV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIVU_FROM_XXXYYXYY, GMODE_STOKES_QIVU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIUQ_FROM_XXXYYXYY, GMODE_STOKES_VIUQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIQU_FROM_XXXYYXYY, GMODE_STOKES_VIQU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIVQ_FROM_XXXYYXYY, GMODE_STOKES_UIVQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIQV_FROM_XXXYYXYY, GMODE_STOKES_UIQV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUIV_FROM_XXXYYXYY, GMODE_STOKES_QUIV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQIV_FROM_XXXYYXYY, GMODE_STOKES_UQIV_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVIQ_FROM_XXXYYXYY, GMODE_STOKES_UVIQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUIQ_FROM_XXXYYXYY, GMODE_STOKES_VUIQ_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQIU_FROM_XXXYYXYY, GMODE_STOKES_VQIU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVIU_FROM_XXXYYXYY, GMODE_STOKES_QVIU_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUVI_FROM_XXXYYXYY, GMODE_STOKES_QUVI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQVI_FROM_XXXYYXYY, GMODE_STOKES_UQVI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVQI_FROM_XXXYYXYY, GMODE_STOKES_UVQI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUQI_FROM_XXXYYXYY, GMODE_STOKES_VUQI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQUI_FROM_XXXYYXYY, GMODE_STOKES_VQUI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVUI_FROM_XXXYYXYY, GMODE_STOKES_QVUI_FROM_XXXYYXYY, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQU_FROM_RRRLLRLL, GMODE_STOKES_IQU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUQ_FROM_RRRLLRLL, GMODE_STOKES_IUQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIQ_FROM_RRRLLRLL, GMODE_STOKES_UIQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQI_FROM_RRRLLRLL, GMODE_STOKES_UQI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUI_FROM_RRRLLRLL, GMODE_STOKES_QUI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIU_FROM_RRRLLRLL, GMODE_STOKES_QIU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQV_FROM_RRRLLRLL, GMODE_STOKES_IQV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVQ_FROM_RRRLLRLL, GMODE_STOKES_IVQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIQ_FROM_RRRLLRLL, GMODE_STOKES_VIQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQI_FROM_RRRLLRLL, GMODE_STOKES_VQI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVI_FROM_RRRLLRLL, GMODE_STOKES_QVI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIV_FROM_RRRLLRLL, GMODE_STOKES_QIV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUV_FROM_RRRLLRLL, GMODE_STOKES_IUV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVU_FROM_RRRLLRLL, GMODE_STOKES_IVU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIU_FROM_RRRLLRLL, GMODE_STOKES_VIU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUI_FROM_RRRLLRLL, GMODE_STOKES_VUI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVI_FROM_RRRLLRLL, GMODE_STOKES_UVI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIV_FROM_RRRLLRLL, GMODE_STOKES_UIV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUV_FROM_RRRLLRLL, GMODE_STOKES_QUV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVU_FROM_RRRLLRLL, GMODE_STOKES_QVU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQU_FROM_RRRLLRLL, GMODE_STOKES_VQU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUQ_FROM_RRRLLRLL, GMODE_STOKES_VUQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVQ_FROM_RRRLLRLL, GMODE_STOKES_UVQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQV_FROM_RRRLLRLL, GMODE_STOKES_UQV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQUV_FROM_RRRLLRLL, GMODE_STOKES_IQUV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IQVU_FROM_RRRLLRLL, GMODE_STOKES_IQVU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUQV_FROM_RRRLLRLL, GMODE_STOKES_IUQV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IUVQ_FROM_RRRLLRLL, GMODE_STOKES_IUVQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVQU_FROM_RRRLLRLL, GMODE_STOKES_IVQU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_IVUQ_FROM_RRRLLRLL, GMODE_STOKES_IVUQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIUV_FROM_RRRLLRLL, GMODE_STOKES_QIUV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QIVU_FROM_RRRLLRLL, GMODE_STOKES_QIVU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIUQ_FROM_RRRLLRLL, GMODE_STOKES_VIUQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VIQU_FROM_RRRLLRLL, GMODE_STOKES_VIQU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIVQ_FROM_RRRLLRLL, GMODE_STOKES_UIVQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UIQV_FROM_RRRLLRLL, GMODE_STOKES_UIQV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUIV_FROM_RRRLLRLL, GMODE_STOKES_QUIV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQIV_FROM_RRRLLRLL, GMODE_STOKES_UQIV_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVIQ_FROM_RRRLLRLL, GMODE_STOKES_UVIQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUIQ_FROM_RRRLLRLL, GMODE_STOKES_VUIQ_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQIU_FROM_RRRLLRLL, GMODE_STOKES_VQIU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVIU_FROM_RRRLLRLL, GMODE_STOKES_QVIU_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QUVI_FROM_RRRLLRLL, GMODE_STOKES_QUVI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UQVI_FROM_RRRLLRLL, GMODE_STOKES_UQVI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_UVQI_FROM_RRRLLRLL, GMODE_STOKES_UVQI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VUQI_FROM_RRRLLRLL, GMODE_STOKES_VUQI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_VQUI_FROM_RRRLLRLL, GMODE_STOKES_VQUI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)
gridder_factory(gridderWPol_QVUI_FROM_RRRLLRLL, GMODE_STOKES_QVUI_FROM_RRRLLRLL, READ_4CORR, MULACCUM_4CORR)


////////////////////


static PyObject *pyGridderWPol(PyObject *self, PyObject *args)
{
  PyObject *ObjGridIn;
  PyArrayObject *np_grid, *vis, *uvw, *cfs, *flags, *weights, *sumwt, *increment, *freqs,*WInfos,
    *SmearMapping,*Sparsification,*np_ChanMapping;

  PyObject *Lcfs,*LOptimisation,*LSmearing;
  PyObject *LJones,*Lmaps;
  PyObject *LcfsConj;
  PyArrayObject *LDataCorrFormat;
  PyArrayObject *LExpectedOutStokes;
  int dopsf;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iO!O!O!O!O!O!O!O!O!O!O!O!O!O!",
			//&ObjGridIn,
			&PyArray_Type,  &np_grid, 
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
			&PyList_Type, &LJones,
			&PyArray_Type,  &SmearMapping,
			&PyArray_Type,  &Sparsification,
			&PyList_Type, &LOptimisation,
			&PyList_Type, &LSmearing,
			&PyArray_Type,  &np_ChanMapping,
			&PyArray_Type, &LDataCorrFormat,
			&PyArray_Type, &LExpectedOutStokes
			))  return NULL;
  int nx,ny,nz,nzz;
  //np_grid = (PyArrayObject *) PyArray_ContiguousFromObject(ObjGridIn, PyArray_COMPLEX64, 0, 4);
  (PyArrayObject *) PyList_GetItem(Lmaps, 0);
  char* stokeslookup[] = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
  size_t ncorr = PyArray_Size((PyObject*)LDataCorrFormat);
  size_t npol = PyArray_Size((PyObject*)LExpectedOutStokes);
  char** inputcorr = (char**) malloc(ncorr * sizeof(char*));
  char** expstokes = (char**) malloc(npol * sizeof(char*));
  short i;
  for (i=0; i < ncorr; ++i) {
    uint16_t corrid = *((uint16_t*) LDataCorrFormat->data + i);
    if (corrid < 5 || corrid > 12) {
      FATAL("Only accepts RR,RL,LR,LL,XX,XY,YX,YY as correlation input type");
    }
    inputcorr[i] = stokeslookup[corrid];
  }
  for (i=0; i < npol; ++i) {
    uint16_t polid = *((uint16_t*) LExpectedOutStokes->data + i);
    if (polid < 1 || polid > 4) {
      FATAL("Only accepts I,Q,U,V as polarization output type");
    }
    expstokes[i] = stokeslookup[polid];
  }
  int LengthJonesList=PyList_Size(LJones);
  #define callgridder(gname) \
    gname(np_grid, vis, uvw, flags, weights, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones, SmearMapping,\
	  Sparsification, LOptimisation,LSmearing,np_ChanMapping);
  if (ncorr == 4 && 
    !strcmp(inputcorr[0], "XX") && !strcmp(inputcorr[1], "XY") && !strcmp(inputcorr[2], "YX") && !strcmp(inputcorr[3], "YY")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	callgridder(gridderWPol_I_FROM_XXXYYXYY);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "Q")){
	callgridder(gridderWPol_Q_FROM_XXXYYXYY);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "U")){
	callgridder(gridderWPol_U_FROM_XXXYYXYY);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "V")){
	callgridder(gridderWPol_V_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_IQ_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_QI_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_IV_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_VI_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_UV_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U")){
	callgridder(gridderWPol_VU_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_UQ_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U")){
	callgridder(gridderWPol_QU_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_UI_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U")){
	callgridder(gridderWPol_IU_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_QV_FROM_XXXYYXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_VQ_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_IQU_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_IUQ_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_UQI_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_QUI_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_QIU_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_IQV_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_IVQ_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_VIQ_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_VQI_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_QVI_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_IQU_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_IQV_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_VIU_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_VUI_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_UVI_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_UIV_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_QUV_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_QVU_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_VQU_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_VUQ_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_UVQ_FROM_XXXYYXYY);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_UQV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_IQUV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_IUQV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_IUVQ_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_IVQU_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_QIUV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_QIVU_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_VIUQ_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_UIVQ_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_UIQV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_QUIV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_UQIV_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_UVIQ_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_VUIQ_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_VQIU_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_QVIU_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_QUVI_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_UQVI_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_UVQI_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_VUQI_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_VQUI_FROM_XXXYYXYY);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_QVUI_FROM_XXXYYXYY);
    } else {
      FATAL("Cannot convert input correlations to desired output correlations.");
    }
  } else if (ncorr == 4 && 
    !strcmp(inputcorr[0], "RR") && !strcmp(inputcorr[1], "RL") && !strcmp(inputcorr[2], "LR") && !strcmp(inputcorr[3], "LL")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	callgridder(gridderWPol_I_FROM_RRRLLRLL);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "Q")){
	callgridder(gridderWPol_Q_FROM_RRRLLRLL);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "U")){
	callgridder(gridderWPol_U_FROM_RRRLLRLL);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "V")){
	callgridder(gridderWPol_V_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_IQ_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_QI_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_IV_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_VI_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_UV_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U")){
	callgridder(gridderWPol_VU_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_UQ_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U")){
	callgridder(gridderWPol_QU_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_UI_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U")){
	callgridder(gridderWPol_IU_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_QV_FROM_RRRLLRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_VQ_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_IQU_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_IUQ_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_UQI_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_QUI_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_QIU_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_IQV_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_IVQ_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_VIQ_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_VQI_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_QVI_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_IQU_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_IQV_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_VIU_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_VUI_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I")){
	callgridder(gridderWPol_UVI_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_UIV_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_QUV_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_QVU_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U")){
	callgridder(gridderWPol_VQU_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_VUQ_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q")){
	callgridder(gridderWPol_UVQ_FROM_RRRLLRLL);
    } else if (npol == 3 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "V")){
	callgridder(gridderWPol_UQV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_IQUV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_IUQV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_IUVQ_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_IVQU_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_QIUV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_QIVU_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_VIUQ_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_UIVQ_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "I") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_UIQV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_QUIV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "V")){
	callgridder(gridderWPol_UQIV_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_UVIQ_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "Q")){
	callgridder(gridderWPol_VUIQ_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_VQIU_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "I") && !strcmp(expstokes[3], "U")){
	callgridder(gridderWPol_QVIU_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_QUVI_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "V") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_UQVI_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "U") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_UVQI_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "U") && !strcmp(expstokes[2], "Q") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_VUQI_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "Q") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_VQUI_FROM_RRRLLRLL);
    } else if (npol == 4 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "V") && !strcmp(expstokes[2], "U") && !strcmp(expstokes[3], "I")){
	callgridder(gridderWPol_QVUI_FROM_RRRLLRLL);
    } else {
      FATAL("Cannot convert input correlations to desired output correlations.");
    }
  } else if (ncorr == 2 && 
    !strcmp(inputcorr[0], "XX") && !strcmp(inputcorr[1], "YY")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	callgridder(gridderWPol_I_FROM_XXYY);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "Q")){
	callgridder(gridderWPol_Q_FROM_XXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "Q")){
	callgridder(gridderWPol_IQ_FROM_XXYY);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "Q") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_QI_FROM_XXYY);
    } else {
      FATAL("Cannot convert input correlations to desired output correlations.");
    }
  } else if (ncorr == 2 && 
    !strcmp(inputcorr[0], "RR") && !strcmp(inputcorr[1], "LL")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	callgridder(gridderWPol_I_FROM_RRLL);
    } else if (npol == 1 &&
	!strcmp(expstokes[0], "V")){
	callgridder(gridderWPol_V_FROM_RRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "I") && !strcmp(expstokes[1], "V")){
	callgridder(gridderWPol_IV_FROM_RRLL);
    } else if (npol == 2 &&
	!strcmp(expstokes[0], "V") && !strcmp(expstokes[1], "I")){
	callgridder(gridderWPol_VI_FROM_RRLL);
    } else {
      FATAL("Cannot convert input correlations to desired output correlations.");
    }
  } else {
    FATAL("Cannot convert input correlations to desired output correlations.");
  }
  free(inputcorr);
  free(expstokes); 
  Py_INCREF(Py_None);
  return Py_None;

}
#define APPLYJONES_4_CORR \
  MatDot(J0,JonesType,corr_vis,SkyType,visBuff);\
  MatDot(visBuff,SkyType,J1H,JonesType,visBuff);\
  Mat_A_l_SumProd(visBuff, SkyType, corr);\
  
#define APPLYJONES_2_CORR \
  /*Currently we only handle the diagonal case (I without Q or V in linear or circular feeds)*/ \
  float _Complex padded_corr_vis[] = {0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I,0+0*_Complex_I}; \
  padded_corr_vis[0] = corr_vis[0]; \
  padded_corr_vis[3] = corr_vis[1]; \
  MatDot(J0,JonesType,padded_corr_vis,SkyType,visBuff);\
  MatDot(visBuff,SkyType,J1H,JonesType,visBuff);\
  Mat_A_l_SumProd(visBuff, SkyType, corr);\
  visBuff[1] = visBuff[3];

#define degridder_factory(degriddername, stokesconversion, nVisPol, nVisCorr, applyjones) \
void degriddername(PyArrayObject *grid, \
		   PyArrayObject *vis, \
		   PyArrayObject *uvw, \
		   PyArrayObject *flags, \
		   PyArrayObject *sumwt, \
		   int dopsf, \
		   PyObject *Lcfs, \
		   PyObject *LcfsConj, \
		   PyArrayObject *Winfos, \
		   PyArrayObject *increment, \
		   PyArrayObject *freqs, \
		   PyObject *Lmaps, PyObject *LJones, \
		   PyArrayObject *SmearMapping, \
		   PyArrayObject *Sparsification, \
		   PyObject *LOptimisation, PyObject *LSmearing, \
		   PyArrayObject *np_ChanMapping) \
{ \
    /* Get size of convolution functions. */ \
    PyArrayObject *cfs; \
    PyArrayObject *NpPolMap; \
    NpPolMap = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 0), PyArray_INT32, 0, 4); \
    int npolsMap=NpPolMap->dimensions[0]; \
    \
    int *p_ChanMapping=p_int32(np_ChanMapping); \
    \
    PyArrayObject *NpFacetInfos; \
    NpFacetInfos = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 1), PyArray_FLOAT64, 0, 4); \
    \
    PyArrayObject *NpRows; \
    NpRows = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 2), PyArray_INT32, 0, 4); \
    int* ptrRows=I_ptr(NpRows); \
    int row0=ptrRows[0]; \
    int row1=ptrRows[1]; \
    \
    int LengthSmearingList=PyList_Size(LSmearing); \
    float DT,Dnu,lmin_decorr,mmin_decorr; \
    double* uvw_dt_Ptr; \
    int DoSmearTime,DoSmearFreq; \
    int DoDecorr=(LengthSmearingList>0); \
    \
    if(DoDecorr){ \
      uvw_dt_Ptr = p_float64((PyArrayObject *) PyList_GetItem(LSmearing, 0));\
      \
      PyObject *_FDT= PyList_GetItem(LSmearing, 1);\
      DT=(float) (PyFloat_AsDouble(_FDT));\
      PyObject *_FDnu= PyList_GetItem(LSmearing, 2);\
      Dnu=(float) (PyFloat_AsDouble(_FDnu));\
      \
      PyObject *_DoSmearTime= PyList_GetItem(LSmearing, 3);\
      DoSmearTime=(int) (PyFloat_AsDouble(_DoSmearTime));\
      \
      PyObject *_DoSmearFreq= PyList_GetItem(LSmearing, 4);\
      DoSmearFreq=(int) (PyFloat_AsDouble(_DoSmearFreq));\
      \
      PyObject *_Flmin_decorr= PyList_GetItem(LSmearing, 5);\
      lmin_decorr=(float) (PyFloat_AsDouble(_Flmin_decorr));\
      PyObject *_Fmmin_decorr= PyList_GetItem(LSmearing, 6);\
      mmin_decorr=(float) (PyFloat_AsDouble(_Fmmin_decorr));\
      \
    }\
    \
    double VarTimeDeGrid=0;\
    int Nop=0;\
    \
    double* ptrFacetInfos=p_float64(NpFacetInfos);\
    double Cu=ptrFacetInfos[0];\
    double Cv=ptrFacetInfos[1];\
    double l0=ptrFacetInfos[2];\
    double m0=ptrFacetInfos[3];\
    double n0=sqrt(1-l0*l0-m0*m0)-1;\
    \
    /* Get size of grid. */\
    double* ptrWinfo = p_float64(Winfos);\
    double WaveRefWave = ptrWinfo[0];\
    double wmax = ptrWinfo[1];\
    double NwPlanes = ptrWinfo[2];\
    int OverS=floor(ptrWinfo[3]);\
    \
    int nGridX    = grid->dimensions[3];\
    int nGridY    = grid->dimensions[2];\
    int nGridPol  = grid->dimensions[1];\
    int nGridChan = grid->dimensions[0];\
    \
    /* Get visibility data size. */\
    int nVisChan  = flags->dimensions[1];\
    int nrows     = uvw->dimensions[0];\
    \
    /* Get oversampling and support size. */\
    int sampx = OverS;\
    int sampy = OverS;\
    \
    double* __restrict__ sumWtPtr = p_float64(sumwt);\
    double complex psfValues[4];\
    psfValues[0] = psfValues[1] = psfValues[2] = psfValues[3] = 1;\
    \
    /*uint inxRowWCorr(0);*/\
    \
    double offset_p[2],uvwScale_p[2];\
    \
    offset_p[0]=nGridX/2;\
    offset_p[1]=nGridY/2;\
    float fnGridX=nGridX;\
    float fnGridY=nGridY;\
    double *incr=p_float64(increment);\
    double *Pfreqs=p_float64(freqs);\
    uvwScale_p[0]=fnGridX*incr[0];\
    uvwScale_p[1]=fnGridX*incr[1];\
    double C=2.99792458e8;\
    int inx;\
    \
    /* ################### Prepare full scalar mode */\
    \
    PyObject *_JonesType  = PyList_GetItem(LOptimisation, 0);\
    int JonesType=(int) PyFloat_AsDouble(_JonesType);\
    PyObject *_ChanEquidistant  = PyList_GetItem(LOptimisation, 1);\
    int ChanEquidistant=(int) PyFloat_AsDouble(_ChanEquidistant);\
    \
    PyObject *_SkyType  = PyList_GetItem(LOptimisation, 2);\
    int SkyType=(int) PyFloat_AsDouble(_SkyType);\
    \
    PyObject *_PolMode  = PyList_GetItem(LOptimisation, 3);\
    int PolMode=(int) PyFloat_AsDouble(_PolMode);\
    \
    int ipol;\
    \
    int *MappingBlock = p_int32(SmearMapping);\
    /* total size is in two words*/\
    size_t NTotBlocks = MappingBlock[1];\
    NTotBlocks <<= 32;\
    NTotBlocks += MappingBlock[0];\
    int * NRowBlocks = MappingBlock+2;\
    int * StartRow = MappingBlock+2+NTotBlocks;\
    size_t iBlock;\
    \
    int NMaxRow=0;\
    for(iBlock=0; iBlock<NTotBlocks; iBlock++){\
      int NRowThisBlock=NRowBlocks[iBlock]-2;\
      if(NRowThisBlock>NMaxRow){\
	NMaxRow=NRowThisBlock;\
      }\
    }\
    float complex *CurrentCorrTerm=calloc(1,(NMaxRow)*sizeof(float complex));\
    float complex *dCorrTerm=calloc(1,(NMaxRow)*sizeof(float complex));\
    /* ######################################################## */\
    double posx,posy;\
    \
    double WaveLengthMean=0.;\
    size_t visChan;\
    for (visChan=0; visChan<nVisChan; ++visChan){\
      WaveLengthMean+=C/Pfreqs[visChan];\
    }\
    WaveLengthMean/=nVisChan;\
    \
    sem_t * Sem_mutex;\
    initJonesServer(LJones,JonesType,WaveLengthMean);\
    \
    for(iBlock=0; iBlock<NTotBlocks; iBlock++){\
      int NRowThisBlock=NRowBlocks[iBlock]-2;\
      int chStart = StartRow[0];\
      int chEnd = StartRow[1];\
      int *Row = StartRow+2;\
      /* advance pointer to next blocklist */\
      StartRow += NRowBlocks[iBlock];\
      \
      float complex Vis[4]={0};\
      double Umean=0;\
      double Vmean=0;\
      double Wmean=0;\
      double FreqMean=0;\
      int NVisThisblock=0;\
      \
      float visChanMean=0.;\
      resetJonesServerCounter();\
      for (inx=0; inx<NRowThisBlock; inx++) {\
	size_t irow = Row[inx];\
	if(irow>nrows){continue;}\
	double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;\
	\
	int ThisPol;\
	for (visChan=chStart; visChan<chEnd; ++visChan) {\
	  size_t doff = (irow * nVisChan + visChan) * nVisCorr;\
	  int OneFlagged=0;\
	  int cond;\
	  \
	  float U=(float)uvwPtr[0];\
	  float V=(float)uvwPtr[1];\
	  float W=(float)uvwPtr[2];\
	  \
	  U+=W*Cu;\
	  V+=W*Cv;\
	  \
	  /*###################### Averaging #######################*/\
	  Umean+=U;\
	  Vmean+=V;\
	  Wmean+=W;\
	  FreqMean+=(float)Pfreqs[visChan];\
	  visChanMean+=p_ChanMapping[visChan];\
	  NVisThisblock+=1;\
	}/* endfor vischan*/\
      }/*endfor RowThisBlock*/\
      \
      if(NVisThisblock==0){continue;}\
      Umean/=NVisThisblock;\
      Vmean/=NVisThisblock;\
      Wmean/=NVisThisblock;\
      FreqMean/=NVisThisblock;\
      \
      visChanMean/=NVisThisblock;\
      int ThisGridChan=p_ChanMapping[chStart];\
      float diffChan=visChanMean-visChanMean;\
      \
      if(fabs(diffChan)>1e-6){printf("degridder: probably there is a problem in the BDA mapping: (ChanMean, ThisGridChan, diff)=(%f, %i, %10f)\n",visChanMean,ThisGridChan,diffChan);}\
      \
      visChanMean=0.;\
      \
      /* ################################################\
         ############## Start Gridding visibility ####### */\
      int gridChan = p_ChanMapping[chStart];\
      \
      int CFChan = 0;\
      double recipWvl = FreqMean / C;\
      double ThisWaveLength=C/FreqMean;\
      \
      /* ############## W-projection #################### */\
      double wcoord=Wmean;\
      \
      int iwplane = floor((NwPlanes-1)*fabs(wcoord)*(WaveRefWave/ThisWaveLength)/wmax+0.5);\
      int skipW=0;\
      if(iwplane>NwPlanes-1){skipW=1;continue;};\
      \
      if(wcoord>0){\
	cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);\
      } else{\
	cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);\
      }\
      int nConvX = cfs->dimensions[0];\
      int nConvY = cfs->dimensions[1];\
      int supx = (nConvX/OverS-1)/2;\
      int supy = (nConvY/OverS-1)/2;\
      int SupportCF=nConvX/OverS;\
      \
      if (gridChan >= 0  &&  gridChan < nGridChan) {\
      	double posx,posy;\
      	\
      	posx = uvwScale_p[0] * Umean * recipWvl + offset_p[0];\
      	posy = uvwScale_p[1] * Vmean * recipWvl + offset_p[1];\
	\
      	int locx = nint (posx);    /* location in grid */\
      	int locy = nint (posy);\
      	\
      	double diffx = locx - posx;\
      	double diffy = locy - posy;\
      	\
      	int offx = nint (diffx * sampx); /* location in*/\
      	int offy = nint (diffy * sampy); /* oversampling*/\
      	\
      	offx += (nConvX-1)/2;\
      	offy += (nConvY-1)/2;\
      	/* Scaling with frequency is not necessary (according to Cyril). */\
      	double freqFact = 1;\
      	int fsampx = nint (sampx * freqFact);\
      	int fsampy = nint (sampy * freqFact);\
      	int fsupx  = nint (supx / freqFact);\
      	int fsupy  = nint (supy / freqFact);\
      \
      	/* Only use visibility point if the full support is within grid. */\
      	if (locx-supx >= 0  &&  locx+supx < nGridX  &&\
      	    locy-supy >= 0  &&  locy+supy < nGridY) {\
      	  float complex stokes_vis[nVisPol]={0 + 0*_Complex_I};\
	  \
      	  int ipol;\
	  \
      	  for (ipol=0; ipol<nVisPol; ++ipol) {\
      	      \
	      int goff = (gridChan*nGridPol + ipol) * nGridX * nGridY;\
	      int sy;\
	      \
	      const float complex* __restrict__ gridPtr;\
	      const float complex* __restrict__ cf0;\
	      \
	      int io=(offy - fsupy*fsampy);\
	      int jo=(offx - fsupx*fsampx);\
	      int cfoff = io * OverS * SupportCF*SupportCF + jo * SupportCF*SupportCF;\
	      cf0 =  p_complex64(cfs) + cfoff;\
	      \
	      for (sy=-fsupy; sy<=fsupy; ++sy) {\
		gridPtr =  p_complex64(grid) + goff + (locy+sy)*nGridX + locx-supx;\
		int sx;\
		for (sx=-fsupx; sx<=fsupx; ++sx) {\
		  stokes_vis[ipol] += *gridPtr  * *cf0;\
		  cf0 ++;\
		  gridPtr++;\
		} /* end for sup x*/\
	      } /* end for sup y */\
      	  } /* end for ipol */\
	  \
	  /*###########################################################\
	   ######## Convert from degridded stokes to MS corrs #########*/\
	  \
	  stokesconversion\
	  \
	  /* ###########################################################\
	     ################### Now do the correction #################*/\
	  \
	  float DeCorrFactor=1.;\
	  if(DoDecorr){\
	    int iRowMeanThisBlock=Row[NRowThisBlock/2];\
	    \
	    double*  __restrict__ uvwPtrMidRow   = p_float64(uvw) + iRowMeanThisBlock*3;\
	    double*  __restrict__ uvw_dt_PtrMidRow   = uvw_dt_Ptr + iRowMeanThisBlock*3;\
	    \
	    DeCorrFactor=GiveDecorrelationFactor(DoSmearFreq,DoSmearTime,\
						 (float)lmin_decorr,\
						 (float)mmin_decorr,\
						 uvwPtrMidRow,\
						 uvw_dt_PtrMidRow,\
						 (float)FreqMean,\
						 (float)Dnu,\
						 (float)DT);\
	  }\
	  \
	  for (inx=0; inx<NRowThisBlock; inx++) {\
	      size_t irow = Row[inx];\
	      if(irow>nrows){continue;}\
	      double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;\
	      \
	      int ThisPol;\
	      for (visChan=chStart; visChan<chEnd; ++visChan) {\
		size_t doff_chan = irow * nVisChan + visChan;\
		size_t doff = (doff_chan) * nVisCorr;\
		bool* __restrict__ flagPtr = p_bool(flags) + doff;\
		int OneFlagged=0;\
		int cond;\
		\
		if(DoApplyJones){\
		  updateJones(irow, visChan, uvwPtr, 0,0);\
		} /*endif DoApplyJones*/\
		\
		/*###################### Facetting #######################\
		  Change coordinate and shift visibility to facet center */\
		float U=(float)uvwPtr[0];\
		float V=(float)uvwPtr[1];\
		float W=(float)uvwPtr[2];\
		/*#######################################################*/\
		\
		float complex corr;\
		if(ChanEquidistant){\
		  if(visChan==0){\
		    float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;\
		    CurrentCorrTerm[inx]=cexp(UVNorm*(U*l0+V*m0+W*n0));\
		    float complex dUVNorm=2.*I*PI*(Pfreqs[1]-Pfreqs[0])/C;\
		    dCorrTerm[inx]=cexp(dUVNorm*(U*l0+V*m0+W*n0));\
		  }else{\
		    CurrentCorrTerm[inx]*=dCorrTerm[inx];\
		  }\
		  corr=CurrentCorrTerm[inx];\
		}\
		else{\
		  float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;\
		  corr=cexp(UVNorm*(U*l0+V*m0+W*n0));\
		}\
		\
		corr*=DeCorrFactor;\
		\
		float complex* __restrict__ visPtr  = p_complex64(vis)  + doff;\
		\
		float complex visBuff[4]={0};\
		if(DoApplyJones){\
		  applyjones \
		} else {\
		  for(ThisPol =0; ThisPol<nVisCorr; ++ThisPol){\
		    visBuff[ThisPol]=corr_vis[ThisPol] * corr;\
		  }\
		}\
		\
		Sem_mutex=GiveSemaphoreFromCell(doff_chan);\
		/* Finally subtract visibilities from current residues*/\
		sem_wait(Sem_mutex);\
		for(ThisPol =0; ThisPol<nVisCorr; ++ThisPol){\
		    visPtr[ThisPol] -= visBuff[ThisPol];\
		}\
		sem_post(Sem_mutex);\
		\
	      }/*endfor vischan*/\
	    }/*endfor RowThisBlock*/\
      	} /* end if ongrid*/\
      } /* end if gridChan*/\
    } /*end for Block*/\
    \
    free(CurrentCorrTerm);\
    free(dCorrTerm);\
 } /* end */\

 
// Stamp out the various degridders needed for Stokes I deconvolution:
// We will not support full polarization cleaning so just stokes I is needed
degridder_factory(degridderWPol_RRLL_FROM_I, GMODE_CORR_RRLL_FROM_I, 1, 2, APPLYJONES_2_CORR)
degridder_factory(degridderWPol_RRRLLRLL_FROM_I, GMODE_CORR_RRRLLRLL_FROM_I, 1, 4, APPLYJONES_4_CORR)
degridder_factory(degridderWPol_XXYY_FROM_I, GMODE_CORR_XXYY_FROM_I, 1, 2, APPLYJONES_2_CORR)
degridder_factory(degridderWPol_XXXYYXYY_FROM_I, GMODE_CORR_XXXYYXYY_FROM_I, 1, 4, APPLYJONES_4_CORR)
 
static PyObject *pyDeGridderWPol(PyObject *self, PyObject *args)
{
  PyObject *ObjGridIn;
  PyObject *ObjVis;
  PyArrayObject *np_grid, *np_vis, *uvw, *cfs, *flags, *sumwt, *increment, *freqs,*WInfos,
                *SmearMapping, *Sparsification, *np_ChanMapping;

  PyObject *Lcfs, *LOptimisation, *LSmear;
  PyObject *Lmaps,*LJones;
  PyObject *LcfsConj;
  PyArrayObject *LDataCorrFormat;
  PyArrayObject *LExpectedOutStokes;
  int dopsf;

  if (!PyArg_ParseTuple(args, "O!OO!O!O!iO!O!O!O!O!O!O!O!O!O!O!O!O!O!",
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
			&PyList_Type, &Lmaps, &PyList_Type, &LJones,
			&PyArray_Type, &SmearMapping,
			&PyArray_Type, &Sparsification,
			&PyList_Type, &LOptimisation,
			&PyList_Type, &LSmear,
			&PyArray_Type, &np_ChanMapping,
			&PyArray_Type, &LDataCorrFormat,
			&PyArray_Type, &LExpectedOutStokes
			))  return NULL;
  int nx,ny,nz,nzz;
  
  np_vis = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVis, PyArray_COMPLEX64, 0, 3);

  
  char* stokeslookup[] = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
  size_t ncorr = PyArray_Size((PyObject*)LDataCorrFormat);
  size_t npol = PyArray_Size((PyObject*)LExpectedOutStokes);
  char** inputcorr = (char**) malloc(ncorr * sizeof(char*));
  char** expstokes = (char**) malloc(npol * sizeof(char*));
  short i;
  for (i=0; i < ncorr; ++i) {
    uint16_t corrid = *((uint16_t*) LDataCorrFormat->data + i);
    if (corrid < 5 || corrid > 12) {
      FATAL("Only accepts RR,RL,LR,LL,XX,XY,YX,YY as correlation output types");
    }
    inputcorr[i] = stokeslookup[corrid];
  }
  for (i=0; i < npol; ++i) {
    uint16_t polid = *((uint16_t*) LExpectedOutStokes->data + i);
    if (polid != 1) {
      FATAL("Only accepts I as polarization input type");
    }
    expstokes[i] = stokeslookup[polid];
  }
  int LengthJonesList=PyList_Size(LJones);
  #define calldegridder(gname) \
    gname(np_grid, np_vis, uvw, flags, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones, SmearMapping, \
	  Sparsification, LOptimisation, LSmear,np_ChanMapping);
  if (ncorr == 4 && 
    !strcmp(inputcorr[0], "XX") && !strcmp(inputcorr[1], "XY") && !strcmp(inputcorr[2], "YX") && !strcmp(inputcorr[3], "YY")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	calldegridder(degridderWPol_XXXYYXYY_FROM_I);
    }
    else {
      FATAL("Cannot convert input stokes parameter to desired output correlations.");
    }
  } else if (ncorr == 2 && 
    !strcmp(inputcorr[0], "XX") && !strcmp(inputcorr[1], "YY")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	calldegridder(degridderWPol_XXYY_FROM_I);
    }
    else {
      FATAL("Cannot convert input stokes parameter to desired output correlations.");
    }
  } else if (ncorr == 4 && 
    !strcmp(inputcorr[0], "RR") && !strcmp(inputcorr[1], "RL") && !strcmp(inputcorr[2], "LR") && !strcmp(inputcorr[3], "LL")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	calldegridder(degridderWPol_RRRLLRLL_FROM_I);
    }
    else {
      FATAL("Cannot convert input stokes parameter to desired output correlations.");
    }
  } else if (ncorr == 2 && 
    !strcmp(inputcorr[0], "RR") && !strcmp(inputcorr[1], "LL")) {
    if (npol == 1 &&
	!strcmp(expstokes[0], "I")){
	calldegridder(degridderWPol_RRLL_FROM_I);
    }
    else {
      FATAL("Cannot convert input stokes parameter to desired output correlations.");
    }
  } else {
    FATAL("Cannot convert input stokes parameter to desired output correlations.");
  }
  free(inputcorr);
  free(expstokes);
  return PyArray_Return(np_vis);
}
