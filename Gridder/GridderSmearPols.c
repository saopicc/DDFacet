#include "GridderSmearPols.h"

clock_t start;

float GiveDecorrelationFactor(int FSmear, int TSmear,
                              float l0, float m0,
                              double* uvwPtr,
                              double* uvw_dt_Ptr,
                              float nu,
                              float Dnu,
                              float DT) {
    //float PI=3.141592;
    //float C=2.99792456e8;

    float n0=sqrt(1.-l0*l0-m0*m0)-1.;
    float DecorrFactor=1.;
    float phase=0;
    float phi=0;
    phase=(uvwPtr[0])*l0;
    phase+=(uvwPtr[1])*m0;
    phase+=(uvwPtr[2])*n0;

    if(FSmear==1) {
        phi=PI*(Dnu/C)*phase;
        if(phi!=0.) {
            DecorrFactor*=(float)(sin((double)phi)/((double)phi));
        };
    };

    float du,dv,dw;
    float dphase;
    if(TSmear==1) {

        du=uvw_dt_Ptr[0]*l0;
        dv=uvw_dt_Ptr[1]*m0;
        dw=uvw_dt_Ptr[2]*n0;
        dphase=(du+dv+dw)*DT;
        phi=PI*(nu/C)*dphase;
        if(phi!=0.) {
            DecorrFactor*=(sin(phi)/(phi));
        };
    };
    return DecorrFactor;
}

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
			  gridding_parameters * out)
{
  PyArrayObject *NpFacetInfos;
  NpFacetInfos = (PyArrayObject *) PyList_GetItem(Lmaps, 0);
  //###################### Decorrelation #######################
  //Activates decorrelation smearing when LSmearing python list is not empty
  out->LengthSmearingList=PyList_Size(LSmearing);
  
  out->DoDecorr=(out->LengthSmearingList>0);
  if (out->DoDecorr)
    if (out->LengthSmearingList != 4)
      cexcept("Expect LSmearing python list to be composed of [DT,Dnu,DoSmearTime,DoSmearFreq]\n");
    
  if(out->DoDecorr) {
      out->uvw_dt_Ptr = p_float64((PyArrayObject *) PyList_GetItem(LSmearing, 0));

      PyObject *_FDT= PyList_GetItem(LSmearing, 1);
      out->DT=(float) (PyFloat_AsDouble(_FDT));
      PyObject *_FDnu= PyList_GetItem(LSmearing, 2);
      out->Dnu=(float) (PyFloat_AsDouble(_FDnu));

      PyObject *_DoSmearTime= PyList_GetItem(LSmearing, 3);
      out->DoSmearTime=(int) (PyFloat_AsDouble(_DoSmearTime));

      PyObject *_DoSmearFreq= PyList_GetItem(LSmearing, 4);
      out->DoSmearFreq=(int) (PyFloat_AsDouble(_DoSmearFreq));
  }
  
  //###################### Facet centre #######################
  double* ptrFacetInfos=p_float64(NpFacetInfos);
  if (NpFacetInfos->nd != 1)
      cexcept("Expect ptrFacetInfos to be composed of [Cu,Cv,l0,m0]\n");
  if (NpFacetInfos->dimensions[0] != 4)
      cexcept("Expect ptrFacetInfos to be composed of [Cu,Cv,l0,m0]\n");
  out->Cu=ptrFacetInfos[0];
  out->Cv=ptrFacetInfos[1];
  out->l0=ptrFacetInfos[2]; //delta l
  out->m0=ptrFacetInfos[3]; //delta m
  out->n0=sqrt(1-out->l0*out->l0-out->m0*out->m0)-1;
  
  //###################### W-projection #######################
  double * ptrWinfo = p_float64(Winfos);
  if (NpFacetInfos->nd != 1)
      cexcept("Expect ptrFacetInfos to be composed of [ref_wavelength,w_max,num_w_planes,oversampling_factor]\n");
  if (NpFacetInfos->dimensions[0] != 4)
      cexcept("Expect ptrFacetInfos to be composed of [ref_wavelength,w_max,num_w_planes,oversampling_factor]\n");
  out->WaveRefWave = ptrWinfo[0];
  out->wmax = ptrWinfo[1];
  out->NwPlanes = ptrWinfo[2];
  out->OverS=floor(ptrWinfo[3]);
  // Get oversampling and support size.
  out->sampx = out->OverS;
  out->sampy = out->OverS;
  if (PyList_Size(Lcfs) != out->NwPlanes)
    cexcept("W plane filter stack should have NwPlanes\n");
  if (PyList_Size(LcfsConj) != out->NwPlanes)
    cexcept("Conjugate W plane filter stack should have NwPlanes\n");
  //TODO: Need to parse these into plain old arrays when doing this on the GPU
  out->Lcfs = Lcfs;
  out->LcfsConj = LcfsConj;
  
  //###################### Grid #######################
  if (grid->nd != 4)
    cexcept("Grid must be 4 dimensional: #channel x #pol x #gridY x #gridX\n");
  out->nGridX    = grid->dimensions[3];
  out->nGridY    = grid->dimensions[2];
  out->nGridPol  = grid->dimensions[1];
  out->nGridChan = grid->dimensions[0];
  out->grid = p_complex64 (grid);
  
  //###################### UV Data #######################
  if (uvw->nd != 2)
    cexcept("uvw coordinates must be 2 dimensional: #rows x 3\n");
  if (uvw->dimensions[1] != 3)
    cexcept("uvw last column must have length 3\n");
  out->nrows = uvw->dimensions[0];
  if (flags->nd != 3)
    cexcept("Flags must be 3 dimensional: #rows x #chan x #corr\n");
  if (vis->nd != 3)
    cexcept("Vis must be 3 dimensional: #rows x #chan x #corr\n");
  if (!((vis->dimensions[0] == flags->dimensions[0]) && 
	(vis->dimensions[1] == flags->dimensions[1]) &&
	(vis->dimensions[2] == flags->dimensions[2])))
    cexcept("Expect Vis and Flags dimensions to be the same - one flag per channel correlation\n");
  if (!(vis->dimensions[0] == uvw->dimensions[0]))
    cexcept("Expect Vis and uvw data to have the same number of rows\n");
  out->dopsf = dopsf;
  out->nVisCorr  = flags->dimensions[2];
  out->nVisChan  = flags->dimensions[1];
  out->uvw   = p_float64(uvw);
  if (weights != NULL)
  {
    if (weights->nd != 2)
      cexcept("Expect imaging weights to be 2 dimensional: #rows x #vis_channels\n");
    if (weights->dimensions[0] != out->nrows)
      cexcept("Weights should have the same number of rows as the other uv data\n");
    if (weights->dimensions[1] != out->nVisChan)
      cexcept("Expecting one weight per visibility channel\n");
    out->weights = p_float64(weights);
  } else
    out->weights = NULL;
  
  out->flags = p_bool(flags);
  out->vis = p_complex64(vis);
  //####################### Mappings #######################
  if (np_ChanMapping->nd != 1)
    cexcept("Expect Channel to Grid Mapping to have one dimension\n");
  if (np_ChanMapping->dimensions[0] != out->nVisChan)
    cexcept("Expect Channel to Grid Mapping to have a mapping for each channel\n");
  out->p_ChanMapping=p_int32(np_ChanMapping);
  
  //###################### Weight stores #######################
  out->sumWtPtr = p_float64(sumwt);
  
  if (increment->nd != 1)
    cexcept("Expect grid increment (cell-size) to have 2 values: cell_l and cell_m\n");
  if (increment->dimensions[0] != 2)
    cexcept("Expect grid increment (cell-size) to have 2 values: cell_l and cell_m\n");
  out->incr[0]=p_float64(increment)[0];
  out->incr[1]=p_float64(increment)[1];
  if (freqs->nd != 1)
    cexcept("References freqs must be one dimensional, a frequency per visibility channel\n");
  if (freqs->dimensions[0] != out->nVisChan)
    cexcept("References freqs must be one dimensional, a frequency per visibility channel\n");
  out->Pfreqs=p_float64(freqs);
  
  //################# Optimization flags ####################### 
//   if (PyList_Size(LOptimisation) != 3)
//     cexcept("Expect LOptimization list to contain [JonesType, ChanEquidistant, SkyType]\n");
  PyObject *_JonesType  = PyList_GetItem(LOptimisation, 0);
  out->JonesType=(int) PyFloat_AsDouble(_JonesType);
  PyObject *_ChanEquidistant  = PyList_GetItem(LOptimisation, 1);
  out->ChanEquidistant=(int) PyFloat_AsDouble(_ChanEquidistant);
  PyObject *_SkyType  = PyList_GetItem(LOptimisation, 2);
  out->SkyType=(int) PyFloat_AsDouble(_SkyType);
  
  //################# BDA ####################### 
  //Work out the maximum number of rows in contained in any of the smearing (compression) blocks:
  //TODO: This needs more rigerous dimensionality and validity checks
  out->MappingBlock = p_int32(SmearMapping);
  out->NTotBlocks=out->MappingBlock[0];
  out->NRowBlocks=out->MappingBlock+1;
  out->StartRow=out->MappingBlock+1+out->NTotBlocks;
  out->NMaxRow=0;
  int iBlock;
  for(iBlock=0; iBlock<out->NTotBlocks; iBlock++) {
      int NRowThisBlock=out->NRowBlocks[iBlock]-2;
      if(NRowThisBlock>out->NMaxRow) {
	  out->NMaxRow=NRowThisBlock;
      }
  }
  
  //################ Ref wavelength ################
  out->WaveLengthMean=0.;
  out->FreqMean0=0.;

  int visChan;
  
  for (visChan=0; visChan<out->nVisChan; ++visChan) {
      WaveLengthMean+=C/out->Pfreqs[visChan];
      out->FreqMean0+=out->Pfreqs[visChan];
  }
  WaveLengthMean/=out->nVisChan;
  out->FreqMean0/=out->nVisChan;
  
  //################ Correlation to Stokes ################
  out->VisCorrDesc= I_ptr(data_corr_products);
  out->VisStokesDesc= I_ptr(output_stokes_products);
  if (data_corr_products->nd != 1)
    cexcept("UV Data correlation discriptor should be one dimensional\n");
  if (data_corr_products->dimensions[0] != out->nVisCorr)  
    cexcept("UV Data correlation discriptor to have a descriptor per Vis correlation\n");
  if (output_stokes_products->nd != 1)
    cexcept("Image stokes discriptor should be one dimensional\n");
  if (output_stokes_products->dimensions[0] != out->nGridPol)
    cexcept("Image stokes discriptor should have one discriptor per grid polarization\n");
  
  //##################### Jones #########################
  //TODO: This needs more rigerous dimensionality and validity checks
  if (PyList_Size(LJones) > 0 && out->nVisCorr != 4)
    cexcept("When applying Jones terms the number of correlations must be 4\n"); //unsupported: applying diagonal terms only
}

static PyObject *pyGridderWPol(PyObject *self, PyObject *args)
{
    PyObject *ObjGridIn;
    PyArrayObject *np_grid, *vis, *uvw, *cfs, *flags, *weights, *sumwt, *increment, *freqs,*WInfos,*SmearMapping,*np_ChanMapping;

    PyObject *Lcfs,*LOptimisation,*LSmearing;
    PyObject *LJones,*Lmaps;
    PyObject *LcfsConj;
    int dopsf;
    PyArrayObject * data_corr_products;
    PyArrayObject * output_stokes_products;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iO!O!O!O!O!O!O!O!O!O!O!O!O!", 
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
			&PyList_Type, &LOptimisation,
			&PyList_Type, &LSmearing,
			&PyArray_Type,  &np_ChanMapping,
			&PyArray_Type, &data_corr_products,
			&PyArray_Type, &output_stokes_products
			))  return NULL;
    gridding_parameters params;
    parse_python_objects(np_grid, vis, uvw, 
			 flags, weights, sumwt, 
			 dopsf, Lcfs, LcfsConj, 
			 WInfos, increment, freqs, 
			 Lmaps, LJones, SmearMapping, 
			 LOptimisation, LSmearing, np_ChanMapping,
			 data_corr_products, output_stokes_products,
			 &params);
    init_stokes_converter(params.nVisCorr,params.nGridPol,
			  params.VisCorrDesc,params.VisStokesDesc);
    initJonesServer(LJones,
		    params.JonesType,
		    params.WaveLengthMean);
    gridderWPol(&params);
    free_stokes_library();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *pyDeGridderWPol(PyObject *self, PyObject *args)
{
    PyObject *ObjVis;
    PyArrayObject *np_grid, *np_vis, *uvw, *cfs, *flags, *sumwt, *increment, *freqs,*WInfos,*SmearMapping,*np_ChanMapping;

    PyObject *Lcfs, *LOptimisation, *LSmear;
    PyObject *Lmaps,*LJones;
    PyObject *LcfsConj;
    int dopsf;
    PyArrayObject * data_corr_products;
    PyArrayObject * output_stokes_products;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iO!O!O!O!O!O!O!O!O!O!O!O!O!",
                          &PyArray_Type,  &np_grid,
                          &PyArray_Type,  &np_vis,
                          &PyArray_Type,  &uvw,
                          &PyArray_Type,  &flags,
                          &PyArray_Type,  &sumwt,
                          &dopsf,
                          &PyList_Type, &Lcfs,
                          &PyList_Type, &LcfsConj,
                          &PyArray_Type,  &WInfos,
                          &PyArray_Type,  &increment,
                          &PyArray_Type,  &freqs,
                          &PyList_Type, &Lmaps,
                          &PyList_Type, &LJones,
                          &PyArray_Type, &SmearMapping,
                          &PyList_Type, &LOptimisation,
                          &PyList_Type, &LSmear,
                          &PyArray_Type, &np_ChanMapping,
			  &PyArray_Type, &data_corr_products,
			  &PyArray_Type, &output_stokes_products
                         ))  return NULL;
//     np_vis = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVis, PyArray_COMPLEX64, 0, 3);
    gridding_parameters params;
    parse_python_objects(np_grid, np_vis, uvw, 
			 flags, NULL, sumwt, 
			 dopsf, Lcfs, LcfsConj, 
			 WInfos, increment, freqs, 
			 Lmaps, LJones, SmearMapping, 
			 LOptimisation, LSmear, np_ChanMapping,
			 data_corr_products, output_stokes_products,
			 &params);
    initJonesServer(LJones,params.JonesType,params.WaveLengthMean);
    init_stokes_converter(params.nGridPol,params.nVisCorr,
			  params.VisStokesDesc,params.VisCorrDesc);
    DeGridderWPol(&params);
    free_stokes_library();

    Py_INCREF(Py_None);
    return Py_None;
}

void gridderWPol(gridding_parameters * params)
{
    size_t inx;
    size_t iBlock;
    size_t ThisPol;
    size_t visChan;
    float factorFreq=1;
    
    PyArrayObject *cfs;

//     double VarTimeGrid=0;
//     int Nop=0;
//     long int TimeShift[1]= {0};
//     long int TimeApplyJones[1]= {0};
//     long int TimeAverage[1]= {0};
//     long int TimeJones[1]= {0};
//     long int TimeGrid[1]= {0};
//     long int TimeGetJones[1]= {0};
//     long int TimeStuff[1]= {0};
//     struct timespec PreviousTime;
    
    //u,v = 0,0 lies in the centre of the grid:
    double offset_p[2],uvwScale_p[2];
    offset_p[0]=params->nGridX/2;
    offset_p[1]=params->nGridY/2;
    float fnGridX=params->nGridX;
    float fnGridY=params->nGridY;
    uvwScale_p[0]=fnGridX*params->incr[0];
    uvwScale_p[1]=fnGridX*params->incr[1];
   
    
    float complex *CurrentCorrTerm=(float complex*)calloc(1,(params->NMaxRow)*sizeof(float complex));
    float complex *dCorrTerm=(float complex*)calloc(1,(params->NMaxRow)*sizeof(float complex));
    
    //Define visibilities (N [normally 4] input, M [#stokes params] output):
    float complex *VisCorr= (float complex*)calloc(params->nVisCorr,sizeof(float complex*));
    float complex *VisMeas= (float complex*)calloc(params->nVisCorr,sizeof(float complex*));
    float complex *VisStokes= (float complex*)calloc(params->nGridPol,sizeof(float complex*));
    
    //need to use #nCorr flags in the channel loop later:
    float * VisFlagWeight = (float *)calloc(params->nVisCorr,sizeof(float *)); 
    float * VisRealWeight = (float *)calloc(params->nVisCorr,sizeof(float *)); 
    float complex * VisComplexWeight = (float complex *)calloc(params->nVisCorr,sizeof(float complex *)); 
    //allocate buffers for jones blocks used in channel loop later on:
    double * BlockVisWeight = (double *)calloc(params->nVisCorr,sizeof(double *)); 
    float * BlockSumJones = (float *)calloc(params->nVisCorr,sizeof(float *)); 
    float * BlockSumSqWeights = (float *)calloc(params->nVisCorr,sizeof(float *)); 
    
    //Define visibilities to be gridded when constructing the PSF
    float complex *VisCorrPSF= (float complex*)calloc(params->nVisCorr,sizeof(float complex*));
    give_psf_vis_32(params->nVisCorr,params->VisCorrDesc,VisCorrPSF);

    float *ThisSumJonesChan=(float*)calloc(1,(params->nVisChan*params->nVisCorr)*sizeof(float));
    float *ThisSumSqWeightsChan=(float*)calloc(1,(params->nVisChan*params->nVisCorr)*sizeof(float));

    //Loop over all smearing blocks:
    for(iBlock=0; iBlock<params->NTotBlocks; ++iBlock) {
        int NRowThisBlock=params->NRowBlocks[iBlock]-2;
        int indexMap=params->StartRow[iBlock];
        int chStart=params->MappingBlock[indexMap];
        int chEnd=params->MappingBlock[indexMap+1];
        int *Row=params->MappingBlock+params->StartRow[iBlock]+2;

        float Umean=0;
        float Vmean=0;
        float Wmean=0;
        float FreqMean=0;
        int NVisThisblock=0;
	//zero out the average visibility accumulators for this block:
        for(ThisPol =0; ThisPol<params->nVisCorr; ThisPol++) {
            VisCorr[ThisPol]=0;
            VisMeas[ThisPol]=0;
	    BlockVisWeight[ThisPol]=0;
	    BlockSumJones[ThisPol]=0;
	    BlockSumSqWeights[ThisPol]=0;
	    for(visChan=0; visChan<params->nVisChan; visChan++) {
	      size_t weightOff = ThisPol*params->nVisChan+visChan;
	      ThisSumJonesChan[weightOff]=0;
	      ThisSumSqWeightsChan[weightOff]=0;
	    }
        }

        float visChanMean=0.;
        resetJonesServerCounter();

        for (inx=0; inx<NRowThisBlock; ++inx) {
            size_t irow = Row[inx];
            if(irow>params->nrows) {
                continue;
            }
            double*  __restrict__ uvwPtr   = params->uvw + irow*3;
            
            WeightVaryJJ=1.;

            float DeCorrFactor=1.;
            if(params->DoDecorr) {
                int iRowMeanThisBlock=Row[NRowThisBlock/2];

                double*  __restrict__ uvwPtrMidRow   = params->uvw + iRowMeanThisBlock*3;
                double*  __restrict__ uvw_dt_PtrMidRow   = params->uvw_dt_Ptr + iRowMeanThisBlock*3;

                DeCorrFactor=GiveDecorrelationFactor(params->DoSmearFreq,params->DoSmearTime,
                                                     (float)params->l0, (float)params->m0,
                                                     uvwPtrMidRow,
                                                     uvw_dt_PtrMidRow,
                                                     (float)params->FreqMean0,
                                                     (float)params->Dnu,
                                                     (float)params->DT);
            }
	    //Average all the channels within this bda block together.
	    //Since the Jones terms may vary faster than the bda smearing, the
	    //bda must happen on the fly.
            for (visChan=chStart; visChan<chEnd; ++visChan) {
                int doff = (irow * params->nVisChan + visChan) * params->nVisCorr;
                //###################### Facetting #######################
                // Change coordinate and shift visibility to facet center
		// This is in line with a coplanar faceting approach, where
		// the projection error at the edge of the facet is removed
		// by w-kernel per facet. Otherwise an additional uvw 
		// transform would be required. This Faceting phase shift 
		// is to be applied to all visibilities, rotating the sky over the
		// image plane of the facet and is direction independent
		// (only the facet reference centre is involved, so it can
		// be taken out of the RIME integral). The term it may be 
		// treated as a scalar (complex) Jones term.
                float U=(float)uvwPtr[0];
                float V=(float)uvwPtr[1];
                float W=(float)uvwPtr[2];
                
                float complex facetPhasor;
                if(params->ChanEquidistant) {
                    if(visChan==0) {
                        float complex UVNorm=2.*I*PI*params->Pfreqs[visChan]/C;
                        CurrentCorrTerm[inx]=cexp(-UVNorm*(U*params->l0+V*params->m0+W*params->n0));
                        float complex dUVNorm=2.*I*PI*(params->Pfreqs[1]-params->Pfreqs[0])/C;
                        dCorrTerm[inx]=cexp(-dUVNorm*(U*params->l0+V*params->m0+W*params->n0));
                    } else {
                        CurrentCorrTerm[inx]*=dCorrTerm[inx];
                    }
                    facetPhasor=CurrentCorrTerm[inx];
                }
                else {
                    float complex UVNorm=2.*I*PI*params->Pfreqs[visChan]/C;
                    facetPhasor=cexp(-UVNorm*(U*params->l0+V*params->m0+W*params->n0));
                }
		//#######################################################
		
		//###################### Apply Jones #######################
                if(DoApplyJones) {
                    updateJones(irow, visChan, uvwPtr, 1);
                } //endif DoApplyJones
		//beam-weight the psf:
                if (params->dopsf==1) {
		    //We want a PSF for every possible correlation
		    for (ThisPol = 0; ThisPol < params->nVisCorr; ++ThisPol){
		      VisMeas[ThisPol]= VisCorrPSF[ThisPol];
		    }
                    facetPhasor=1.;
                    if(DoApplyJones) {
                        MatDot(J0,params->JonesType,VisMeas,params->SkyType,VisMeas);
                        MatDot(VisMeas,params->SkyType,J1H,params->JonesType,VisMeas);
                    }
                    if(params->DoDecorr) {
                        for(ThisPol =0; ThisPol<params->nVisCorr; ThisPol++) {
                            VisMeas[ThisPol]*=DeCorrFactor;
                        }
                    }
                } else {
		    float complex* __restrict__ visPtrMeas  = params->vis  + doff;
                    for(ThisPol =0; ThisPol<params->nVisCorr; ThisPol++) {
                        VisMeas[ThisPol]=visPtrMeas[ThisPol];
                    }
                }
		
		if(DoApplyJones) {
		    float complex visPtr[4];
                    MatDot(J0H,params->JonesType,VisMeas,params->SkyType,visPtr);
                    MatDot(visPtr,params->SkyType,J1,params->JonesType,visPtr);
		    {
		      size_t corr;
		      for (corr=0;corr<params->nVisCorr;++corr)
			VisMeas[corr] = visPtr[corr];
		    }
                } 
                //#######################################################
		
		//###################### Flagging #######################
		//Weights flagged visibilities down to 0. Each correlation
		//may have its' own flag, so this term is not scalar and
		//should be viewed as being applied per baseline.
                bool* __restrict__ flagPtr = params->flags + doff;
		{
		  int corr;
		  for (corr = 0; corr < params->nVisCorr; ++corr)
		    VisFlagWeight[corr] = (flagPtr[0]==0) ? 1.0f : 0.0f;
		}
		//#######################################################
		
		//################# Visibility Weighting ################
		//These are visibility weights which may take the
		//visibility noise level and uniform/robust weighting
		//into account. Each correlation may have its own weight
		//so this term is not scalar. This term is also applied
		//per baseline. Pointwise apply per baseline-weights to 
		//jones-corrected visibilities:
		//TODO: the weights should include a weight for each correlation
		double* imgWtPtr = params->weights + irow  * params->nVisChan + visChan;
		{
		  int corr;
		  for (corr = 0; corr < params->nVisCorr; ++corr){
		    VisRealWeight[corr] = (*imgWtPtr)*WeightVaryJJ*DeCorrFactor*VisFlagWeight[corr];
		    VisComplexWeight[corr] = VisRealWeight[corr] * facetPhasor;
		    BlockVisWeight[corr]+=VisRealWeight[corr];
		    VisCorr[corr] += VisMeas[corr]*VisComplexWeight[corr];
		  }
		}
		//#######################################################

                U+=W*params->Cu;
                V+=W*params->Cv;
                //###################### Averaging #######################
                Umean+=U;
                Vmean+=V;
                Wmean+=W;
		if(DoApplyJones) {
		  int corr;
		  for (corr=0;corr<params->nVisCorr;++corr){
		    float FWeightSq=(VisRealWeight[corr])*(VisRealWeight[corr]);
		    BlockSumJones[corr]+=BB*FWeightSq;
		    BlockSumSqWeights[corr]+=FWeightSq;
		    size_t weightsOffset = visChan*params->nVisCorr + corr;
		    ThisSumJonesChan[weightsOffset]+=BB*FWeightSq;
		    ThisSumSqWeightsChan[weightsOffset]+=FWeightSq;
		  }
		}//DoApplyJones 
                FreqMean+=factorFreq*(float)params->Pfreqs[visChan];
                visChanMean+=params->p_ChanMapping[visChan];
                NVisThisblock+=1.;
                //#######################################################
            }//endfor vischan
        }//endfor RowThisBlock
        if(NVisThisblock==0) {
            continue;
        }
        Umean/=NVisThisblock;
        Vmean/=NVisThisblock;
        Wmean/=NVisThisblock;
        FreqMean/=NVisThisblock;	
        visChanMean/=NVisThisblock;
        int ThisGridChan=params->p_ChanMapping[chStart];
        float diffChan=visChanMean-visChanMean;
        if(diffChan!=0.) {
            printf("gridder: probably there is a problem in the BDA mapping: (ChanMean, ThisGridChan, diff)=(%f, %i, %f)\n",visChanMean,ThisGridChan,diffChan);
	    cexcept("Check your BDA mapping");
        }
	//Use the mean (smeared) frequency for uv scaling in this block:
        int gridChan = params->p_ChanMapping[chStart];
        double recipWvl = FreqMean / C;
        double ThisWaveLength=C/FreqMean;
	
        // ############## Vis Correlations -> Stokes ################
        //Now that the Jones matricies (correlations) have been applied we can 
	//convert visibility correlations of MS to Stokes parameters:
        convert_corrs_32(VisCorr,VisStokes);
        // ##########################################################
	
        // ############## W-projection ####################
	// 
        double wcoord=Wmean;
        int iwplane = floor((params->NwPlanes-1)*abs(wcoord)*(params->WaveRefWave/ThisWaveLength)/params->wmax+0.5);
        int skipW=0;
        if(iwplane>params->NwPlanes-1) {
            skipW=1;
            continue;
        };
        if(wcoord>0) {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(params->Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);
        } else {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(params->LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
        }
        int nConvX = cfs->dimensions[0];
        int nConvY = cfs->dimensions[1];
        int supx = (nConvX/params->OverS-1)/2;
        int supy = (nConvY/params->OverS-1)/2;
        int SupportCF=nConvX/params->OverS;
        // ################################################
	
	// ############## Start Gridding visibility #######
        if (gridChan >= 0  &&  gridChan < params->nGridChan) {
            double posx,posy;
            //For Even/Odd take the -1 off
            posx = uvwScale_p[0] * Umean * recipWvl + offset_p[0];//#-1;
            posy = uvwScale_p[1] * Vmean * recipWvl + offset_p[1];//-1;

            int locx = nint (posx);    // round to nearest location in grid
            int locy = nint (posy);
            double diffx = locx - posx;
            double diffy = locy - posy;
            int offx = nint (diffx * params->sampx); // location in
            int offy = nint (diffy * params->sampy); // oversampling
            offx += (nConvX-1)/2;
            offy += (nConvY-1)/2;
            //support and oversampling factors should be ints:
            int fsampx = nint (params->sampx);
            int fsampy = nint (params->sampy);
            int fsupx  = nint (supx);
            int fsupy  = nint (supy);

            // Only use visibility point if the full support is within grid.
            if (locx-supx >= 0  &&  locx+supx < params->nGridX  &&
                    locy-supy >= 0  &&  locy+supy < params->nGridY) {

                size_t ipol;
                for ( ipol=0; ipol<params->nGridPol; ++ipol ) {
                    float complex VisVal = VisStokes[ipol];
                    // Map to grid polarization. Only use pol if needed.
                    int goff = ( gridChan*params->nGridPol + ipol ) * params->nGridX * params->nGridY;
                    int sy;
                    float complex* __restrict__ gridPtr;
                    const float complex* __restrict__ cf0;
                    int io= ( offy - fsupy*fsampy );
                    int jo= ( offx - fsupx*fsampx );
                    int cfoff = io * params->OverS * SupportCF*SupportCF + jo * SupportCF*SupportCF;
                    cf0 =  p_complex64 ( cfs ) + cfoff;
                    for ( sy=-fsupy; sy<=fsupy; ++sy ) {
                        gridPtr =  params->grid + goff + ( locy+sy ) *params->nGridX + locx-supx;
                        int sx;
                        for ( sx=-fsupx; sx<=fsupx; ++sx ) {
                            *gridPtr++ += VisVal * *cf0;
                            cf0 ++;
                        }

                    }
                    
                    //Accumulate normalization weights for this facet
                    params->sumWtPtr[gridChan*params->nGridPol + ipol] += BlockVisWeight[ipol];
                    if ( DoApplyJones ) {
			//TODO: This has to be changed to include all polarizations
                        ptrSumJones[gridChan]+=BlockSumJones[0];
                        ptrSumJones[gridChan+params->nGridChan]+=BlockSumSqWeights[0];

                        for ( visChan=0; visChan<params->nVisChan; visChan++ ) {
			    //TODO: This has to be changed to include all polarizations
			    size_t weightOffset = visChan*params->nVisCorr + 0;
                            ptrSumJonesChan[visChan]+=ThisSumJonesChan[weightOffset];
                            ptrSumJonesChan[params->nVisChan+visChan]+=ThisSumSqWeightsChan[weightOffset];
                        }
                    } //end DoApplyJones
                } // end for ipol
            } // end if ongrid
        } // end if gridChan
        //AddTimeit(PreviousTime,TimeGrid);

    } //end for Block
    
    //Finally free allocated memory of this method:
    free(CurrentCorrTerm);
    free(VisCorr);
    free(VisMeas);
    free(VisStokes);
    free(ThisSumJonesChan);
    free(ThisSumSqWeightsChan);
    free(dCorrTerm);
    free(VisFlagWeight);
    free(VisRealWeight);
    free(VisComplexWeight);
    free(BlockVisWeight);
    free(BlockSumJones);
    free(BlockSumSqWeights);

    /* /\* printf("Times:\n"); *\/ */
    /* double tottime=*TimeShift+*TimeApplyJones+*TimeJones+*TimeGrid+*TimeAverage+*TimeGetJones+*TimeStuff; */
    /* double tShift=100.*(((double)(*TimeShift))/tottime); */
    /* double tApplyJones=100.*(((double)(*TimeApplyJones))/tottime); */
    /* double tJones=100.*(((double)(*TimeJones))/tottime); */
    /* double tGrid=100.*(((double)(*TimeGrid))/tottime); */
    /* double tAverage=100.*(((double)(*TimeAverage))/tottime); */
    /* double tGetJones=100.*(((double)(*TimeGetJones))/tottime); */
    /* double tStuff=100.*(((double)(*TimeStuff))/tottime); */

    /* printf("TimeShift:      %5.2f\n",tShift); */
    /* printf("TimeApplyJones: %5.2f\n",tApplyJones); */
    /* printf("TimeJones:      %5.2f\n",tJones); */
    /* printf("TimeGrid:       %5.2f\n",tGrid); */
    /* printf("TimeAverage:    %5.2f\n",tAverage); */
    /* printf("TimeGetJones:   %5.2f\n",tGetJones); */
    /* printf("TimeStuff:      %5.2f\n",tStuff); */

} // end

void DeGridderWPol(gridding_parameters *  params)
{
    size_t inx;
    size_t iBlock;
    size_t visChan;
    PyArrayObject *cfs;
    
//     double VarTimeDeGrid=0;
//     int Nop=0;
    
    //Offset uv coverage to centre of grid:
    double offset_p[2],uvwScale_p[2];
    offset_p[0]=params->nGridX/2;//(nGridX-1)/2.;
    offset_p[1]=params->nGridY/2;
    float fnGridX=params->nGridX;
    float fnGridY=params->nGridY;
    uvwScale_p[0]=fnGridX*params->incr[0];
    uvwScale_p[1]=fnGridX*params->incr[1];

    
    float complex *CurrentCorrTerm=(float complex *)calloc(1,(params->NMaxRow)*sizeof(float complex));
    float complex *dCorrTerm=(float complex *)calloc(1,(params->NMaxRow)*sizeof(float complex));
    
    float complex *model_vis_stokes=(float complex *)calloc(1,(params->nGridPol)*sizeof(float complex));
    float complex *model_vis_corr=(float complex *)calloc(1,(params->nVisCorr)*sizeof(float complex));
    float complex *phased_vis_corr=(float complex *)calloc(1,(params->nVisCorr)*sizeof(float complex));
    // ########################################################
    // For each BDA smearing block:
    for(iBlock=0; iBlock<params->NTotBlocks; iBlock++) {
        //for(iBlock=3507; iBlock<3508; iBlock++){
        int NRowThisBlock=params->NRowBlocks[iBlock]-2;
        int indexMap=params->StartRow[iBlock];
        int chStart=params->MappingBlock[indexMap];
        int chEnd=params->MappingBlock[indexMap+1];
        int *Row=params->MappingBlock+params->StartRow[iBlock]+2;

        float Umean=0;
        float Vmean=0;
        float Wmean=0;
        float FreqMean=0;
        int NVisThisblock=0;
        float visChanMean=0.;
        resetJonesServerCounter();
	//Compute average u,v,w coordinate and frequency of the block:
        for (inx=0; inx<NRowThisBlock; inx++) {
            int irow = Row[inx];
            if(irow>params->nrows) {
                continue;
            }
            double*  __restrict__ uvwPtr   = params->uvw + irow*3;

            for (visChan=chStart; visChan<chEnd; ++visChan) {		
                float U=(float)uvwPtr[0];
                float V=(float)uvwPtr[1];
                float W=(float)uvwPtr[2];

                U+=W*params->Cu;
                V+=W*params->Cv;
		
                Umean+=U;
                Vmean+=V;
                Wmean+=W;
                FreqMean+=(float)params->Pfreqs[visChan];
                visChanMean+=params->p_ChanMapping[visChan];
                NVisThisblock+=1;
            }//endfor vischan
        }//endfor RowThisBlock
        if(NVisThisblock==0) {
            continue;
        }
        Umean/=NVisThisblock;
        Vmean/=NVisThisblock;
        Wmean/=NVisThisblock;
        FreqMean/=NVisThisblock;

        visChanMean/=NVisThisblock;
        int ThisGridChan=params->p_ChanMapping[chStart];
        float diffChan=visChanMean-visChanMean;

        if(diffChan!=0.) {
            printf("degridder: probably there is a problem in the BDA mapping: (ChanMean, ThisGridChan, diff)=(%f, %i, %f)\n",visChanMean,ThisGridChan,diffChan);
        }
        visChanMean=0.;

        int gridChan = params->p_ChanMapping[chStart];

        double recipWvl = FreqMean / C;
        double ThisWaveLength=C/FreqMean;

        // ############## W-reprojection ####################
        double wcoord=Wmean;
        int iwplane = floor((params->NwPlanes-1)*abs(wcoord)*(params->WaveRefWave/ThisWaveLength)/params->wmax+0.5);
        int skipW=0;
        if(iwplane>params->NwPlanes-1) {
            skipW=1;
            continue;
        };
        if(wcoord>0) {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(params->Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);
        } else {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(params->LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
        }
        int nConvX = cfs->dimensions[0];
        int nConvY = cfs->dimensions[1];
        int supx = (nConvX/params->OverS-1)/2;
        int supy = (nConvY/params->OverS-1)/2;
        int SupportCF=nConvX/params->OverS;
	// ################################################

	// ############## Start Degridding visibility #######
	// This gathers the bda average visibility from the convolution
	// area around the average uvw,lambda coordinates
        if (gridChan >= 0  &&  gridChan < params->nGridChan) {
            double posx,posy;
            posx = uvwScale_p[0] * Umean * recipWvl + offset_p[0];
            posy = uvwScale_p[1] * Vmean * recipWvl + offset_p[1];

            int locx = nint (posx);    // location in grid
            int locy = nint (posy);
            double diffx = locx - posx;
            double diffy = locy - posy;
            int offx = nint (diffx * params->sampx); // location in
            int offy = nint (diffy * params->sampy); // oversampling
            offx += (nConvX-1)/2;
            offy += (nConvY-1)/2;
	    //support and oversampling factor should be integers:
            int fsampx = nint (params->sampx);
            int fsampy = nint (params->sampy);
            int fsupx  = nint (supx);
            int fsupy  = nint (supy);

            // Only use visibility point if the full support is within grid.
            if (locx-supx >= 0  &&  locx+supx < params->nGridX  &&
                    locy-supy >= 0  &&  locy+supy < params->nGridY) {                
                int ipol;
                for (ipol=0; ipol<params->nGridPol; ++ipol) {
		    model_vis_stokes[ipol] = 0 + 0*_Complex_I; //reset accumulator
		    int goff = (gridChan*params->nGridPol + ipol) * params->nGridX * params->nGridY;
		    int sy;

		    const float complex* __restrict__ gridPtr;
		    const float complex* __restrict__ cf0;

		    int io=(offy - fsupy*fsampy);
		    int jo=(offx - fsupx*fsampx);
		    int cfoff = io * params->OverS * SupportCF*SupportCF + jo * SupportCF*SupportCF;
		    cf0 =  p_complex64(cfs) + cfoff;
		    for (sy=-fsupy; sy<=fsupy; ++sy) {
			gridPtr =  params->grid + goff + (locy+sy)*params->nGridX + locx-supx;
			int sx;
			for (sx=-fsupx; sx<=fsupx; ++sx) {
			    model_vis_stokes[ipol] += *gridPtr  * *cf0;
			    cf0 ++;
			    gridPtr++;
			}
		    }
                } // end for ipol

                // ################### Stokes -> Corrs #################
		// Need to convert stokes fourier components to correlations
		// before applying the Jones corruptions
                convert_corrs_32(model_vis_stokes,model_vis_corr);
		// ###########################################################
		
                // ################### Decorrelation #################
                float DeCorrFactor=1.;
                if(params->DoDecorr) {
                    int iRowMeanThisBlock=Row[NRowThisBlock/2];

                    double*  __restrict__ uvwPtrMidRow   = params->uvw + iRowMeanThisBlock*3;
                    double*  __restrict__ uvw_dt_PtrMidRow   = params->uvw_dt_Ptr + iRowMeanThisBlock*3;

                    DeCorrFactor=GiveDecorrelationFactor(params->DoSmearFreq,params->DoSmearTime,
                                                         (float)params->l0, (float)params->m0,
                                                         uvwPtrMidRow,
                                                         uvw_dt_PtrMidRow,
                                                         (float)FreqMean,
                                                         (float)params->Dnu,
                                                         (float)params->DT);


                }
                // ###########################################################
		//Now put the gathered average visibility into the contributing channels
                for (inx=0; inx<NRowThisBlock; inx++) {
                    size_t irow = Row[inx];
                    if(irow>params->nrows) {
                        continue;
                    }
                    double*  __restrict__ uvwPtr   = params->uvw + irow*3;
                    size_t ThisPol;
                    for (visChan=chStart; visChan<chEnd; ++visChan) {
                        size_t doff = (irow * params->nVisChan + visChan) * params->nVisCorr;

                        if(DoApplyJones) {
                            updateJones(irow, visChan, uvwPtr, 0);
                        } //endif DoApplyJones

                        //###################### Facetting #######################
                        // Change coordinate and shift visibility from facet center
                        // to original projection pole.
                        // See the comment in the gridder. This is the inverse
                        // phasor.
                        float U=(float)uvwPtr[0];
                        float V=(float)uvwPtr[1];
                        float W=(float)uvwPtr[2];
                        float complex phasor;
                        if(params->ChanEquidistant) {
                            if(visChan==0) {
                                float complex UVNorm=2.*I*PI*params->Pfreqs[visChan]/C;
                                CurrentCorrTerm[inx]=cexp(UVNorm*(U*params->l0+V*params->m0+W*params->n0));
                                float complex dUVNorm=2.*I*PI*(params->Pfreqs[1]-params->Pfreqs[0])/C;
                                dCorrTerm[inx]=cexp(dUVNorm*(U*params->l0+V*params->m0+W*params->n0));
                            } else {
                                CurrentCorrTerm[inx]*=dCorrTerm[inx];
                            }
                            phasor=CurrentCorrTerm[inx];
                        }
                        else {
                            float complex UVNorm=2.*I*PI*params->Pfreqs[visChan]/C;
                            phasor=cexp(UVNorm*(U*params->l0+V*params->m0+W*params->n0));
                        }
                        //#######################################################
                        
                        phasor*=DeCorrFactor;
			
			//###################### Jones Corruption #######################
			//The model visibilities are only phase-steered after applying the
			//corrupting jones terms.
			//No weights need be applied here because the observed
			//visibilities are unweighted in the DATA column of the MS, contrary
			//to what is done in gridding
                        float complex* __restrict__ visPtr  = params->vis  + doff;
                        
                        if(DoApplyJones) {
			    float complex visBuff[4]= {0};
                            MatDot(J0,JonesType,model_vis_corr,params->SkyType,visBuff);
                            MatDot(visBuff,params->SkyType,J1H,JonesType,visBuff);
			    for(ThisPol =0; ThisPol<4; ThisPol++) {
                                phased_vis_corr[ThisPol]=visBuff[ThisPol]*phasor;
                            }
                        } else {
			  for(ThisPol =0; ThisPol < params->nVisCorr; ThisPol++) {
                                phased_vis_corr[ThisPol]=model_vis_corr[ThisPol]*phasor;
                          }
                        }
			//#######################################################
			
			//###################### Form residuals #######################
			for (ThisPol=0; ThisPol < params->nVisCorr; ++ThisPol){
			    visPtr[ThisPol] = visPtr[ThisPol] - phased_vis_corr[ThisPol];
			}
			//#######################################################
                    }//endfor vischan
                }//endfor RowThisBlock
            } // end if ongrid
        } // end if gridChan
    } //end for Block
    
    //finally free dynamic memory:
    free(CurrentCorrTerm);
    free(dCorrTerm);
    free(model_vis_stokes);
    free(model_vis_corr);
    free(phased_vis_corr);
} // end
