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

static PyObject *pyGridderWPol(PyObject *self, PyObject *args)
{
    PyObject *ObjGridIn;
    PyArrayObject *np_grid, *vis, *uvw, *cfs, *flags, *weights, *sumwt, *increment, *freqs,*WInfos,*SmearMapping,*np_ChanMapping;

    PyObject *Lcfs,*LOptimisation,*LSmearing;
    PyObject *LJones,*Lmaps;
    PyObject *LcfsConj;
    int dopsf;
    PyObject * data_corr_products;
    PyObject * output_stokes_products;

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
			&PyList_Type, &data_corr_products,
			&PyList_Type, &output_stokes_products
			))  return NULL;

    gridderWPol(np_grid, vis, uvw, 
		flags, weights, sumwt, 
		dopsf, Lcfs, LcfsConj, 
		WInfos, increment, freqs, 
		Lmaps, LJones, SmearMapping, 
		LOptimisation, LSmearing, np_ChanMapping,
		data_corr_products, output_stokes_products);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *pyDeGridderWPol(PyObject *self, PyObject *args)
{
    PyObject *ObjGridIn;
    PyObject *ObjVis;
    PyArrayObject *np_grid, *np_vis, *uvw, *cfs, *flags, *sumwt, *increment, *freqs,*WInfos,*SmearMapping,*np_ChanMapping;

    PyObject *Lcfs, *LOptimisation, *LSmear;
    PyObject *Lmaps,*LJones;
    PyObject *LcfsConj;
    int dopsf;

    if (!PyArg_ParseTuple(args, "O!OO!O!O!iO!O!O!O!O!O!O!O!O!O!O!",
                          &PyArray_Type,  &np_grid,
                          &ObjVis,
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
                          &PyArray_Type, &np_ChanMapping
                         ))  return NULL;

    np_vis = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVis, PyArray_COMPLEX64, 0, 3);
    DeGridderWPol(np_grid, np_vis, uvw, flags, sumwt, dopsf, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones, SmearMapping, LOptimisation, LSmear,np_ChanMapping);

    return PyArray_Return(np_vis);
}

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
                 PyObject *Lmaps,
                 PyObject *LJones,
                 PyArrayObject *SmearMapping,
                 PyObject *LOptimisation,
                 PyObject *LSmearing,
                 PyArrayObject *np_ChanMapping,
		 PyObject *data_corr_products,
		 PyObject *output_stokes_products)
{
    
    int nrows     = uvw->dimensions[0];
    PyArrayObject *cfs;
    
    PyArrayObject *NpFacetInfos;
    NpFacetInfos = (PyArrayObject *) PyList_GetItem(Lmaps, 0);

    //###################### Decorrelation #######################
    //Activates decorrelation smearing when LSmearing python list is not empty
    int LengthSmearingList=PyList_Size(LSmearing);
    float DT,Dnu;
    double* uvw_dt_Ptr;
    int DoSmearTime,DoSmearFreq;
    int DoDecorr=(LengthSmearingList>0);
    if (DoDecorr)
      if (LengthSmearingList != 4)
	cexcept("Expect LSmearing python list to be composed of [DT,Dnu,DoSmearTime,DoSmearFreq]");
      
    if(DoDecorr) {
        uvw_dt_Ptr = p_float64((PyArrayObject *) PyList_GetItem(LSmearing, 0));

        PyObject *_FDT= PyList_GetItem(LSmearing, 1);
        DT=(float) (PyFloat_AsDouble(_FDT));
        PyObject *_FDnu= PyList_GetItem(LSmearing, 2);
        Dnu=(float) (PyFloat_AsDouble(_FDnu));

        PyObject *_DoSmearTime= PyList_GetItem(LSmearing, 3);
        DoSmearTime=(int) (PyFloat_AsDouble(_DoSmearTime));

        PyObject *_DoSmearFreq= PyList_GetItem(LSmearing, 4);
        DoSmearFreq=(int) (PyFloat_AsDouble(_DoSmearFreq));
    }
    // ########################################################
    //###################### Facet centre #######################
    double* ptrFacetInfos=p_float64(NpFacetInfos);
    double Cu=ptrFacetInfos[0];
    double Cv=ptrFacetInfos[1];
    double l0=ptrFacetInfos[2];
    double m0=ptrFacetInfos[3];
    double n0=sqrt(1-l0*l0-m0*m0)-1;
    // ########################################################

    double VarTimeGrid=0;
    int Nop=0;

    double* ptrWinfo = p_float64(Winfos);
    double WaveRefWave = ptrWinfo[0];
    double wmax = ptrWinfo[1];
    double NwPlanes = ptrWinfo[2];
    int OverS=floor(ptrWinfo[3]);

    // Get size of grid.
    if (grid->nd != 4)
      cexcept("Grid must be 4 dimensional: #channel x #pol x #gridY x #gridX\n");
    int nGridX    = grid->dimensions[3];
    int nGridY    = grid->dimensions[2];
    int nGridPol  = grid->dimensions[1];
    int nGridChan = grid->dimensions[0];

    // Get visibility data size.
    if (flags->nd != 3)
      cexcept("Flags must be 3 dimensional: #rows x #chan x #corr\n");
    if (vis->nd != 3)
      cexcept("Vis must be 3 dimensional: #rows x #chan x #corr\n");
    if (!((vis->dimensions[0] == flags->dimensions[0]) && 
	  (vis->dimensions[1] == flags->dimensions[1]) &&
	  (vis->dimensions[2] == flags->dimensions[2])))
      cexcept("Expect Vis and Flags dimensions to be the same - one flag per channel correlation\n");
    int nVisCorr   = flags->dimensions[2];
    int nVisChan  = flags->dimensions[1];
    if (DoApplyJones && nVisCorr != 4)
      cexcept("When applying Jones terms the number of correlations must be 4\n");
    if (np_ChanMapping->nd != 1)
      cexcept("Expect Channel to Grid Mapping to have one dimension\n");
    if (np_ChanMapping->dimensions[0] != nVisChan)
      cexcept("Expect Channel to Grid Mapping to have a mapping for each channel\n");
    int *p_ChanMapping=p_int32(np_ChanMapping);
    
    // Get oversampling and support size.
    int sampx = OverS;//int (cfs.sampling[0]);
    int sampy = OverS;//int (cfs.sampling[1]);

    double* __restrict__ sumWtPtr = p_float64(sumwt);

    double offset_p[2],uvwScale_p[2];
    
    //u,v = 0,0 lies in the centre of the grid:
    offset_p[0]=nGridX/2;
    offset_p[1]=nGridY/2;
    float fnGridX=nGridX;
    float fnGridY=nGridY;
    double *incr=p_float64(increment);
    double *Pfreqs=p_float64(freqs);
    uvwScale_p[0]=fnGridX*incr[0];
    uvwScale_p[1]=fnGridX*incr[1];
   
    int inx;
    
    // Get any optimizations to apply
    if (PyList_Size(LOptimisation) != 3)
      cexcept("Expect LOptimization list to contain [JonesType, ChanEquidistant, SkyType]\n");
    PyObject *_JonesType  = PyList_GetItem(LOptimisation, 0);
    int JonesType=(int) PyFloat_AsDouble(_JonesType);
    PyObject *_ChanEquidistant  = PyList_GetItem(LOptimisation, 1);
    int ChanEquidistant=(int) PyFloat_AsDouble(_ChanEquidistant);
    PyObject *_SkyType  = PyList_GetItem(LOptimisation, 2);
    int SkyType=(int) PyFloat_AsDouble(_SkyType);

    //Work out the maximum number of rows in contained in any of the smearing (compression) blocks:
    int *MappingBlock = p_int32(SmearMapping);
    int NTotBlocks=MappingBlock[0];
    int *NRowBlocks=MappingBlock+1;
    int *StartRow=MappingBlock+1+NTotBlocks;
    int iBlock;
    int NMaxRow=0;
    for(iBlock=0; iBlock<NTotBlocks; iBlock++) {
        int NRowThisBlock=NRowBlocks[iBlock]-2;
        if(NRowThisBlock>NMaxRow) {
            NMaxRow=NRowThisBlock;
        }
    }
    float complex *CurrentCorrTerm=(float complex*)calloc(1,(NMaxRow)*sizeof(float complex));
    float complex *dCorrTerm=(float complex*)calloc(1,(NMaxRow)*sizeof(float complex));

    // ########################################################

    double WaveLengthMean=0.;
    double FreqMean0=0.;

    int visChan;
    float factorFreq=1;

    for (visChan=0; visChan<nVisChan; ++visChan) {
        WaveLengthMean+=C/Pfreqs[visChan];
        FreqMean0+=Pfreqs[visChan];
    }
    WaveLengthMean/=nVisChan;
    FreqMean0/=nVisChan;
    float FracFreqWidth=0;
    if (nVisChan>1) {
        float DeltaFreq=(Pfreqs[nVisChan-1]-Pfreqs[0]);
        FracFreqWidth=DeltaFreq/FreqMean0;
    }

    initJonesServer(LJones,JonesType,WaveLengthMean);

    long int TimeShift[1]= {0};
    long int TimeApplyJones[1]= {0};
    long int TimeAverage[1]= {0};
    long int TimeJones[1]= {0};
    long int TimeGrid[1]= {0};
    long int TimeGetJones[1]= {0};
    long int TimeStuff[1]= {0};
    struct timespec PreviousTime;
    
    //Define visibilities (N [normally 4] input, M [#stokes params] output):
    float complex *VisCorr= (float complex*)calloc(nVisCorr,sizeof(float complex*));
    float complex *VisMeas= (float complex*)calloc(nVisCorr,sizeof(float complex*));
    float complex *VisStokes= (float complex*)calloc(nGridPol,sizeof(float complex*));
    
    //need to use #nCorr flags in the channel loop later:
    float * VisFlagWeight = (float *)calloc(nVisCorr,sizeof(float *)); 
    float * VisRealWeight = (float *)calloc(nVisCorr,sizeof(float *)); 
    float complex * VisComplexWeight = (float complex *)calloc(nVisCorr,sizeof(float complex *)); 
    //allocate buffers for jones blocks used in channel loop later on:
    double * BlockVisWeight = (double *)calloc(nVisCorr,sizeof(double *)); 
    float * BlockSumJones = (float *)calloc(nVisCorr,sizeof(float *)); 
    float * BlockSumSqWeights = (float *)calloc(nVisCorr,sizeof(float *)); 
    // Get visibility correlation and output stokes descriptors:
    int* VisCorrDesc= (int*)calloc(nVisCorr,sizeof(int*));
    int* VisStokesDesc= (int*)calloc(nGridPol,sizeof(int*));
    {
      size_t k = 0;
      for (k = 0; k < nVisCorr; ++k){
	VisCorrDesc[k] = (int)PyLong_AsLong(PyList_GetItem(data_corr_products, k));
      }
      for (k = 0; k < nGridPol; ++k){
	VisStokesDesc[k] = (int)PyLong_AsLong(PyList_GetItem(output_stokes_products, k));
      }
      //Init stokes conversion library:
      init_stokes_converter(nVisCorr,nGridPol,VisCorrDesc,VisStokesDesc);
    }
    
    //Define visibilities to be gridded when constructing the PSF
    float complex *VisCorrPSF= (float complex*)calloc(nVisCorr,sizeof(float complex*));
    give_psf_vis_32(nVisCorr,VisCorrDesc,VisCorrPSF);
    
    int ThisPol;

    float *ThisSumJonesChan=(float*)calloc(1,(nVisChan*nVisCorr)*sizeof(float));
    float *ThisSumSqWeightsChan=(float*)calloc(1,(nVisChan*nVisCorr)*sizeof(float));

    //Loop over all smearing blocks:
    for(iBlock=0; iBlock<NTotBlocks; ++iBlock) {
        int NRowThisBlock=NRowBlocks[iBlock]-2;
        int indexMap=StartRow[iBlock];
        int chStart=MappingBlock[indexMap];
        int chEnd=MappingBlock[indexMap+1];
        int *Row=MappingBlock+StartRow[iBlock]+2;

        float Umean=0;
        float Vmean=0;
        float Wmean=0;
        float FreqMean=0;
        int NVisThisblock=0;
	//zero out the average visibility accumulators for this block:
        for(ThisPol =0; ThisPol<nVisCorr; ThisPol++) {
            VisCorr[ThisPol]=0;
            VisMeas[ThisPol]=0;
	    BlockVisWeight[ThisPol]=0;
	    BlockSumJones[ThisPol]=0;
	    BlockSumSqWeights[ThisPol]=0;
	    for(visChan=0; visChan<nVisChan; visChan++) {
	      size_t weightOff = ThisPol*nVisChan+visChan;
	      ThisSumJonesChan[weightOff]=0;
	      ThisSumSqWeightsChan[weightOff]=0;
	    }
        }

        float visChanMean=0.;
        resetJonesServerCounter();

        for (inx=0; inx<NRowThisBlock; ++inx) {
            int irow = Row[inx];
            if(irow>nrows) {
                continue;
            }
            double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;
            
            WeightVaryJJ=1.;

            float DeCorrFactor=1.;
            if(DoDecorr) {
                int iRowMeanThisBlock=Row[NRowThisBlock/2];

                double*  __restrict__ uvwPtrMidRow   = p_float64(uvw) + iRowMeanThisBlock*3;
                double*  __restrict__ uvw_dt_PtrMidRow   = uvw_dt_Ptr + iRowMeanThisBlock*3;

                DeCorrFactor=GiveDecorrelationFactor(DoSmearFreq,DoSmearTime,
                                                     (float)l0, (float)m0,
                                                     uvwPtrMidRow,
                                                     uvw_dt_PtrMidRow,
                                                     (float)FreqMean0,
                                                     (float)Dnu,
                                                     (float)DT);
            }
	    //Average all the channels within this bda block together.
	    //Since the Jones terms may vary faster than the bda smearing, the
	    //bda must happen on the fly.
            for (visChan=chStart; visChan<chEnd; ++visChan) {
                int doff = (irow * nVisChan + visChan) * nVisCorr;
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
                if(ChanEquidistant) {
                    if(visChan==0) {
                        float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;
                        CurrentCorrTerm[inx]=cexp(-UVNorm*(U*l0+V*m0+W*n0));
                        float complex dUVNorm=2.*I*PI*(Pfreqs[1]-Pfreqs[0])/C;
                        dCorrTerm[inx]=cexp(-dUVNorm*(U*l0+V*m0+W*n0));
                    } else {
                        CurrentCorrTerm[inx]*=dCorrTerm[inx];
                    }
                    facetPhasor=CurrentCorrTerm[inx];
                }
                else {
                    float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;
                    facetPhasor=cexp(-UVNorm*(U*l0+V*m0+W*n0));
                }
		//#######################################################
		
		//###################### Apply Jones #######################
                if(DoApplyJones) {
                    updateJones(irow, visChan, uvwPtr, 1);
                } //endif DoApplyJones
		//beam-weight the psf:
                if (dopsf==1) {
		    //We want a PSF for every possible correlation
		    for (ThisPol = 0; ThisPol < nVisCorr; ++ThisPol){
		      VisMeas[ThisPol]= VisCorrPSF[ThisPol];
		    }
                    facetPhasor=1.;
                    if(DoApplyJones) {
                        MatDot(J0,JonesType,VisMeas,SkyType,VisMeas);
                        MatDot(VisMeas,SkyType,J1H,JonesType,VisMeas);
                    }
                    if(DoDecorr) {
                        for(ThisPol =0; ThisPol<nVisCorr; ThisPol++) {
                            VisMeas[ThisPol]*=DeCorrFactor;
                        }
                    }
                } else {
		    float complex* __restrict__ visPtrMeas  = p_complex64(vis)  + doff;
                    for(ThisPol =0; ThisPol<nVisCorr; ThisPol++) {
                        VisMeas[ThisPol]=visPtrMeas[ThisPol];
                    }
                }
		
		if(DoApplyJones) {
		    float complex visPtr[4];
                    MatDot(J0H,JonesType,VisMeas,SkyType,visPtr);
                    MatDot(visPtr,SkyType,J1,JonesType,visPtr);
		    {
		      size_t corr;
		      for (corr=0;corr<nVisCorr;++corr)
			VisMeas[corr] = visPtr[corr];
		    }
                } 
                //#######################################################
		
		//###################### Flagging #######################
		//Weights flagged visibilities down to 0. Each correlation
		//may have its' own flag, so this term is not scalar and
		//should be viewed as being applied per baseline.
                bool* __restrict__ flagPtr = p_bool(flags) + doff;
		{
		  int corr;
		  for (corr = 0; corr < nVisCorr; ++corr)
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
		double* imgWtPtr = p_float64(weights) + irow  * nVisChan + visChan;
		{
		  int corr;
		  for (corr = 0; corr < nVisCorr; ++corr){
		    VisRealWeight[corr] = (*imgWtPtr)*WeightVaryJJ*DeCorrFactor*VisFlagWeight[corr];
		    VisComplexWeight[corr] = VisRealWeight[corr] * facetPhasor;
		    BlockVisWeight[corr]+=VisRealWeight[corr];
		    VisCorr[corr] += VisMeas[corr]*VisComplexWeight[corr];
		  }
		}
		//#######################################################

                U+=W*Cu;
                V+=W*Cv;
                //###################### Averaging #######################
                Umean+=U;
                Vmean+=V;
                Wmean+=W;
		if(DoApplyJones) {
		  int corr;
		  size_t jones_weight_offset;
		  for (corr=0;corr<nVisCorr;++corr){
		    float FWeightSq=(VisRealWeight[corr])*(VisRealWeight[corr]);
		    BlockSumJones[corr]+=BB*FWeightSq;
		    BlockSumSqWeights[corr]+=FWeightSq;
		    size_t weightsOffset = visChan*nVisCorr + corr;
		    ThisSumJonesChan[weightsOffset]+=BB*FWeightSq;
		    ThisSumSqWeightsChan[weightsOffset]+=FWeightSq;
		  }
		}//DoApplyJones 
                FreqMean+=factorFreq*(float)Pfreqs[visChan];
                visChanMean+=p_ChanMapping[visChan];
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
        int ThisGridChan=p_ChanMapping[chStart];
        float diffChan=visChanMean-visChanMean;
        if(diffChan!=0.) {
            printf("gridder: probably there is a problem in the BDA mapping: (ChanMean, ThisGridChan, diff)=(%f, %i, %f)\n",visChanMean,ThisGridChan,diffChan);
	    cexcept("Check your BDA mapping");
        }
	//Use the mean (smeared) frequency for uv scaling in this block:
        int gridChan = p_ChanMapping[chStart];
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
        int iwplane = floor((NwPlanes-1)*abs(wcoord)*(WaveRefWave/ThisWaveLength)/wmax+0.5);
        int skipW=0;
        if(iwplane>NwPlanes-1) {
            skipW=1;
            continue;
        };
        if(wcoord>0) {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);
        } else {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
        }
        int nConvX = cfs->dimensions[0];
        int nConvY = cfs->dimensions[1];
        int supx = (nConvX/OverS-1)/2;
        int supy = (nConvY/OverS-1)/2;
        int SupportCF=nConvX/OverS;
        // ################################################
	
	// ############## Start Gridding visibility #######
        if (gridChan >= 0  &&  gridChan < nGridChan) {
            double posx,posy;
            //For Even/Odd take the -1 off
            posx = uvwScale_p[0] * Umean * recipWvl + offset_p[0];//#-1;
            posy = uvwScale_p[1] * Vmean * recipWvl + offset_p[1];//-1;

            int locx = nint (posx);    // round to nearest location in grid
            int locy = nint (posy);
            double diffx = locx - posx;
            double diffy = locy - posy;
            int offx = nint (diffx * sampx); // location in
            int offy = nint (diffy * sampy); // oversampling
            offx += (nConvX-1)/2;
            offy += (nConvY-1)/2;
            //support and oversampling factors should be ints:
            int fsampx = nint (sampx);
            int fsampy = nint (sampy);
            int fsupx  = nint (supx);
            int fsupy  = nint (supy);

            // Only use visibility point if the full support is within grid.
            if (locx-supx >= 0  &&  locx+supx < nGridX  &&
                    locy-supy >= 0  &&  locy+supy < nGridY) {

                int ipol;
                for ( ipol=0; ipol<nGridPol; ++ipol ) {
                    float complex VisVal = VisStokes[ipol];
                    // Map to grid polarization. Only use pol if needed.
                    int goff = ( gridChan*nGridPol + ipol ) * nGridX * nGridY;
                    int sy;
                    float complex* __restrict__ gridPtr;
                    const float complex* __restrict__ cf0;
                    int io= ( offy - fsupy*fsampy );
                    int jo= ( offx - fsupx*fsampx );
                    int cfoff = io * OverS * SupportCF*SupportCF + jo * SupportCF*SupportCF;
                    cf0 =  p_complex64 ( cfs ) + cfoff;
                    for ( sy=-fsupy; sy<=fsupy; ++sy ) {
                        gridPtr =  p_complex64 ( grid ) + goff + ( locy+sy ) *nGridX + locx-supx;
                        int sx;
                        for ( sx=-fsupx; sx<=fsupx; ++sx ) {
                            *gridPtr++ += VisVal * *cf0;
                            cf0 ++;
                        }

                    }
                    
                    //Accumulate normalization weights for this facet
                    sumWtPtr[gridChan*nGridPol + ipol] += BlockVisWeight[ipol];
                    if ( DoApplyJones ) {
			//TODO: This has to be changed to include all polarizations
                        ptrSumJones[gridChan]+=BlockSumJones[0];
                        ptrSumJones[gridChan+nGridChan]+=BlockSumSqWeights[0];

                        for ( visChan=0; visChan<nVisChan; visChan++ ) {
			    //TODO: This has to be changed to include all polarizations
			    size_t weightOffset = visChan*nVisCorr + 0;
                            ptrSumJonesChan[visChan]+=ThisSumJonesChan[weightOffset];
                            ptrSumJonesChan[nVisChan+visChan]+=ThisSumSqWeightsChan[weightOffset];
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
    free(VisCorrDesc);
    free(VisStokesDesc);
    free(ThisSumJonesChan);
    free(ThisSumSqWeightsChan);
    free(dCorrTerm);
    free(VisFlagWeight);
    free(VisRealWeight);
    free(VisComplexWeight);
    free(BlockVisWeight);
    free(BlockSumJones);
    free(BlockSumSqWeights);
    free_stokes_library();

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

void DeGridderWPol(PyArrayObject *grid,
                   PyArrayObject *vis,
                   PyArrayObject *uvw,
                   PyArrayObject *flags,
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
                   PyArrayObject *np_ChanMapping)
{
    // Get size of convolution functions.
    PyArrayObject *cfs;
    PyArrayObject *NpPolMap;
    NpPolMap = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 0), PyArray_INT32, 0, 4);
    int npolsMap=NpPolMap->dimensions[0];
    int* PolMap=I_ptr(NpPolMap);

    int *p_ChanMapping=p_int32(np_ChanMapping);

    PyArrayObject *NpFacetInfos;
    NpFacetInfos = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 1), PyArray_FLOAT64, 0, 4);

    PyArrayObject *NpRows;
    NpRows = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 2), PyArray_INT32, 0, 4);
    int* ptrRows=I_ptr(NpRows);
    int row0=ptrRows[0];
    int row1=ptrRows[1];


    /////////////////////////////////////////
    int LengthSmearingList=PyList_Size(LSmearing);
    float DT,Dnu;
    double* uvw_dt_Ptr;
    int DoSmearTime,DoSmearFreq;
    int DoDecorr=(LengthSmearingList>0);

    if(DoDecorr) {
        uvw_dt_Ptr = p_float64((PyArrayObject *) PyList_GetItem(LSmearing, 0));

        PyObject *_FDT= PyList_GetItem(LSmearing, 1);
        DT=(float) (PyFloat_AsDouble(_FDT));
        PyObject *_FDnu= PyList_GetItem(LSmearing, 2);
        Dnu=(float) (PyFloat_AsDouble(_FDnu));

        PyObject *_DoSmearTime= PyList_GetItem(LSmearing, 3);
        DoSmearTime=(int) (PyFloat_AsDouble(_DoSmearTime));

        PyObject *_DoSmearFreq= PyList_GetItem(LSmearing, 4);
        DoSmearFreq=(int) (PyFloat_AsDouble(_DoSmearFreq));

    }
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////




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
    double C=2.99792456e8;
    int inx;


    // ################### Prepare full scalar mode

    PyObject *_JonesType  = PyList_GetItem(LOptimisation, 0);
    int JonesType=(int) PyFloat_AsDouble(_JonesType);
    PyObject *_ChanEquidistant  = PyList_GetItem(LOptimisation, 1);
    int ChanEquidistant=(int) PyFloat_AsDouble(_ChanEquidistant);

    PyObject *_SkyType  = PyList_GetItem(LOptimisation, 2);
    int SkyType=(int) PyFloat_AsDouble(_SkyType);

    PyObject *_PolMode  = PyList_GetItem(LOptimisation, 3);
    int PolMode=(int) PyFloat_AsDouble(_PolMode);

    int ipol;
    if(PolMode==0) {
        for (ipol=1; ipol<4; ++ipol) {
            PolMap[ipol]=5;
        }
    }

    /* PyObject *_FullScalarMode  = PyList_GetItem(LOptimisation, 0); */
    /* FullScalarMode=(int) PyFloat_AsDouble(_FullScalarMode); */
    /* PyObject *_ChanEquidistant  = PyList_GetItem(LOptimisation, 1); */
    /* int ChanEquidistant=(int) PyFloat_AsDouble(_ChanEquidistant); */
    /* ScalarJones=0; */
    /* ScalarVis=0; */
    /* int nPolJones = 4; */
    /* int nPolVis   = 4; */
    /* if(FullScalarMode){ */
    /*   //printf("full scalar mode\n"); */
    /*   //printf("ChanEquidistant: %i\n",ChanEquidistant); */
    /*   ScalarJones=1; */
    /*   ScalarVis=1; */
    /*   nPolJones=1; */
    /*   nPolVis=1; */
    /*   int ipol; */
    /*   for (ipol=1; ipol<nVisPol; ++ipol) { */
    /* 	PolMap[ipol]=5; */
    /*   } */
    /* } */




    int *MappingBlock = p_int32(SmearMapping);
    int NTotBlocks=MappingBlock[0];
    int *NRowBlocks=MappingBlock+1;
    int *StartRow=MappingBlock+1+NTotBlocks;
    int iBlock;

    int NMaxRow=0;
    for(iBlock=0; iBlock<NTotBlocks; iBlock++) {
        int NRowThisBlock=NRowBlocks[iBlock]-2;
        if(NRowThisBlock>NMaxRow) {
            NMaxRow=NRowThisBlock;
        }
    }
    float complex *CurrentCorrTerm=(float complex *)calloc(1,(NMaxRow)*sizeof(float complex));
    float complex *dCorrTerm=(float complex *)calloc(1,(NMaxRow)*sizeof(float complex));
    // ########################################################


    double posx,posy;

    double WaveLengthMean=0.;
    int visChan;
    for (visChan=0; visChan<nVisChan; ++visChan) {
        WaveLengthMean+=C/Pfreqs[visChan];
    }
    WaveLengthMean/=nVisChan;


    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    initJonesServer(LJones,JonesType,WaveLengthMean);
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    for(iBlock=0; iBlock<NTotBlocks; iBlock++) {
        //for(iBlock=3507; iBlock<3508; iBlock++){
        int NRowThisBlock=NRowBlocks[iBlock]-2;
        int indexMap=StartRow[iBlock];
        int chStart=MappingBlock[indexMap];
        int chEnd=MappingBlock[indexMap+1];
        int *Row=MappingBlock+StartRow[iBlock]+2;

        float complex Vis[4]= {0};
        float Umean=0;
        float Vmean=0;
        float Wmean=0;
        float FreqMean=0;
        int NVisThisblock=0;
        //printf("\n");
        //printf("Block[%i] Nrows=%i %i>%i\n",iBlock,NRowThisBlock,chStart,chEnd);

        float visChanMean=0.;
        resetJonesServerCounter();
        for (inx=0; inx<NRowThisBlock; inx++) {
            int irow = Row[inx];
            if(irow>nrows) {
                continue;
            }
            double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;
            //printf("[%i] %i>%i bl=(%i-%i)\n",irow,chStart,chEnd,ptrA0[irow],ptrA1[irow]);
            //printf("  row=[%i] %i>%i \n",irow,chStart,chEnd);

            int ThisPol;
            for (visChan=chStart; visChan<chEnd; ++visChan) {
                int doff = (irow * nVisChan + visChan) * nVisPol;
                bool* __restrict__ flagPtr = p_bool(flags) + doff;
                int OneFlagged=0;
                int cond;
                //char ch="a";
                //if(flagPtr[0]==1){OneFlagged=1;}
                //if(OneFlagged){continue;}

                float U=(float)uvwPtr[0];
                float V=(float)uvwPtr[1];
                float W=(float)uvwPtr[2];

                U+=W*Cu;
                V+=W*Cv;
                //###################### Averaging #######################
                Umean+=U;
                Vmean+=V;
                Wmean+=W;
                FreqMean+=(float)Pfreqs[visChan];
                visChanMean+=p_ChanMapping[visChan];
                NVisThisblock+=1;
                //printf("      [%i,%i], fmean=%f %f\n",inx,visChan,(FreqMean/1e6),Pfreqs[visChan]);

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
        int ThisGridChan=p_ChanMapping[chStart];
        float diffChan=visChanMean-visChanMean;


        //if(diffChan!=0.){printf("degridder: probably there is a problem in the BDA mapping\n");}
        if(diffChan!=0.) {
            printf("degridder: probably there is a problem in the BDA mapping: (ChanMean, ThisGridChan, diff)=(%f, %i, %f)\n",visChanMean,ThisGridChan,diffChan);
        }
        visChanMean=0.;
        //printf("  iblock: %i [%i], (uvw)=(%f, %f, %f) fmean=%f\n",iBlock,NVisThisblock,Umean,Vmean,Wmean,(FreqMean/1e6));
        /* int ThisPol; */
        /* for(ThisPol =0; ThisPol<4;ThisPol++){ */
        /* 	printf("   vis: %i (%f, %f)\n",ThisPol,creal(Vis[ThisPol]),cimag(Vis[ThisPol])); */
        /* } */


        // ################################################
        // ############## Start Gridding visibility #######
        int gridChan = p_ChanMapping[chStart];//0;//chanMap_p[visChan];

        int CFChan = 0;//ChanCFMap[visChan];
        double recipWvl = FreqMean / C;
        double ThisWaveLength=C/FreqMean;

        // ############## W-projection ####################
        double wcoord=Wmean;
        /* int iwplane = floor((NwPlanes-1)*abs(wcoord)*(WaveRefWave/ThisWaveLength)/wmax); */
        /* int skipW=0; */
        /* if(iwplane>NwPlanes-1){skipW=1;continue;}; */

        /* if(wcoord>0){ */
        /* 	cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2); */
        /* } else{ */
        /* 	cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2); */
        /* } */
        /* int nConvX = cfs->dimensions[0]; */
        /* int nConvY = cfs->dimensions[1]; */
        /* int supx = (nConvX/OverS-1)/2; */
        /* int supy = (nConvY/OverS-1)/2; */
        /* int SupportCF=nConvX/OverS; */
        /* // ################################################ */


        int iwplane = floor((NwPlanes-1)*abs(wcoord)*(WaveRefWave/ThisWaveLength)/wmax+0.5);
        int skipW=0;
        if(iwplane>NwPlanes-1) {
            skipW=1;
            continue;
        };

        //int iwplane = floor((NwPlanes-1)*abs(wcoord)/wmax);

        //printf("wcoord=%f, iw=%i \n",wcoord,iwplane);

        if(wcoord>0) {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(Lcfs, iwplane), PyArray_COMPLEX64, 0, 2);
        } else {
            cfs=(PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
        }
        int nConvX = cfs->dimensions[0];
        int nConvY = cfs->dimensions[1];
        int supx = (nConvX/OverS-1)/2;
        int supy = (nConvY/OverS-1)/2;
        int SupportCF=nConvX/OverS;



        if (gridChan >= 0  &&  gridChan < nGridChan) {
            double posx,posy;
            //For Even/Odd take the -1 off
            posx = uvwScale_p[0] * Umean * recipWvl + offset_p[0];//#-1;
            posy = uvwScale_p[1] * Vmean * recipWvl + offset_p[1];//-1;

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
            if (locx-supx >= 0  &&  locx+supx < nGridX  &&
                    locy-supy >= 0  &&  locy+supy < nGridY) {
                ///            cout << "in grid"<<endl;
                // Get pointer to data and flags for this channel.
                //int doff = (irow * nVisChan + visChan) * nVisPol;
                //float complex* __restrict__ visPtr  = p_complex64(vis)  + doff;
                //bool* __restrict__ flagPtr = p_bool(flags) + doff;
                float complex ThisVis[4]= {0};

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
                    //if (((int)flagPtr[ipol])==0) {
                    // Map to grid polarization. Only use pol if needed.
                    int gridPol = PolMap[ipol];
                    if (gridPol >= 0  &&  gridPol < nGridPol) {

                        int goff = (gridChan*nGridPol + gridPol) * nGridX * nGridY;
                        int sy;

                        const float complex* __restrict__ gridPtr;
                        const float complex* __restrict__ cf0;

                        int io=(offy - fsupy*fsampy);
                        int jo=(offx - fsupx*fsampx);
                        int cfoff = io * OverS * SupportCF*SupportCF + jo * SupportCF*SupportCF;
                        cf0 =  p_complex64(cfs) + cfoff;




                        for (sy=-fsupy; sy<=fsupy; ++sy) {
                            gridPtr =  p_complex64(grid) + goff + (locy+sy)*nGridX + locx-supx;
                            int sx;
                            for (sx=-fsupx; sx<=fsupx; ++sx) {
                                ThisVis[ipol] += *gridPtr  * *cf0;
                                cf0 ++;
                                gridPtr++;
                            }
                        }
                    } // end if gridPol



                    //} // end if !flagPtr
                } // end for ipol

                // ###########################################################
                // ################### Now do the correction #################

                if(PolMode==0) {
                    ThisVis[3]=ThisVis[0];
                }


                ///////////////////////////////////////////////////////
                float DeCorrFactor=1.;
                if(DoDecorr) {
                    int iRowMeanThisBlock=Row[NRowThisBlock/2];

                    double*  __restrict__ uvwPtrMidRow   = p_float64(uvw) + iRowMeanThisBlock*3;
                    double*  __restrict__ uvw_dt_PtrMidRow   = uvw_dt_Ptr + iRowMeanThisBlock*3;

                    DeCorrFactor=GiveDecorrelationFactor(DoSmearFreq,DoSmearTime,
                                                         (float)l0, (float)m0,
                                                         uvwPtrMidRow,
                                                         uvw_dt_PtrMidRow,
                                                         (float)FreqMean,
                                                         (float)Dnu,
                                                         (float)DT);

                    //printf("DeCorrFactor %f %f: %f\n",l0,m0,DeCorrFactor);

                }
                //////////////////////////////////////////////////////

                for (inx=0; inx<NRowThisBlock; inx++) {
                    int irow = Row[inx];
                    if(irow>nrows) {
                        continue;
                    }
                    double*  __restrict__ uvwPtr   = p_float64(uvw) + irow*3;
                    //printf("[%i] %i>%i bl=(%i-%i)\n",irow,chStart,chEnd,ptrA0[irow],ptrA1[irow]);
                    //printf("  row=[%i] %i>%i \n",irow,chStart,chEnd);


                    int ThisPol;
                    for (visChan=chStart; visChan<chEnd; ++visChan) {
                        int doff = (irow * nVisChan + visChan) * nVisPol;
                        bool* __restrict__ flagPtr = p_bool(flags) + doff;
                        int OneFlagged=0;
                        int cond;
                        //if(flagPtr[0]==1){OneFlagged=1;}
                        //if(OneFlagged){continue;}

                        if(DoApplyJones) {
                            updateJones(irow, visChan, uvwPtr, 0);
                        } //endif DoApplyJones

                        //###################### Facetting #######################
                        // Change coordinate and shift visibility to facet center
                        float U=(float)uvwPtr[0];
                        float V=(float)uvwPtr[1];
                        float W=(float)uvwPtr[2];
                        //AddTimeit(PreviousTime,TimeShift);
                        //#######################################################

                        float complex corr;
                        if(ChanEquidistant) {
                            if(visChan==0) {
                                float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;
                                CurrentCorrTerm[inx]=cexp(UVNorm*(U*l0+V*m0+W*n0));
                                float complex dUVNorm=2.*I*PI*(Pfreqs[1]-Pfreqs[0])/C;
                                dCorrTerm[inx]=cexp(dUVNorm*(U*l0+V*m0+W*n0));
                            } else {
                                CurrentCorrTerm[inx]*=dCorrTerm[inx];
                            }
                            corr=CurrentCorrTerm[inx];
                        }
                        else {
                            float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C;
                            corr=cexp(UVNorm*(U*l0+V*m0+W*n0));
                        }
                        /* float complex UVNorm=2.*I*PI*Pfreqs[visChan]/C; */
                        /* corr=cexp(-UVNorm*(U*l0+V*m0+W*n0)); */


                        corr*=DeCorrFactor;

                        float complex* __restrict__ visPtr  = p_complex64(vis)  + doff;
                        float complex visBuff[4]= {0};


                        if(DoApplyJones) {
                            MatDot(J0,JonesType,ThisVis,SkyType,visBuff);
                            MatDot(visBuff,SkyType,J1H,JonesType,visBuff);
                        } else {
                            for(ThisPol =0; ThisPol<4; ThisPol++) {
                                visBuff[ThisPol]=ThisVis[ThisPol];
                            }

                        }

                        Mat_A_l_SumProd(visBuff, SkyType, corr);

                        Mat_A_Bl_Sum(visPtr, 2, visBuff, 2, (float complex)(-1.));





                    }//endfor vischan
                }//endfor RowThisBlock



            } // end if ongrid
        } // end if gridChan

    } //end for Block

} // end