#pragma once
#include <complex.h>
#include <Python.h>
#include <stdint.h>
typedef struct{
  //Smearing information
  double* uvw_dt_Ptr;
  int LengthSmearingList;
  float DT,Dnu;
  int DoSmearTime,DoSmearFreq;
  int DoDecorr;
  //Faceting information
  double Cu,Cv,l0,m0,n0;
  //Wprojection filter information
  double WaveRefWave,wmax,NwPlanes;
  int OverS,sampx,sampy;
  PyObject * Lcfs; // implicit support size per w. Lower layers will need smaller support
  PyObject * LcfsConj; //Conjugate w planes
  int * nConvX, nConvY; //filter length of each of the #WPlanes-worth of filters
  //Grid dimensions, offsets and uv scales
  float _Complex * grid;
  int nGridX,nGridY,nGridPol,nGridChan;
  double incr[2];
  //Data columns
  bool* flags;
  float _Complex * vis;
  bool dopsf;
  double * uvw;
  double * weights;
  //uv frequency information
  double WaveLengthMean;
  double FreqMean0;
  double * Pfreqs; //ref frequencies
  //uv data dimensions
  int nrows;
  int nVisCorr;
  int nVisChan;
  //mappings
  int *p_ChanMapping;
  int32_t *VisCorrDesc;
  int32_t *VisStokesDesc;
  //weight arrays
  double* __restrict__ sumWtPtr;
  //optimizations
  int JonesType;
  int ChanEquidistant;
  int SkyType;
  //BDA information
  int *MappingBlock;
  int *NRowBlocks;
  int *StartRow;
  int NTotBlocks;
  int NMaxRow;
} gridding_parameters;