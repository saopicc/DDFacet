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
#include "common.h"
#include "Semaphores.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

namespace {

#include "JonesServer.h"
#include "Stokes.h"

void FATAL(const string &msg) { cerr << "FATAL: " << msg << endl; exit(1); }

double GiveDecorrelationFactor(bool FSmear, bool TSmear, double l0,
  double m0, const double *uvwPtr, const double *uvw_dt_Ptr, double nu,
  double Dnu, double DT)
  {
  double n0=sqrt(1.-l0*l0-m0*m0)-1.;
  double DecorrFactor=1.;

  if (FSmear)
    {
    double phase = uvwPtr[0]*l0 + uvwPtr[1]*m0 + uvwPtr[2]*n0;
    double phi=PI*Dnu/C*phase;
    if (phi!=0.)
      DecorrFactor*=max(0.,sin(phi)/phi);
    }

  if (TSmear)
    {
    double dphase = (uvw_dt_Ptr[0]*l0 + uvw_dt_Ptr[1]*m0 + uvw_dt_Ptr[2]*n0)*DT;
    double phi=PI*nu/C*dphase;
    if (phi!=0.)
      DecorrFactor*=max(0.,sin(phi)/phi);
    }
  return DecorrFactor;
  }

PyObject *pyAccumulateWeightsOntoGrid(PyObject */*self*/, PyObject *args)
  {
  PyArrayObject *grid, *weights, *index;
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type,  &grid,
                        &PyArray_Type, &weights, &PyArray_Type, &index))
    return NULL;

  double *pgrid = p_float64(grid);
  const float *pweights = p_float32(weights);
  const long int *pindex = p_int64(index);
  size_t n = size_t(weights->dimensions[0]);

  for(size_t i=0; i<n; i++)
    {
    size_t igrid = size_t(pindex[i]);
    double w = pweights[i];
    if (w!=0)
      {
      sem_t *psem = GiveSemaphoreFromCell(igrid);
      sem_wait(psem);
      pgrid[igrid] += w;
      sem_post(psem);
      }
    }

  Py_RETURN_NONE;
  }

//////////////////////////////////////////////////////////////////////
using ReadCorrType = void (*) (const fcmplx *visPtrMeas, dcMat &VisMeas);

void Read_4Corr(const fcmplx *visPtrMeas, dcMat &VisMeas)
  {
  for (auto i=0; i<4; ++i)
    VisMeas[i]=visPtrMeas[i];
  }
void Read_2Corr_Pad(const fcmplx *visPtrMeas, dcMat &VisMeas)
  {
  VisMeas[0]=visPtrMeas[0];
  VisMeas[1]=VisMeas[2]=0;
  VisMeas[3]=visPtrMeas[1];
  }

using MulaccumType = void (*) (const dcMat &VisMeas, dcmplx Weight, dcMat &Vis);

void Mulaccum_4Corr(const dcMat &VisMeas, dcmplx Weight, dcMat &Vis)
  {
  for (auto i=0; i<4; ++i)
    Vis[i] += VisMeas[i]*Weight;
  }
void Mulaccum_2Corr_Unpad(const dcMat &VisMeas, dcmplx Weight, dcMat &Vis)
  {
  Vis[0] += VisMeas[0]*Weight;
  Vis[1] += VisMeas[3]*Weight;
  }

template <typename T> bool contains(const std::vector<T>& Vec, const T &Element)
  { return find(Vec.begin(), Vec.end(), Element) != Vec.end(); }

void getPolMap(const vector<string> &stokes, int *PolMap)
  {
  int ii=0,iq=0,iu=0,iv=0;
  if (contains<string>(stokes,string("I"))) { ++iq; ++iu; ++iv; }
  if (contains<string>(stokes,string("Q"))) { ++iu; ++iv; }
  if (contains<string>(stokes,string("U"))) { ++iv; }
  for (size_t n=0; n<stokes.size(); ++n)
    {
    if (stokes[n]=="I") PolMap[n]=ii;
    else if (stokes[n]=="Q") PolMap[n]=iq;
    else if (stokes[n]=="U") PolMap[n]=iu;
    else if (stokes[n]=="V") PolMap[n]=iv;
    }
  }

vector<string> sortStokes(const vector<string> &stokes)
  {
  vector<string> res;
  if (contains<string>(stokes,"I")) res.push_back("I");
  if (contains<string>(stokes,"Q")) res.push_back("Q");
  if (contains<string>(stokes,"U")) res.push_back("U");
  if (contains<string>(stokes,"V")) res.push_back("V");
  return res;
  }

template<ReadCorrType readcorr, MulaccumType mulaccum, StokesGridType stokesgrid>
void gridder(
  PyArrayObject *grid,
  PyArrayObject *vis,
  PyArrayObject *uvw,
  PyArrayObject *flags,
  PyArrayObject *weights,
  PyArrayObject *sumwt,
  bool dopsf,
  PyObject *Lcfs,
  PyObject *LcfsConj,
  PyArrayObject *Winfos,
  PyArrayObject *increment,
  PyArrayObject *freqs,
  PyObject *Lmaps,
  PyObject *LJones,
  PyArrayObject *SmearMapping,
  PyArrayObject *Sparsification,
  PyObject *LOptimisation,
  PyObject *LSmearing,
  PyArrayObject *np_ChanMapping,
  const vector<string> &expstokes)
  {
  auto nVisPol = expstokes.size();
  int PolMap[4];
  getPolMap(expstokes, PolMap);
  double DT=-1e30, Dnu=-1e30, lmin_decorr=-1e30, mmin_decorr=-1e30;
  const double* uvw_dt_Ptr=0;
  bool DoSmearTime=false, DoSmearFreq=false;
  bool DoDecorr=(PyList_Size(LSmearing)>0);
  if (DoDecorr)
    {
    uvw_dt_Ptr = p_float64((PyArrayObject *)PyList_GetItem(LSmearing, 0));
    DT = PyFloat_AsDouble(PyList_GetItem(LSmearing, 1));
    Dnu = PyFloat_AsDouble(PyList_GetItem(LSmearing, 2));
    DoSmearTime = bool(PyFloat_AsDouble(PyList_GetItem(LSmearing, 3)));
    DoSmearFreq = bool(PyFloat_AsDouble(PyList_GetItem(LSmearing, 4)));
    lmin_decorr = PyFloat_AsDouble(PyList_GetItem(LSmearing, 5));
    mmin_decorr = PyFloat_AsDouble(PyList_GetItem(LSmearing, 6));
    }

  const double *ptrFacetInfos=p_float64((PyArrayObject *) PyList_GetItem(Lmaps, 1));
  double Cu=ptrFacetInfos[0];
  double Cv=ptrFacetInfos[1];
  double l0=ptrFacetInfos[2];
  double m0=ptrFacetInfos[3];
  double n0=sqrt(1-l0*l0-m0*m0)-1;

  /* Get size of grid. */
  double *ptrWinfo = p_float64(Winfos);
  double WaveRefWave = ptrWinfo[0];
  double wmax = ptrWinfo[1];
  double NwPlanes = ptrWinfo[2];
  int OverS=int(floor(ptrWinfo[3]));

  int nGridX    = int(grid->dimensions[3]);
  int nGridY    = int(grid->dimensions[2]);
  int nGridPol  = int(grid->dimensions[1]);
  int nGridChan = int(grid->dimensions[0]);

  /* Get visibility data size. */
  size_t nVisCorr = size_t(flags->dimensions[2]);
  size_t nVisChan = size_t(flags->dimensions[1]);
  size_t nrows    = size_t(uvw->dimensions[0]);

  double* __restrict__ sumWtPtr = p_float64(sumwt);

  /* MR FIXME: should this be "/2" or "/2."? */
  const double offset_p[] = {double(nGridX/2), double(nGridY/2)};
  const double *incr=p_float64(increment);
  const double *Pfreqs=p_float64(freqs);
  /* MR FIXME: should the second entry depend on nGridY instead of nGridX? */
  const double uvwScale_p[]= {nGridX*incr[0], nGridX*incr[1]};

  bool ChanEquidistant= bool(PyFloat_AsDouble(PyList_GetItem(LOptimisation, 1)));

  const int *MappingBlock = p_int32(SmearMapping);
  /* total size is in two words */
  size_t NTotBlocks = size_t(MappingBlock[0]) + (size_t(MappingBlock[1])<<32);
  const int *NRowBlocks = MappingBlock+2;
  const int *StartRow = MappingBlock+2+NTotBlocks;

  /* in sparsification mode, the Sparsification argument is an array of length NTotBlocks flags. */
  /* Only blocks with a True flag will be gridded. */
  const bool *sparsificationFlag = 0;
  if (PyArray_Size((PyObject*)Sparsification))
    {
    if (size_t(PyArray_Size((PyObject*)Sparsification)) != NTotBlocks)
      {
      PyErr_SetString(PyExc_TypeError, "sparsification argument must be an array of length NTotBlocks");
      return;
      }
    sparsificationFlag = p_bool(Sparsification);
    }

  size_t NMaxRow=0;
  for (size_t iBlock=0; iBlock<NTotBlocks; iBlock++)
    {
    if (sparsificationFlag && !sparsificationFlag[iBlock])
      continue;
    NMaxRow = max(NMaxRow, size_t(NRowBlocks[iBlock]-2));
    }
  /* these are used for equidistant channels: one holds the phase term in channel 0, */
  /* the other one holds the delta-phase across channels */
  vector<dcmplx> CurrentCorrTerm(NMaxRow), dCorrTerm(NMaxRow);
  /* and this indicates for which channel the CurrentCorrTerm is currently computed */
  vector<int> CurrentCorrChan(NMaxRow);
  int CurrentCorrRow0 = -1;
  /* ######################################################## */
  double WaveLengthMean=0., FreqMean0=0.;
  for (size_t visChan=0; visChan<nVisChan; ++visChan)
    {
    WaveLengthMean+=C/Pfreqs[visChan];
    FreqMean0+=Pfreqs[visChan];
    }
  WaveLengthMean/=(double)nVisChan;
  FreqMean0/=(double)nVisChan;

  JonesServer JS(LJones,WaveLengthMean);

  vector<double> ThisSumJonesChan(nVisChan), ThisSumSqWeightsChan(nVisChan);

  const int *p_ChanMapping=p_int32(np_ChanMapping);
  for (size_t iBlock=0; iBlock<NTotBlocks; iBlock++)
    {
    if (sparsificationFlag && !sparsificationFlag[iBlock])
      continue;

    int NRowThisBlock=NRowBlocks[iBlock]-2;
    size_t chStart = size_t(StartRow[0]),
           chEnd   = size_t(StartRow[1]);
    const int *Row = StartRow+2;
    /* advance pointer to next blocklist */
    StartRow += NRowBlocks[iBlock];

    dcMat Vis(0,0,0,0);

    for (size_t visChan=0; visChan<nVisChan; ++visChan)
      ThisSumJonesChan[visChan] = ThisSumSqWeightsChan[visChan] = 0;

    /* when moving to a new block of rows, init this to -1 so the code below knows to initialize*/
    /* CurrentCorrTerm when the first channel of each row comes in*/
    if (Row[0]!=CurrentCorrRow0)
      {
      for (auto inx=0; inx<NRowThisBlock; inx++)
        CurrentCorrChan[inx] = -1;
      CurrentCorrRow0 = Row[0];
      }

    double DeCorrFactor=1.;
    if (DoDecorr)
      {
      int iRowMeanThisBlock=Row[NRowThisBlock/2];

      const double* __restrict__ uvwPtrMidRow = p_float64(uvw) + iRowMeanThisBlock*3;
      const double* __restrict__ uvw_dt_PtrMidRow = uvw_dt_Ptr + iRowMeanThisBlock*3;

      DeCorrFactor=GiveDecorrelationFactor(DoSmearFreq,DoSmearTime,
        lmin_decorr, mmin_decorr, uvwPtrMidRow, uvw_dt_PtrMidRow,
        FreqMean0, Dnu, DT);
      }

    double visChanMean=0., Umean=0, Vmean=0, Wmean=0, FreqMean=0;
    double ThisWeight=0., ThisSumJones=0., ThisSumSqWeights=0.;
    int NVisThisblock=0;
    JS.resetJonesServerCounter();
    for (auto inx=0; inx<NRowThisBlock; inx++)
      {
      size_t irow = size_t(Row[inx]);
      if (irow>nrows) continue;
      const double* __restrict__ uvwPtr = p_float64(uvw) + irow*3;
      JS.WeightVaryJJ=1.;

      for (size_t visChan=chStart; visChan<chEnd; ++visChan)
        {
        size_t doff = size_t((irow*nVisChan + visChan) * nVisCorr);
        const float *imgWtPtr = p_float32(weights) + irow*nVisChan + visChan;

        /* We can do that since all flags in 4-pols are equalised in ClassVisServer */
        if (p_bool(flags)[doff]) continue;

        double U=uvwPtr[0];
        double V=uvwPtr[1];
        double W=uvwPtr[2];
        dcmplx corr;
        if (ChanEquidistant)
          {
          /* init correlation term for first channel that it's not initialized in */
          if (CurrentCorrChan[inx]==-1)
            {
            double angle = -2.*PI*(U*l0+V*m0+W*n0)/C;
            CurrentCorrTerm[inx] = polar(1.,Pfreqs[visChan]*angle);
            dCorrTerm[inx]       = polar(1.,(Pfreqs[1]-Pfreqs[0])*angle);
            CurrentCorrChan[inx] = int(visChan);
            }
          /* else, wind the correlation term forward by as many channels as necessary */
          /* this modification allows us to support blocks that skip across channels */
          else
            {
            while (size_t(CurrentCorrChan[inx]) < visChan)
              {
              CurrentCorrTerm[inx] *= dCorrTerm[inx];
              CurrentCorrChan[inx]++;
              }
            }
          corr = CurrentCorrTerm[inx];
          }
        else /* Not chan-equidistant */
          corr = polar(1.,-2.*PI*Pfreqs[visChan]/C*(U*l0+V*m0+W*n0));

        if (JS.DoApplyJones)
          JS.updateJones(irow, visChan, uvwPtr, true, true);

        dcMat VisMeas;
        if (dopsf)
          {
          corr=1.;
          if (JS.DoApplyJones)
            VisMeas=(JS.J0).times(JS.J1H); // MR FIXME: precompute?
          else
            VisMeas.setUnity();
          if (DoDecorr)
            for(int ThisPol=0; ThisPol<4;ThisPol++)
              VisMeas[ThisPol]*=DeCorrFactor;
          }
        else
          readcorr(p_complex64(vis)+doff, VisMeas);

        double FWeight = imgWtPtr[0]*JS.WeightVaryJJ;
        dcmplx Weight = FWeight*corr;
        if (JS.DoApplyJones)
          {
          VisMeas=(JS.J0H.times(VisMeas)).times(JS.J1);
          mulaccum(VisMeas, Weight, Vis);

          /*Compute per channel and overall approximate matrix sqroot:*/
          double FWeightSq=FWeight*DeCorrFactor*DeCorrFactor;
          ThisSumJones+=JS.BB*FWeightSq;
          ThisSumSqWeights+=FWeightSq;

          ThisSumJonesChan[visChan]+=JS.BB*FWeightSq;
          ThisSumSqWeightsChan[visChan]+=FWeightSq;
          }
        else /* Don't apply Jones */
          mulaccum(VisMeas, Weight, Vis);

        U+=W*Cu;
        V+=W*Cv;
        /*###################### Averaging #######################*/
        Umean+=U;
        Vmean+=V;
        Wmean+=W;
        FreqMean+=Pfreqs[visChan];
        ThisWeight+=FWeight;

        visChanMean+=p_ChanMapping[visChan];
        ++NVisThisblock;
        }/*endfor vischan*/
      }/*endfor RowThisBlock*/

    if (NVisThisblock==0) continue;

    visChanMean/=NVisThisblock;

    int gridChan = p_ChanMapping[chStart];
    double diffChan=visChanMean-gridChan;
    if(abs(diffChan)>1e-6)
      {
      printf("gridder: probably there is a problem in the BDA mapping: (ChanMean, gridChan, diff)=(%lf, %i, %lf)\n",visChanMean,gridChan,diffChan);
      for (size_t visChan=chStart; visChan<chEnd; ++visChan)
        printf("%d ", gridChan-p_ChanMapping[visChan]);
      printf("\n");
      }

    /* ################################################ */
    /* ######## Convert correlations to stokes ######## */
    dcMat stokes_vis;//[nVisPol];
    stokesgrid(Vis, stokes_vis);

    /* ################################################ */
    /* ############## Start Gridding visibility ####### */
    if (gridChan<0 || gridChan>=nGridChan) continue;

    Umean/=NVisThisblock;
    Vmean/=NVisThisblock;
    Wmean/=NVisThisblock;
    FreqMean/=NVisThisblock;
    double recipWvl = FreqMean / C;

    /* ############## W-projection #################### */
    int iwplane = (int)lrint((NwPlanes-1)*abs(Wmean)*(WaveRefWave*recipWvl)/wmax);
    if (iwplane>NwPlanes-1) continue;

    PyArrayObject *cfs=(PyArrayObject *) PyArray_ContiguousFromObject(
      PyList_GetItem((Wmean>0) ? Lcfs : LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
    int nConvX = (int)cfs->dimensions[0];
    int nConvY = (int)cfs->dimensions[1];
    int supx = (nConvX/OverS-1)/2;
    int supy = (nConvY/OverS-1)/2;
    int SupportCF=nConvX/OverS;

    double posx = uvwScale_p[0]*Umean*recipWvl + offset_p[0];
    double posy = uvwScale_p[1]*Vmean*recipWvl + offset_p[1];

    int locx = int(lrint(posx));    /* location in grid */
    int locy = int(lrint(posy));

    /* Only use visibility point if the full support is within grid. */
    if (locx-supx<0 || locx+supx>=nGridX || locy-supy<0 || locy+supy>=nGridY)
      continue;

    int offx = int(lrint((locx-posx)*OverS) + (nConvX-1)/2); /* location in */
    int offy = int(lrint((locy-posy)*OverS) + (nConvY-1)/2); /* oversampling */

    int io = offy - supy*OverS;
    int jo = offx - supx*OverS;
    int cfoff = (io*OverS + jo)*SupportCF*SupportCF;

    for (size_t ipol=0; ipol<nVisPol; ++ipol)
      {
      /* Map to grid polarization. Only use pol if needed.*/
      int gridPol = PolMap[ipol];
      if (gridPol<0 || gridPol>=nGridPol) continue;

      size_t goff = size_t((gridChan*nGridPol + gridPol) * nGridX*nGridY);
      dcmplx VisVal =stokes_vis[ipol];
      const fcmplx* __restrict__ cf0 = p_complex64(cfs) + cfoff;
      fcmplx* __restrict__ gridPtr = p_complex64(grid) + goff + (locy-supy)*nGridX + locx;
      for (int sy=-supy; sy<=supy; ++sy, gridPtr+=nGridX)
        for (int sx=-supx; sx<=supx; ++sx)
          gridPtr[sx] += VisVal * dcmplx(*cf0++);
      sumWtPtr[gridPol+gridChan*nGridPol] += ThisWeight;
      if (JS.DoApplyJones)
        {
        JS.ptrSumJones[gridChan]+=ThisSumJones;
        JS.ptrSumJones[gridChan+nGridChan]+=ThisSumSqWeights;

        for(size_t visChan=0; visChan<nVisChan; visChan++)
          {
          JS.ptrSumJonesChan[visChan]+=ThisSumJonesChan[visChan];
          JS.ptrSumJonesChan[nVisChan+visChan]+=ThisSumSqWeightsChan[visChan];
          }
        }
      } /* end for ipol */
    } /*end for Block*/
  } /* end */

PyObject *pyGridderWPol(PyObject */*self*/, PyObject *args)
{
  PyArrayObject *np_grid, *vis, *uvw, *flags, *weights, *sumwt, *increment,
    *freqs, *WInfos, *SmearMapping, *Sparsification, *np_ChanMapping,
    *LDataCorrFormat, *LExpectedOutStokes;
  PyObject *Lcfs, *LOptimisation, *LSmearing, *LJones, *Lmaps, *LcfsConj;
  int dopsf;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iO!O!O!O!O!O!O!O!O!O!O!O!O!O!",
                        //&ObjGridIn,
                        &PyArray_Type, &np_grid,
                        &PyArray_Type, &vis,
                        &PyArray_Type, &uvw,
                        &PyArray_Type, &flags,
                        &PyArray_Type, &weights,
                        &PyArray_Type, &sumwt,
                        &dopsf,
                        &PyList_Type, &Lcfs,
                        &PyList_Type, &LcfsConj,
                        &PyArray_Type, &WInfos,
                        &PyArray_Type, &increment,
                        &PyArray_Type, &freqs,
                        &PyList_Type, &Lmaps,
                        &PyList_Type, &LJones,
                        &PyArray_Type, &SmearMapping,
                        &PyArray_Type, &Sparsification,
                        &PyList_Type, &LOptimisation,
                        &PyList_Type, &LSmearing,
                        &PyArray_Type, &np_ChanMapping,
                        &PyArray_Type, &LDataCorrFormat,
                        &PyArray_Type, &LExpectedOutStokes
                        ))  return NULL;
  using svec = vector<string>;
  svec stokeslookup = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
  size_t ncorr = size_t(PyArray_Size((PyObject*)LDataCorrFormat));
  size_t npol = size_t(PyArray_Size((PyObject*)LExpectedOutStokes));
  svec inputcorr(ncorr), expstokes(npol);
  for (size_t i=0; i<ncorr; ++i)
    {
    uint16_t corrid = p_uint16(LDataCorrFormat)[i];
    if (corrid<5 || corrid>12)
      FATAL("Only accepts RR,RL,LR,LL,XX,XY,YX,YY as correlation input type");
    inputcorr[i] = stokeslookup[corrid];
    }
  for (size_t i=0; i<npol; ++i)
    {
    uint16_t polid = p_uint16(LExpectedOutStokes)[i];
    if (polid<1 || polid>4)
      FATAL("Only accepts I,Q,U,V as polarization output type");
    expstokes[i] = stokeslookup[polid];
    }
  #define callgridder(stokesgrid, nVisPol) \
    {\
    gridder<readcorr, mulaccum, stokesgrid>(np_grid, vis, uvw, flags, weights, sumwt, bool(dopsf), Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones, SmearMapping, Sparsification, LOptimisation,LSmearing,np_ChanMapping, expstokes); \
    done=true;\
    }
  auto expstokes_sorted = sortStokes(expstokes);
  bool done=false;
  if (inputcorr==svec{"XX", "XY", "YX", "YY"})
    {
    #define readcorr Read_4Corr
    #define mulaccum Mulaccum_4Corr
    if (expstokes_sorted==svec{"I"})
      callgridder(I_from_XXXYYXYY, 1)
    else if (expstokes_sorted==svec{"Q"})
      callgridder(Q_from_XXXYYXYY, 1)
    else if (expstokes_sorted==svec{"U"})
      callgridder(U_from_XXXYYXYY, 1)
    else if (expstokes_sorted==svec{"V"})
      callgridder(V_from_XXXYYXYY, 1)
    else if (expstokes_sorted==svec{"I", "Q"})
      callgridder(IQ_from_XXXYYXYY, 2)
    else if (expstokes_sorted==svec{"I", "V"})
      callgridder(IV_from_XXXYYXYY, 2)
    else if (expstokes_sorted==svec{"U", "V"})
      callgridder(UV_from_XXXYYXYY, 2)
    else if (expstokes_sorted==svec{"Q", "U"})
      callgridder(QU_from_XXXYYXYY, 2)
    else if (expstokes_sorted==svec{"I", "U"})
      callgridder(IU_from_XXXYYXYY, 2)
    else if (expstokes_sorted==svec{"Q", "V"})
      callgridder(QV_from_XXXYYXYY, 2)
    else if (expstokes_sorted==svec{"I", "Q", "U"})
      callgridder(IQU_from_XXXYYXYY, 3)
    else if (expstokes_sorted==svec{"I", "Q", "V"})
      callgridder(IQV_from_XXXYYXYY, 3)
    else if (expstokes_sorted==svec{"I", "U", "V"})
      callgridder(IUV_from_XXXYYXYY, 3)
    else if (expstokes_sorted==svec{"Q", "U", "V"})
      callgridder(QUV_from_XXXYYXYY, 3)
    else if (expstokes_sorted==svec{"I", "Q", "U", "V"})
      callgridder(IQUV_from_XXXYYXYY, 4)
    }
  if (inputcorr==svec{"RR", "RL", "LR", "LL"})
    {
    #define readcorr Read_4Corr
    #define mulaccum Mulaccum_4Corr
    if (expstokes_sorted==svec{"I"})
      callgridder(I_from_RRRLLRLL, 1)
    else if (expstokes_sorted==svec{"Q"})
      callgridder(Q_from_RRRLLRLL, 1)
    else if (expstokes_sorted==svec{"U"})
      callgridder(U_from_RRRLLRLL, 1)
    else if (expstokes_sorted==svec{"V"})
      callgridder(V_from_RRRLLRLL, 1)
    else if (expstokes_sorted==svec{"I", "Q"})
      callgridder(IQ_from_RRRLLRLL, 2)
    else if (expstokes_sorted==svec{"I", "V"})
      callgridder(IV_from_RRRLLRLL, 2)
    else if (expstokes_sorted==svec{"U", "V"})
      callgridder(UV_from_RRRLLRLL, 2)
    else if (expstokes_sorted==svec{"Q", "U"})
      callgridder(QU_from_RRRLLRLL, 2)
    else if (expstokes_sorted==svec{"I", "U"})
      callgridder(IU_from_RRRLLRLL, 2)
    else if (expstokes_sorted==svec{"Q", "V"})
      callgridder(QV_from_RRRLLRLL, 2)
    else if (expstokes_sorted==svec{"I", "Q", "U"})
      callgridder(IQU_from_RRRLLRLL, 3)
    else if (expstokes_sorted==svec{"I", "Q", "V"})
      callgridder(IQV_from_RRRLLRLL, 3)
    else if (expstokes_sorted==svec{"I", "U", "V"})
      callgridder(IUV_from_RRRLLRLL, 3)
    else if (expstokes_sorted==svec{"Q", "U", "V"})
      callgridder(QUV_from_RRRLLRLL, 3)
    else if (expstokes_sorted==svec{"I", "Q", "U", "V"})
      callgridder(IQUV_from_RRRLLRLL, 4)
    }
  else if (inputcorr==svec{"XX", "YY"})
    {
    #define readcorr Read_2Corr_Pad
    #define mulaccum Mulaccum_2Corr_Unpad
    if (expstokes_sorted==svec{"I"})
      callgridder(I_from_XXYY, 1)
    else if (expstokes_sorted==svec{"Q"})
      callgridder(Q_from_XXYY, 1)
    else if (expstokes_sorted==svec{"I", "Q"})
      callgridder(IQ_from_XXYY, 2)
    }
  else if (inputcorr==svec{"RR", "LL"})
    {
    #define readcorr Read_2Corr_Pad
    #define mulaccum Mulaccum_2Corr_Unpad
    if (expstokes_sorted==svec{"I"})
      callgridder(I_from_RRLL, 1)
    else if (expstokes_sorted==svec{"V"})
      callgridder(V_from_RRLL, 1)
    else if (expstokes_sorted==svec{"I", "V"})
      callgridder(IV_from_RRLL, 2)
    }
  if (!done)
    FATAL("Cannot convert input correlations to desired output Stokes parameters.");
  Py_RETURN_NONE;
}

using ApplyJonesType = void (*) (const JonesServer &JS, const dcMat &corr_vis, dcmplx corr, dcMat &visBuff);

void ApplyJones_4_Corr(const JonesServer &JS, const dcMat &corr_vis, dcmplx corr, dcMat &visBuff)
  {
  visBuff = JS.J0.times(corr_vis);
  visBuff = visBuff.times(JS.J1H);
  visBuff.scale(corr);
  }

void ApplyJones_2_Corr(const JonesServer &JS, const dcMat &corr_vis, dcmplx corr, dcMat &visBuff)
  {
  dcMat padded_corr_vis(corr_vis[0],0.,0.,corr_vis[1]);
  visBuff = JS.J0.times(padded_corr_vis);
  visBuff = visBuff.times(JS.J1H);
  visBuff.scale(corr);
  visBuff[1] = visBuff[3];
  }

template <StokesDegridType StokesDegrid, int nVisPol, int nVisCorr, ApplyJonesType ApplyJones>
void degridder(
  PyArrayObject *grid,
  PyArrayObject *vis,
  PyArrayObject *uvw,
  PyArrayObject *flags,
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
  double DT=-1e30, Dnu=-1e30, lmin_decorr=-1e30, mmin_decorr=-1e30;
  const double* uvw_dt_Ptr=0;
  bool DoSmearTime=false, DoSmearFreq=false;
  bool DoDecorr=(PyList_Size(LSmearing)>0);
  if (DoDecorr)
    {
    uvw_dt_Ptr = p_float64((PyArrayObject *)PyList_GetItem(LSmearing, 0));
    DT = PyFloat_AsDouble(PyList_GetItem(LSmearing, 1));
    Dnu = PyFloat_AsDouble(PyList_GetItem(LSmearing, 2));
    DoSmearTime = bool(PyFloat_AsDouble(PyList_GetItem(LSmearing, 3)));
    DoSmearFreq = bool(PyFloat_AsDouble(PyList_GetItem(LSmearing, 4)));
    lmin_decorr = PyFloat_AsDouble(PyList_GetItem(LSmearing, 5));
    mmin_decorr = PyFloat_AsDouble(PyList_GetItem(LSmearing, 6));
    }

  const double *ptrFacetInfos=p_float64((PyArrayObject *)PyArray_ContiguousFromObject(PyList_GetItem(Lmaps, 1), PyArray_FLOAT64, 0, 4));
  const double Cu=ptrFacetInfos[0];
  const double Cv=ptrFacetInfos[1];
  const double l0=ptrFacetInfos[2];
  const double m0=ptrFacetInfos[3];
  const double n0=sqrt(1-l0*l0-m0*m0)-1;

  /* Get size of grid. */
  const double *ptrWinfo = p_float64(Winfos);
  const double WaveRefWave = ptrWinfo[0];
  const double wmax = ptrWinfo[1];
  const double NwPlanes = ptrWinfo[2];
  const int OverS=(int)floor(ptrWinfo[3]);

  const int nGridX    = int(grid->dimensions[3]);
  const int nGridY    = int(grid->dimensions[2]);
  const int nGridPol  = int(grid->dimensions[1]);
  const int nGridChan = int(grid->dimensions[0]);

  /* Get visibility data size. */
  const size_t nVisChan = size_t(flags->dimensions[1]);
  const size_t nrows    = size_t(uvw->dimensions[0]);

  /* MR FIXME: should this be "/2" or "/2."? */
  const double offset_p[] = {double(nGridX/2), double(nGridY/2)};
  const double *incr=p_float64(increment);
  const double *Pfreqs=p_float64(freqs);
  /* MR FIXME: should the second entry depend on nGridY instead of nGridX? */
  const double uvwScale_p[]= {nGridX*incr[0], nGridX*incr[1]};

  const bool ChanEquidistant= bool(PyFloat_AsDouble(PyList_GetItem(LOptimisation, 1)));

  const int *MappingBlock = p_int32(SmearMapping);
  /* total size is in two words */
  size_t NTotBlocks = size_t(MappingBlock[0]) + (size_t(MappingBlock[1])<<32);
  const int *NRowBlocks = MappingBlock+2;
  const int *StartRow = MappingBlock+2+NTotBlocks;

  size_t NMaxRow=0;
  for (size_t iBlock=0; iBlock<NTotBlocks; iBlock++)
    NMaxRow = max(NMaxRow, size_t(NRowBlocks[iBlock]-2));
  vector<dcmplx> CurrentCorrTerm(NMaxRow), dCorrTerm(NMaxRow);
  /* ######################################################## */
  double WaveLengthMean=0.;
  for (size_t visChan=0; visChan<nVisChan; ++visChan)
    WaveLengthMean+=C/Pfreqs[visChan];
  WaveLengthMean/=double(nVisChan);

  JonesServer JS(LJones,WaveLengthMean);

  const int *p_ChanMapping=p_int32(np_ChanMapping);
  for (size_t iBlock=0; iBlock<NTotBlocks; iBlock++)
    {
    const int NRowThisBlock=NRowBlocks[iBlock]-2;
    const size_t chStart = size_t(StartRow[0]),
                 chEnd   = size_t(StartRow[1]);
    const int *Row = StartRow+2;
    /* advance pointer to next blocklist */
    StartRow += NRowBlocks[iBlock];

    const int gridChan = p_ChanMapping[chStart];
    if (gridChan<0 || gridChan>=nGridChan) continue;

    double FreqMean=0;
    for (auto visChan=chStart; visChan<chEnd; ++visChan)
      FreqMean+=Pfreqs[visChan];
    FreqMean/=double(chEnd-chStart);

    JS.resetJonesServerCounter();
    int NVisThisblock=0;
    double Umean=0, Vmean=0, Wmean=0;
    for (auto inx=0; inx<NRowThisBlock; inx++)
      {
      size_t irow = size_t(Row[inx]);
      if (irow>nrows) continue;
      const double* __restrict__ uvwPtr = p_float64(uvw) + irow*3;

      double W = uvwPtr[2];
      Umean += uvwPtr[0] + W*Cu;
      Vmean += uvwPtr[1] + W*Cv;
      Wmean += W;
      ++NVisThisblock;
      }

    if (NVisThisblock==0) continue;

    Umean/=NVisThisblock;
    Vmean/=NVisThisblock;
    Wmean/=NVisThisblock;
    const double recipWvl = FreqMean / C;

    /* ############## W-projection #################### */
    const int iwplane = int(lrint((NwPlanes-1)*abs(Wmean)*WaveRefWave*recipWvl/wmax));
    if (iwplane>=NwPlanes) continue;

    PyArrayObject *cfs=(PyArrayObject *) PyArray_ContiguousFromObject(
      PyList_GetItem((Wmean>0) ? Lcfs : LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
    int nConvX = int(cfs->dimensions[0]);
    int nConvY = int(cfs->dimensions[1]);
    int supx = (nConvX/OverS-1)/2;
    int supy = (nConvY/OverS-1)/2;
    int SupportCF=nConvX/OverS;

    double posx = uvwScale_p[0]*Umean*recipWvl + offset_p[0];
    double posy = uvwScale_p[1]*Vmean*recipWvl + offset_p[1];

    int locx = int(lrint(posx));    /* location in grid */
    int locy = int(lrint(posy));

    /* Only use visibility point if the full support is within grid. */
    if (locx-supx<0 || locx+supx>=nGridX || locy-supy<0 || locy+supy>=nGridY)
      continue;

    dcMat stokes_vis;

    int offx = int(lrint((locx-posx)*OverS) + (nConvX-1)/2); /* location in */
    int offy = int(lrint((locy-posy)*OverS) + (nConvY-1)/2); /* oversampling */

    int io = offy - supy*OverS;
    int jo = offx - supx*OverS;
    int cfoff = (io*OverS + jo)*SupportCF*SupportCF;
    for (size_t ipol=0; ipol<nVisPol; ++ipol)
      {
      size_t goff = size_t((gridChan*nGridPol + ipol) * nGridX*nGridY);
      const fcmplx* __restrict__ cf0 = p_complex64(cfs) + cfoff;
      const fcmplx* __restrict__ gridPtr = p_complex64(grid) + goff + (locy-supy)*nGridX + locx;
      dcmplx svi = 0.;
      for (int sy=-supy; sy<=supy; ++sy, gridPtr+=nGridX)
        for (int sx=-supx; sx<=supx; ++sx)
          svi += gridPtr[sx] * *cf0++;
      stokes_vis[ipol] = svi;
      }

    /*######## Convert from degridded stokes to MS corrs #########*/
    dcMat corr_vis;
    StokesDegrid(stokes_vis, corr_vis);

    /*################### Now do the correction #################*/
    double DeCorrFactor=1.;
    if (DoDecorr)
      {
      // MR FIXME: why take the middle of the block?
      int iRowMeanThisBlock = Row[NRowThisBlock/2];

      const double* __restrict__ uvwPtrMidRow = p_float64(uvw) + iRowMeanThisBlock*3;
      const double* __restrict__ uvw_dt_PtrMidRow = uvw_dt_Ptr + iRowMeanThisBlock*3;

      DeCorrFactor = GiveDecorrelationFactor(DoSmearFreq,DoSmearTime,
        lmin_decorr, mmin_decorr, uvwPtrMidRow, uvw_dt_PtrMidRow,
        FreqMean, Dnu, DT);
      }

    for (auto inx=0; inx<NRowThisBlock; inx++)
      {
      size_t irow = size_t(Row[inx]);
      if (irow>nrows) continue;
      const double* __restrict__ uvwPtr = p_float64(uvw) + irow*3;
      double phase = uvwPtr[0]*l0 + uvwPtr[1]*m0 + uvwPtr[2]*n0;

      for (auto visChan=chStart; visChan<chEnd; ++visChan)
        {
        size_t doff_chan = size_t(irow*nVisChan + visChan);
        size_t doff = doff_chan*nVisCorr;

        if (JS.DoApplyJones)
          JS.updateJones(irow, visChan, uvwPtr, false, false);

        dcmplx corr;
        if (ChanEquidistant)
          {
          if(visChan==0)
            {
            double UVNorm = 2.*PI*Pfreqs[visChan]/C;
            CurrentCorrTerm[inx]=polar(1.,UVNorm*phase);
            double dUVNorm = 2.*PI*(Pfreqs[1]-Pfreqs[0])/C;
            dCorrTerm[inx]=polar(1.,dUVNorm*phase);
            }
          else
            CurrentCorrTerm[inx]*=dCorrTerm[inx];
          corr=CurrentCorrTerm[inx];
          }
        else
          corr=polar(1.,2.*PI*Pfreqs[visChan]/C*phase);

        corr*=DeCorrFactor;

        dcMat visBuff;
        if (JS.DoApplyJones)
          ApplyJones(JS, corr_vis, corr, visBuff);
        else
          for(auto ThisPol=0; ThisPol<nVisCorr; ++ThisPol)
            visBuff[ThisPol] = corr_vis[ThisPol]*corr;

        fcmplx* __restrict__ visPtr = p_complex64(vis) + doff;
        auto Sem_mutex = GiveSemaphoreFromCell(doff_chan);
        /* Finally subtract visibilities from current residues */
        sem_wait(Sem_mutex);
        for (auto ThisPol=0; ThisPol<nVisCorr; ++ThisPol)
          visPtr[ThisPol] -= visBuff[ThisPol];
        sem_post(Sem_mutex);
        }/*endfor vischan*/
      }/*endfor RowThisBlock*/
    } /*end for Block*/
  } /* end */

PyObject *pyDeGridderWPol(PyObject */*self*/, PyObject *args)
  {
  PyObject *ObjVis, *Lcfs, *LOptimisation, *LSmear, *Lmaps, *LJones, *LcfsConj;
  PyArrayObject *np_grid, *np_vis, *uvw, *flags, *sumwt, *increment, *freqs,
    *WInfos, *SmearMapping, *Sparsification, *np_ChanMapping, *LDataCorrFormat,
    *LExpectedOutStokes;
  int dopsf;

  if (!PyArg_ParseTuple(args, "O!OO!O!O!iO!O!O!O!O!O!O!O!O!O!O!O!O!O!",
                        &PyArray_Type, &np_grid,
                        &ObjVis,
                        &PyArray_Type, &uvw,
                        &PyArray_Type, &flags,
                        &PyArray_Type, &sumwt,
                        &dopsf,
                        &PyList_Type, &Lcfs,
                        &PyList_Type, &LcfsConj,
                        &PyArray_Type, &WInfos,
                        &PyArray_Type, &increment,
                        &PyArray_Type, &freqs,
                        &PyList_Type, &Lmaps, &PyList_Type, &LJones,
                        &PyArray_Type, &SmearMapping,
                        &PyArray_Type, &Sparsification,
                        &PyList_Type, &LOptimisation,
                        &PyList_Type, &LSmear,
                        &PyArray_Type, &np_ChanMapping,
                        &PyArray_Type, &LDataCorrFormat,
                        &PyArray_Type, &LExpectedOutStokes
                        )) return NULL;

  using svec = vector<string>;
  np_vis = (PyArrayObject *)PyArray_ContiguousFromObject(ObjVis, PyArray_COMPLEX64, 0, 3);

  svec stokeslookup = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
  size_t ncorr = size_t(PyArray_Size((PyObject*)LDataCorrFormat));
  size_t npol = size_t(PyArray_Size((PyObject*)LExpectedOutStokes));
  svec inputcorr(ncorr), expstokes(npol);
  for (size_t i=0; i<ncorr; ++i)
    {
    uint16_t corrid = p_uint16(LDataCorrFormat)[i];
    if (corrid<5 || corrid>12)
      FATAL("Only accepts RR,RL,LR,LL,XX,XY,YX,YY as correlation output types");
    inputcorr[i] = stokeslookup[corrid];
    }
  for (size_t i=0; i<npol; ++i)
    {
    uint16_t polid = p_uint16(LExpectedOutStokes)[i];
    if (polid!=1)
      FATAL("Only accepts I as polarization input type");
    expstokes[i] = stokeslookup[polid];
    }
  bool done=false;
  #define CALL_DEGRIDDER(STOKES, NVISPOL, NVISCORR, APPLYJONES)\
    {\
    degridder<STOKES, NVISPOL, NVISCORR, APPLYJONES>(np_grid, np_vis, uvw, flags, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones, SmearMapping, LOptimisation, LSmear,np_ChanMapping);\
    done=true;\
    }
  if (expstokes==svec{"I"})
    {
    if (inputcorr==svec{"XX", "XY", "YX", "YY"})
      CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_I, 1, 4, ApplyJones_4_Corr)
    else if (inputcorr==svec{"XX", "YY"})
      CALL_DEGRIDDER(gmode_corr_XXYY_from_I, 1, 2, ApplyJones_2_Corr)
    else if (inputcorr==svec{"RR", "RL", "LR", "LL"})
      CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_I, 1, 4, ApplyJones_4_Corr)
    else if (inputcorr==svec{"RR", "LL"})
      CALL_DEGRIDDER(gmode_corr_RRLL_from_I, 1, 2, ApplyJones_2_Corr)
    }
  if (!done)
    FATAL("Cannot convert input Stokes parameter to desired output correlations.");
  return PyArray_Return(np_vis);
}

/* ==== Set up the methods table ====================== */
PyMethodDef _pyGridderSmearPols_testMethods[] = {
        {"pyGridderWPol", pyGridderWPol, METH_VARARGS, 0},
        {"pyDeGridderWPol", pyDeGridderWPol, METH_VARARGS, 0},
        {"pySetSemaphores", pySetSemaphores, METH_VARARGS, 0},
        {"pyDeleteSemaphore", pyDeleteSemaphore, METH_VARARGS, 0},
        {"pyAccumulateWeightsOntoGrid", pyAccumulateWeightsOntoGrid, METH_VARARGS, 0},
        {NULL, NULL, 0, 0} /* Sentinel - marks the end of this structure */
};

} // unnamed namespace

extern "C" {

void init_pyGridderSmearPols()
  {
  Py_InitModule("_pyGridderSmearPols", _pyGridderSmearPols_testMethods);
  import_array(); // Must be present for NumPy. Called first after above line.
  }

}
