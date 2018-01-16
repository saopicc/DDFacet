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

class DecorrelationHelper
  {
  private:
    double DT, Dnu, l0, m0;
    const double *uvw_Ptr, *uvw_dt_Ptr;
    bool DoDecorr, TSmear, FSmear;

  public:
    DecorrelationHelper(PyObject *LSmearing, PyArrayObject *uvw)
      {
      DoDecorr=(PyList_Size(LSmearing)>0);
      if (DoDecorr)
        {
        uvw_Ptr = p_float64(uvw);
        uvw_dt_Ptr = p_float64((PyArrayObject *)PyList_GetItem(LSmearing, 0));
        DT = PyFloat_AsDouble(PyList_GetItem(LSmearing, 1));
        Dnu = PyFloat_AsDouble(PyList_GetItem(LSmearing, 2));
        TSmear = bool(PyFloat_AsDouble(PyList_GetItem(LSmearing, 3)));
        FSmear = bool(PyFloat_AsDouble(PyList_GetItem(LSmearing, 4)));
        l0 = PyFloat_AsDouble(PyList_GetItem(LSmearing, 5));
        m0 = PyFloat_AsDouble(PyList_GetItem(LSmearing, 6));
        }
      }

    double get(double nu, size_t idx)
      {
      if (!DoDecorr) return 1.;

      double n0=sqrt(1.-l0*l0-m0*m0)-1.;
      double DecorrFactor=1.;

      if (FSmear)
        {
        double phase = uvw_Ptr[3+idx+0]*l0 + uvw_Ptr[3*idx+1]*m0 + uvw_Ptr[3*idx+2]*n0;
        double phi=PI*Dnu/C*phase;
        if (phi!=0.)
          DecorrFactor*=max(0.,sin(phi)/phi);
        }

      if (TSmear)
        {
        double dphase = (uvw_dt_Ptr[3*idx+0]*l0 + uvw_dt_Ptr[3*idx+1]*m0 + uvw_dt_Ptr[3*idx+2]*n0)*DT;
        double phi=PI*nu/C*dphase;
        if (phi!=0.)
          DecorrFactor*=max(0.,sin(phi)/phi);
        }
      return DecorrFactor;
      }
  };

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

class CorrectionCalculator
  {
  private:
    bool ChanEquidistant;
    const bool *sparsificationFlag;
    size_t NMaxRow;
    /* these are used for equidistant channels: one holds the phase term in the
       current channel, the other one holds the delta-phase across channels */
    vector<dcmplx> CurrentCorrTerm, dCorrTerm;
    /* and this indicates for which channel the CurrentCorrTerm is currently computed */
    vector<int> CurrentCorrChan;
    int CurrentCorrRow0;

  public:
    CorrectionCalculator(PyObject *LOptimisation, const bool *sparsificationFlag_,
      const size_t NTotBlocks, const int *NRowBlocks)
      {
      sparsificationFlag = sparsificationFlag_;
      ChanEquidistant= bool(PyFloat_AsDouble(PyList_GetItem(LOptimisation, 1)));
      NMaxRow=0;
      if (ChanEquidistant)
        for (size_t i=0; i<NTotBlocks; ++i)
          if (!sparsificationFlag || sparsificationFlag[i])
            NMaxRow = max(NMaxRow, size_t(NRowBlocks[i]-2));
      CurrentCorrTerm.resize(NMaxRow);
      dCorrTerm.resize(NMaxRow);
      CurrentCorrChan.resize(NMaxRow,-1);
      CurrentCorrRow0 = -1;
      }

    void update(int Row0, int NRowThisBlock)
      {
      /* when moving to a new block of rows, init this to -1 so the code below knows to initialize*/
      /* CurrentCorrTerm when the first channel of each row comes in*/
      if (ChanEquidistant)
        if (Row0!=CurrentCorrRow0)
        {
        for (auto inx=0; inx<NRowThisBlock; inx++)
          CurrentCorrChan[inx] = -1;
        CurrentCorrRow0 = Row0;
        }
      }

    dcmplx getCorr(int inx, const double *Pfreqs, size_t visChan, double angle)
      {
      if (!ChanEquidistant)
        return polar(1.,Pfreqs[visChan]*angle);

      /* init correlation term for first channel that it's not initialized in */
      if (CurrentCorrChan[inx]==-1)
        {
        CurrentCorrTerm[inx] = polar(1.,Pfreqs[visChan]*angle);
        dCorrTerm[inx]       = polar(1.,(Pfreqs[1]-Pfreqs[0])*angle);
        CurrentCorrChan[inx] = int(visChan);
        }
      /* else, wind the correlation term forward by as many channels as necessary */
      /* this modification allows us to support blocks that skip across channels */
      else
        {
        while (size_t(CurrentCorrChan[inx])<visChan)
          {
          CurrentCorrTerm[inx] *= dCorrTerm[inx];
          CurrentCorrChan[inx]++;
          }
        }
      return CurrentCorrTerm[inx];
      }
    };

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
  DecorrelationHelper decorr(LSmearing, uvw);

  const double *ptrFacetInfos=p_float64((PyArrayObject *) PyList_GetItem(Lmaps, 1));
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
  const int OverS=int(floor(ptrWinfo[3]));

  const int nGridX    = int(grid->dimensions[3]);
  const int nGridY    = int(grid->dimensions[2]);
  const int nGridPol  = int(grid->dimensions[1]);
  const int nGridChan = int(grid->dimensions[0]);

  /* Get visibility data size. */
  const size_t nVisCorr = size_t(flags->dimensions[2]);
  const size_t nVisChan = size_t(flags->dimensions[1]);
  const size_t nrows    = size_t(uvw->dimensions[0]);

  double* __restrict__ sumWtPtr = p_float64(sumwt);

  /* MR FIXME: should this be "/2" or "/2."? */
  const double offset_p[] = {double(nGridX/2), double(nGridY/2)};
  const double *incr=p_float64(increment);
  const double *Pfreqs=p_float64(freqs);
  /* MR FIXME: should the second entry depend on nGridY instead of nGridX? */
  const double uvwScale_p[]= {nGridX*incr[0], nGridX*incr[1]};

  const int *MappingBlock = p_int32(SmearMapping);
  /* total size is in two words */
  const size_t NTotBlocks = size_t(MappingBlock[0]) + (size_t(MappingBlock[1])<<32);
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

  CorrectionCalculator Corrcalc(LOptimisation, sparsificationFlag, NTotBlocks, NRowBlocks);

  /* ######################################################## */
  double WaveLengthMean=0., FreqMean0=0.;
  for (size_t visChan=0; visChan<nVisChan; ++visChan)
    {
    WaveLengthMean+=C/Pfreqs[visChan];
    FreqMean0+=Pfreqs[visChan];
    }
  WaveLengthMean/=double(nVisChan);
  FreqMean0/=double(nVisChan);

  JonesServer JS(LJones,WaveLengthMean);

  vector<double> ThisSumJonesChan(nVisChan), ThisSumSqWeightsChan(nVisChan);

  const int *p_ChanMapping=p_int32(np_ChanMapping);
  for (size_t iBlock=0; iBlock<NTotBlocks; iBlock++)
    {
    if (sparsificationFlag && !sparsificationFlag[iBlock])
      continue;

    const int NRowThisBlock=NRowBlocks[iBlock]-2;
    const size_t chStart = size_t(StartRow[0]),
                 chEnd   = size_t(StartRow[1]);
    const int *Row = StartRow+2;
    /* advance pointer to next blocklist */
    StartRow += NRowBlocks[iBlock];

    dcMat Vis(0,0,0,0);

    for (size_t visChan=0; visChan<nVisChan; ++visChan)
      ThisSumJonesChan[visChan] = ThisSumSqWeightsChan[visChan] = 0;

    Corrcalc.update(Row[0], NRowThisBlock);

    double DeCorrFactor=decorr.get(FreqMean0, Row[NRowThisBlock/2]);

    double visChanMean=0., FreqMean=0;
    double ThisWeight=0., ThisSumJones=0., ThisSumSqWeights=0.;
    JS.resetJonesServerCounter();
    int NVisThisblock=0;
    double Umean=0, Vmean=0, Wmean=0;
    for (auto inx=0; inx<NRowThisBlock; inx++)
      {
      const size_t irow = size_t(Row[inx]);
      if (irow>nrows) continue;
      const double* __restrict__ uvwPtr = p_float64(uvw) + irow*3;
      const double U=uvwPtr[0];
      const double V=uvwPtr[1];
      const double W=uvwPtr[2];
      const double angle = -2.*PI*(U*l0+V*m0+W*n0)/C;
      JS.WeightVaryJJ=1.;

      for (size_t visChan=chStart; visChan<chEnd; ++visChan)
        {
        size_t doff = size_t((irow*nVisChan + visChan) * nVisCorr);
        const float *imgWtPtr = p_float32(weights) + irow*nVisChan + visChan;

        /* We can do that since all flags in 4-pols are equalised in ClassVisServer */
        if (p_bool(flags)[doff]) continue;

        dcmplx corr = Corrcalc.getCorr(inx, Pfreqs, visChan, angle);

        if (JS.DoApplyJones)
          JS.updateJones(irow, visChan, uvwPtr, true, true);

        dcMat VisMeas;
        if (dopsf)
          {
          // MR FIXME: why reset corr here? Seems like wasted work...
          corr=1.;
          if (JS.DoApplyJones)
            VisMeas=(JS.J0).times(JS.J1H); // MR FIXME: precompute?
          else
            VisMeas.setUnity();
          if (DeCorrFactor!=1.)
            for(int ThisPol=0; ThisPol<4;ThisPol++)
              VisMeas[ThisPol]*=DeCorrFactor;
          }
        else
          readcorr(p_complex64(vis)+doff, VisMeas);

        const double FWeight = imgWtPtr[0]*JS.WeightVaryJJ;
        const dcmplx Weight = FWeight*corr;
        if (JS.DoApplyJones)
          {
          VisMeas=(JS.J0H.times(VisMeas)).times(JS.J1);
          mulaccum(VisMeas, Weight, Vis);

          /*Compute per channel and overall approximate matrix sqroot:*/
          const double FWeightSq=FWeight*DeCorrFactor*DeCorrFactor;
          ThisSumJones+=JS.BB*FWeightSq;
          ThisSumSqWeights+=FWeightSq;

          ThisSumJonesChan[visChan]+=JS.BB*FWeightSq;
          ThisSumSqWeightsChan[visChan]+=FWeightSq;
          }
        else /* Don't apply Jones */
          mulaccum(VisMeas, Weight, Vis);

        /*###################### Averaging #######################*/
        Umean += U + W*Cu;
        Vmean += V + W*Cv;
        Wmean += W;
        FreqMean+=Pfreqs[visChan];
        ThisWeight+=FWeight;

        visChanMean+=p_ChanMapping[visChan];
        ++NVisThisblock;
        }/*endfor vischan*/
      }/*endfor RowThisBlock*/

    if (NVisThisblock==0) continue;

    visChanMean/=NVisThisblock;

    const int gridChan = p_ChanMapping[chStart];
    const double diffChan=visChanMean-gridChan;
    if(abs(diffChan)>1e-6)
      {
      printf("gridder: probably there is a problem in the BDA mapping: (ChanMean, gridChan, diff)=(%lf, %i, %lf)\n",visChanMean,gridChan,diffChan);
      for (size_t visChan=chStart; visChan<chEnd; ++visChan)
        printf("%d ", gridChan-p_ChanMapping[visChan]);
      printf("\n");
      }

    /* ################################################ */
    /* ######## Convert correlations to stokes ######## */
    const dcMat stokes_vis(stokesgrid(Vis));

    /* ################################################ */
    /* ############## Start Gridding visibility ####### */
    if (gridChan<0 || gridChan>=nGridChan) continue;

    Umean/=NVisThisblock;
    Vmean/=NVisThisblock;
    Wmean/=NVisThisblock;
    FreqMean/=NVisThisblock;
    const double recipWvl = FreqMean / C;

    /* ############## W-projection #################### */
    const int iwplane = int(lrint((NwPlanes-1)*abs(Wmean)*(WaveRefWave*recipWvl)/wmax));
    if (iwplane>=NwPlanes) continue;

    PyArrayObject *cfs=(PyArrayObject *) PyArray_ContiguousFromObject(
      PyList_GetItem((Wmean>0) ? Lcfs : LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
    const int nConvX = int(cfs->dimensions[0]);
    const int nConvY = int(cfs->dimensions[1]);
    const int supx = (nConvX/OverS-1)/2;
    const int supy = (nConvY/OverS-1)/2;
    const int SupportCF=nConvX/OverS;

    const double posx = uvwScale_p[0]*Umean*recipWvl + offset_p[0];
    const double posy = uvwScale_p[1]*Vmean*recipWvl + offset_p[1];

    const int locx = int(lrint(posx));    /* location in grid */
    const int locy = int(lrint(posy));

    /* Only use visibility point if the full support is within grid. */
    if (locx-supx<0 || locx+supx>=nGridX || locy-supy<0 || locy+supy>=nGridY)
      continue;

    const int offx = int(lrint((locx-posx)*OverS) + (nConvX-1)/2); /* location in */
    const int offy = int(lrint((locy-posy)*OverS) + (nConvY-1)/2); /* oversampling */

    const int io = offy - supy*OverS;
    const int jo = offx - supx*OverS;
    const int cfoff = (io*OverS + jo)*SupportCF*SupportCF;

    for (size_t ipol=0; ipol<nVisPol; ++ipol)
      {
      if (ipol>=size_t(nGridPol)) continue;
      const size_t goff = size_t((gridChan*nGridPol + ipol) * nGridX*nGridY);
      const dcmplx VisVal =stokes_vis[ipol];
      const fcmplx* __restrict__ cf0 = p_complex64(cfs) + cfoff;
      fcmplx* __restrict__ gridPtr = p_complex64(grid) + goff + (locy-supy)*nGridX + locx;
      for (int sy=-supy; sy<=supy; ++sy, gridPtr+=nGridX)
        for (int sx=-supx; sx<=supx; ++sx)
          gridPtr[sx] += VisVal * dcmplx(*cf0++);
      sumWtPtr[ipol+gridChan*nGridPol] += ThisWeight;
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
  bool done=false;
  if (inputcorr==svec{"XX", "XY", "YX", "YY"})
    {
    #define readcorr Read_4Corr
    #define mulaccum Mulaccum_4Corr
    if (expstokes==svec{"I"})
      callgridder(I_from_XXXYYXYY, 1)
    else if (expstokes==svec{"I", "Q"})
      callgridder(IQ_from_XXXYYXYY, 2)
    else if (expstokes==svec{"I", "V"})
      callgridder(IV_from_XXXYYXYY, 2)
    else if (expstokes==svec{"Q", "U"})
      callgridder(QU_from_XXXYYXYY, 2)
    else if (expstokes==svec{"I", "Q", "U", "V"})
      callgridder(IQUV_from_XXXYYXYY, 4)
    #undef readcorr
    #undef mulaccum
    }
  if (inputcorr==svec{"RR", "RL", "LR", "LL"})
    {
    #define readcorr Read_4Corr
    #define mulaccum Mulaccum_4Corr
    if (expstokes==svec{"I"})
      callgridder(I_from_RRRLLRLL, 1)
    else if (expstokes==svec{"I", "Q"})
      callgridder(IQ_from_RRRLLRLL, 2)
    else if (expstokes==svec{"I", "V"})
      callgridder(IV_from_RRRLLRLL, 2)
    else if (expstokes==svec{"Q", "U"})
      callgridder(QU_from_RRRLLRLL, 2)
    else if (expstokes==svec{"I", "Q", "U", "V"})
      callgridder(IQUV_from_RRRLLRLL, 4)
    #undef readcorr
    #undef mulaccum
    }
  else if (inputcorr==svec{"XX", "YY"})
    {
    #define readcorr Read_2Corr_Pad
    #define mulaccum Mulaccum_2Corr_Unpad
    if (expstokes==svec{"I"})
      callgridder(I_from_XXYY, 1)
    else if (expstokes==svec{"I", "Q"})
      callgridder(IQ_from_XXYY, 2)
    #undef readcorr
    #undef mulaccum
    }
  else if (inputcorr==svec{"RR", "LL"})
    {
    #define readcorr Read_2Corr_Pad
    #define mulaccum Mulaccum_2Corr_Unpad
    if (expstokes==svec{"I"})
      callgridder(I_from_RRLL, 1)
    else if (expstokes==svec{"I", "V"})
      callgridder(IV_from_RRLL, 2)
    #undef readcorr
    #undef mulaccum
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
  DecorrelationHelper decorr(LSmearing, uvw);

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
  const int OverS=int(floor(ptrWinfo[3]));

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

  const int *MappingBlock = p_int32(SmearMapping);
  /* total size is in two words */
  const size_t NTotBlocks = size_t(MappingBlock[0]) + (size_t(MappingBlock[1])<<32);
  const int *NRowBlocks = MappingBlock+2;
  const int *StartRow = MappingBlock+2+NTotBlocks;

  CorrectionCalculator Corrcalc(LOptimisation, 0, NTotBlocks, NRowBlocks);
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
      const size_t irow = size_t(Row[inx]);
      if (irow>nrows) continue;
      const double* __restrict__ uvwPtr = p_float64(uvw) + irow*3;
      const double U=uvwPtr[0];
      const double V=uvwPtr[1];
      const double W=uvwPtr[2];

      Umean += U + W*Cu;
      Vmean += V + W*Cv;
      Wmean += W;
      ++NVisThisblock;
      }

    if (NVisThisblock==0) continue;

    Umean/=NVisThisblock;
    Vmean/=NVisThisblock;
    Wmean/=NVisThisblock;
    const double recipWvl = FreqMean / C;

    /* ############## W-projection #################### */
    const int iwplane = int(lrint((NwPlanes-1)*abs(Wmean)*(WaveRefWave*recipWvl)/wmax));
    if (iwplane>=NwPlanes) continue;

    PyArrayObject *cfs=(PyArrayObject *) PyArray_ContiguousFromObject(
      PyList_GetItem((Wmean>0) ? Lcfs : LcfsConj, iwplane), PyArray_COMPLEX64, 0, 2);
    const int nConvX = int(cfs->dimensions[0]);
    const int nConvY = int(cfs->dimensions[1]);
    const int supx = (nConvX/OverS-1)/2;
    const int supy = (nConvY/OverS-1)/2;
    const int SupportCF=nConvX/OverS;

    const double posx = uvwScale_p[0]*Umean*recipWvl + offset_p[0];
    const double posy = uvwScale_p[1]*Vmean*recipWvl + offset_p[1];

    const int locx = int(lrint(posx));    /* location in grid */
    const int locy = int(lrint(posy));

    /* Only use visibility point if the full support is within grid. */
    if (locx-supx<0 || locx+supx>=nGridX || locy-supy<0 || locy+supy>=nGridY)
      continue;

    dcMat stokes_vis;

    const int offx = int(lrint((locx-posx)*OverS) + (nConvX-1)/2); /* location in */
    const int offy = int(lrint((locy-posy)*OverS) + (nConvY-1)/2); /* oversampling */

    const int io = offy - supy*OverS;
    const int jo = offx - supx*OverS;
    const int cfoff = (io*OverS + jo)*SupportCF*SupportCF;

    for (size_t ipol=0; ipol<nVisPol; ++ipol)
      {
      const size_t goff = size_t((gridChan*nGridPol + ipol) * nGridX*nGridY);
      const fcmplx* __restrict__ cf0 = p_complex64(cfs) + cfoff;
      const fcmplx* __restrict__ gridPtr = p_complex64(grid) + goff + (locy-supy)*nGridX + locx;
      dcmplx svi = 0.;
      for (int sy=-supy; sy<=supy; ++sy, gridPtr+=nGridX)
        for (int sx=-supx; sx<=supx; ++sx)
          svi += gridPtr[sx] * *cf0++;
      stokes_vis[ipol] = svi;
      }

    /*######## Convert from degridded stokes to MS corrs #########*/
    dcMat corr_vis = StokesDegrid(stokes_vis);

    Corrcalc.update(Row[0], NRowThisBlock);

    /*################### Now do the correction #################*/
    double DeCorrFactor=decorr.get(FreqMean, Row[NRowThisBlock/2]);

    for (auto inx=0; inx<NRowThisBlock; inx++)
      {
      size_t irow = size_t(Row[inx]);
      if (irow>nrows) continue;
      const double* __restrict__ uvwPtr = p_float64(uvw) + irow*3;
      const double angle = 2.*PI*(uvwPtr[0]*l0+uvwPtr[1]*m0+uvwPtr[2]*n0)/C;

      for (auto visChan=chStart; visChan<chEnd; ++visChan)
        {
        const size_t doff_chan = size_t(irow*nVisChan + visChan);
        const size_t doff = doff_chan*nVisCorr;

        if (JS.DoApplyJones)
          JS.updateJones(irow, visChan, uvwPtr, false, false);

        dcmplx corr=Corrcalc.getCorr(inx, Pfreqs, visChan, angle);
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

  const svec stokeslookup = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
  const size_t ncorr = size_t(PyArray_Size((PyObject*)LDataCorrFormat));
  const size_t npol = size_t(PyArray_Size((PyObject*)LExpectedOutStokes));
  svec inputcorr(ncorr), expstokes(npol);
  for (size_t i=0; i<ncorr; ++i)
    {
    const uint16_t corrid = p_uint16(LDataCorrFormat)[i];
    if (corrid<5 || corrid>12)
      FATAL("Only accepts RR,RL,LR,LL,XX,XY,YX,YY as correlation output types");
    inputcorr[i] = stokeslookup[corrid];
    }
  for (size_t i=0; i<npol; ++i)
    {
    const uint16_t polid = p_uint16(LExpectedOutStokes)[i];
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
