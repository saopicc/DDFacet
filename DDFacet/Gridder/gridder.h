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

#ifndef GRIDDER_GRIDDER_H
#define GRIDDER_GRIDDER_H

#include "common.h"
#include "Semaphores.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "JonesServer.h"
#include "Stokes.h"
#include "DecorrelationHelper.h"
#include "CorrelationCalculator.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace DDF {
  namespace gridder {
    namespace py=pybind11;
    using namespace std;
    namespace policies {
      using ReadCorrType = void (*) (const fcmplx*, dcMat &);
      inline void Read_4Corr(const fcmplx *visPtrMeas, dcMat &VisMeas)
        {
	  for (auto i=0; i<4; ++i)
	    VisMeas[i]=visPtrMeas[i];
        }
      inline void Read_2Corr_Pad(const fcmplx *visPtrMeas, dcMat &VisMeas)
        {
	  VisMeas[0]=visPtrMeas[0];
	  VisMeas[1]=VisMeas[2]=0;
	  VisMeas[3]=visPtrMeas[1];
        }

      using MulaccumType = void (*) (const dcMat &, dcmplx, dcMat &);
      inline void Mulaccum_4Corr(const dcMat &VisMeas, dcmplx Weight, dcMat &Vis)
	{
	  for (auto i=0; i<4; ++i)
	    Vis[i] += VisMeas[i]*Weight;
	}
      inline void Mulaccum_2Corr_Unpad(const dcMat &VisMeas, dcmplx Weight, dcMat &Vis)
	{
	  Vis[0] += VisMeas[0]*Weight;
	  Vis[1] += VisMeas[3]*Weight;
	}
    }
    template<policies::ReadCorrType readcorr, policies::MulaccumType mulaccum, policies::StokesGridType stokesgrid>
    void gridder(py::array_t<std::complex<float>, py::array::c_style>& grid,
		const py::array_t<std::complex<float>, py::array::c_style>& vis,
		const py::array_t<double, py::array::c_style>& uvw,
		const py::array_t<bool, py::array::c_style>& flags,
		const py::array_t<float, py::array::c_style>& weights,
		py::array_t<double, py::array::c_style>& sumwt,
		bool dopsf,
		const py::list& Lcfs,
		const py::list& LcfsConj,
		const py::array_t<double, py::array::c_style>& Winfos,
		const py::array_t<double, py::array::c_style>& increment,
		const py::array_t<double, py::array::c_style>& freqs,
		const py::list& Lmaps,
		py::list& LJones,
		const py::array_t<int32_t, py::array::c_style>& SmearMapping,
		const py::array_t<bool, py::array::c_style>& Sparsification,
		const py::list& LOptimisation,
		const py::list& LSmearing,
		const py::array_t<int, py::array::c_style>& np_ChanMapping,
		const vector<string> &expstokes)
      {
      auto nVisPol = expstokes.size();
      DDEs::DecorrelationHelper decorr(LSmearing, uvw);
      const double *ptrFacetInfos=py::array_t<double, py::array::c_style>(Lmaps[1]).data(0);
      const double Cu=ptrFacetInfos[0];
      const double Cv=ptrFacetInfos[1];
      const double l0=ptrFacetInfos[2];
      const double m0=ptrFacetInfos[3];
      const double n0=sqrt(1-l0*l0-m0*m0)-1;

      /* Get size of grid. */
      const double *ptrWinfo = Winfos.data(0);
      const double WaveRefWave = ptrWinfo[0];
      const double wmax = ptrWinfo[1];
      const double NwPlanes = ptrWinfo[2];
      const int OverS=int(floor(ptrWinfo[3]));

      const int nGridX    = int(grid.shape(3));
      const int nGridY    = int(grid.shape(2));
      const int nGridPol  = int(grid.shape(1));
      const int nGridChan = int(grid.shape(0));

      /* Get visibility data size. */
      const size_t nVisCorr = size_t(flags.shape(2));
      const size_t nVisChan = size_t(flags.shape(1));
      const size_t nrows    = size_t(uvw.shape(0));

      double* __restrict__ sumWtPtr =  sumwt.mutable_data(0);

      /* MR FIXME: should this be "/2" or "/2."? */
      const double offset_p[] = {double(nGridX/2), double(nGridY/2)};
      const double *incr=increment.data(0);
      const double *Pfreqs=freqs.data(0);
      /* MR FIXME: should the second entry depend on nGridY instead of nGridX? */
      const double uvwScale_p[]= {nGridX*incr[0], nGridX*incr[1]};

      const int *MappingBlock = SmearMapping.data(0);
      /* total size is in two words */
      const size_t NTotBlocks = size_t(MappingBlock[0]) + (size_t(MappingBlock[1])<<32);
      const int *NRowBlocks = MappingBlock+2;
      const int *StartRow = MappingBlock+2+NTotBlocks;

      /* in sparsification mode, the Sparsification argument is an array of length NTotBlocks flags. */
      /* Only blocks with a True flag will be gridded. */
      const bool *sparsificationFlag = 0;
      if (Sparsification.size())
	{
	if (size_t(Sparsification.size()) != NTotBlocks)
	  throw std::invalid_argument("sparsification argument must be an array of length NTotBlocks");

	sparsificationFlag = Sparsification.data(0);
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

      DDEs::JonesServer JS(LJones,WaveLengthMean);

      vector<double> ThisSumJonesChan(nVisChan), ThisSumSqWeightsChan(nVisChan);

      const int *p_ChanMapping=np_ChanMapping.data(0);
      for (size_t iBlock=0; iBlock<NTotBlocks; iBlock++)
	{
	const int NRowThisBlock=NRowBlocks[iBlock]-2;
	const size_t chStart = size_t(StartRow[0]),
		    chEnd   = size_t(StartRow[1]);
	const int *Row = StartRow+2;
	/* advance pointer to next blocklist */
	StartRow += NRowBlocks[iBlock];

	if (sparsificationFlag && !sparsificationFlag[iBlock])
	  continue;

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
	dcMat VisMeas_psf;
	VisMeas_psf.setUnity();
	for (auto inx=0; inx<NRowThisBlock; inx++)
	  {
	  const size_t irow = size_t(Row[inx]);
	  if (irow>nrows) continue;
	  const double* __restrict__ uvwPtr = uvw.data(0) + irow*3;
	  const double U=uvwPtr[0];
	  const double V=uvwPtr[1];
	  const double W=uvwPtr[2];
	  const double angle = -2.*PI*(U*l0+V*m0+W*n0)/C;
	  JS.WeightVaryJJ=1.;

	  for (size_t visChan=chStart; visChan<chEnd; ++visChan)
	    {
	    size_t doff = size_t((irow*nVisChan + visChan) * nVisCorr);
	    const float *imgWtPtr = weights.data(0) + irow*nVisChan + visChan;

	    /* We can do that since all flags in 4-pols are equalised in ClassVisServer */
	    if (flags.data(0)[doff]) continue;

	    dcmplx corr = dopsf ? 1 : Corrcalc.getCorr(inx, Pfreqs, visChan, angle);
	    dcMat VisMeas;

	    if (JS.DoApplyJones)
	      {
	      JS.updateJones(irow, visChan, uvwPtr, true, true);
	      if (dopsf)
	        VisMeas_psf = (JS.J0).times(JS.J1H); // precompute for the PSF case
	      }

	    if (dopsf)
	      {
	      VisMeas = VisMeas_psf;
	      if (DeCorrFactor!=1.)
		VisMeas.scale(DeCorrFactor);
	      }
	    else
	      readcorr(vis.data(0)+doff, VisMeas);

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

	auto cfs=py::array_t<complex<float>, py::array::c_style>(
	  (Wmean>0) ? Lcfs[iwplane] : LcfsConj[iwplane]);
	const int nConvX = int(cfs.shape(0));
	const int nConvY = int(cfs.shape(1));
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
	  const fcmplx* __restrict__ cf0 = cfs.data(0) + cfoff;
	  fcmplx* __restrict__ gridPtr = grid.mutable_data(0) + goff + (locy-supy)*nGridX + locx;
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
    }
}

#endif /*GRIDDER_GRIDDER_H*/
