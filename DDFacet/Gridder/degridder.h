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

#ifndef GRIDDER_DEGRIDDER_H
#define GRIDDER_DEGRIDDER_H

#include "common.h"
#include "Semaphores.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include "JonesServer.h"
#include "Stokes.h"
#include "DecorrelationHelper.h"
#include "CorrelationCalculator.h"

namespace DDF{
  namespace degridder {
    namespace policies {
      using ApplyJonesType = void (*) (const DDEs::JonesServer &JS, const dcMat &corr_vis, dcmplx corr, dcMat &visBuff);
      inline void ApplyJones_4_Corr(const DDEs::JonesServer &JS, const dcMat &corr_vis, dcmplx corr, dcMat &visBuff)
	{
	visBuff = JS.J0.times(corr_vis);
	visBuff = visBuff.times(JS.J1H);
	visBuff.scale(corr);
	}

      inline void ApplyJones_2_Corr(const DDEs::JonesServer &JS, const dcMat &corr_vis, dcmplx corr, dcMat &visBuff)
	{
	dcMat padded_corr_vis(corr_vis[0],0.,0.,corr_vis[1]);
	visBuff = JS.J0.times(padded_corr_vis);
	visBuff = visBuff.times(JS.J1H);
	visBuff.scale(corr);
	visBuff[1] = visBuff[3];
	}
    }

    template <StokesDegridType StokesDegrid, int nVisPol, int nVisCorr, policies::ApplyJonesType ApplyJones>
    void degridder(
      const py::array_t<std::complex<float>, py::array::c_style>& grid,
      py::array_t<std::complex<float>, py::array::c_style>& vis,
      const py::array_t<double, py::array::c_style>& uvw,
      const py::array_t<bool, py::array::c_style>& flags,
      const py::list& Lcfs,
      const py::list& LcfsConj,
      const py::array_t<double, py::array::c_style>& Winfos,
      const py::array_t<double, py::array::c_style>& increment,
      const py::array_t<double, py::array::c_style>& freqs,
      const py::list& Lmaps,
      py::list& LJones,
      const py::array_t<int32_t, py::array::c_style>& SmearMapping,
      const py::list& LOptimisation,
      const py::list& LSmearing,
      const py::array_t<int, py::array::c_style>& np_ChanMapping)
      {
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
      const size_t nVisChan = size_t(flags.shape(1));
      const size_t nrows    = size_t(uvw.shape(0));
      const double *uvwdata = uvw.data(0);

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

      CorrectionCalculator Corrcalc(LOptimisation, 0, NTotBlocks, NRowBlocks);
      /* ######################################################## */
      double WaveLengthMean=0.;
      for (size_t visChan=0; visChan<nVisChan; ++visChan)
	WaveLengthMean+=C/Pfreqs[visChan];
      WaveLengthMean/=double(nVisChan);

      DDEs::JonesServer JS(LJones,WaveLengthMean);

      const int *p_ChanMapping=np_ChanMapping.data(0);
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
	  const double* __restrict__ uvwPtr = uvwdata + irow*3;
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

	dcMat stokes_vis;

	const int offx = int(lrint((locx-posx)*OverS) + (nConvX-1)/2); /* location in */
	const int offy = int(lrint((locy-posy)*OverS) + (nConvY-1)/2); /* oversampling */

	const int io = offy - supy*OverS;
	const int jo = offx - supx*OverS;
	const int cfoff = (io*OverS + jo)*SupportCF*SupportCF;

	for (size_t ipol=0; ipol<nVisPol; ++ipol)
	  {
	  const size_t goff = size_t((gridChan*nGridPol + ipol) * nGridX*nGridY);
	  const fcmplx* __restrict__ cf0 = cfs.data(0) + cfoff;
	  const fcmplx* __restrict__ gridPtr = grid.data(0) + goff + (locy-supy)*nGridX + locx;
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
	  const double* __restrict__ uvwPtr = uvwdata + irow*3;
	  const double angle = 2.*PI*(uvwPtr[0]*l0+uvwPtr[1]*m0+uvwPtr[2]*n0)/C;

	  for (auto visChan=chStart; visChan<chEnd; ++visChan)
	    {
	    const size_t doff_chan = size_t(irow*nVisChan + visChan);
	    const size_t doff = doff_chan*nVisCorr;

	    if (JS.DoApplyJones)
	      JS.updateJones(irow, visChan, uvwPtr, false, false);

	    dcmplx corr = Corrcalc.getCorr(inx, Pfreqs, visChan, angle);
	    corr*=DeCorrFactor;

	    dcMat visBuff;
	    if (JS.DoApplyJones)
	      ApplyJones(JS, corr_vis, corr, visBuff);
	    else
	      for(auto ThisPol=0; ThisPol<nVisCorr; ++ThisPol)
		visBuff[ThisPol] = corr_vis[ThisPol]*corr;

	    fcmplx* __restrict__ visPtr = vis.mutable_data(0) + doff;
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
  }
}

#endif /*GRIDDER_DEGRIDDER_H*/
