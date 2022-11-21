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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gridder.h"
#include "degridder.h"
#include <cstdint>

#include <iostream>
#include <vector>
#include <string>

namespace DDF {
  void pyAccumulateWeightsOntoGrid(py::array_t<double, py::array::c_style>& grid,
				  const py::array_t<float, py::array::c_style>& weights,
				  const py::array_t<long int, py::array::c_style>& index)
  {
    size_t n = weights.shape(0);
    double* pgrid = grid.mutable_data<double>(0);
    const float* pweights = weights.data<float>(0);
    const long int* pindex = index.data<long int>(0);

    for(size_t i=0; i<n; ++i)
      {
      float w = pweights[i];
      if (w!=0)
	{
	size_t igrid = size_t(pindex[i]);
	sem_t *psem = GiveSemaphoreFromCell(igrid);
	sem_wait(psem);
	pgrid[igrid] += w;
	sem_post(psem);
	}
      }
  }

  // semaphore-free version for use in non-parallel contexts
  void pyAccumulateWeightsOntoGridNoSem(py::array_t<double, py::array::c_style>& grid,
				  const py::array_t<float, py::array::c_style>& weights,
				  const py::array_t<long int, py::array::c_style>& index)
  {
    size_t n = weights.shape(0);
    double* pgrid = grid.mutable_data<double>(0);
    const float* pweights = weights.data<float>(0);
    const long int* pindex = index.data<long int>(0);

    for(size_t i=0; i<n; ++i)
      {
      float w = pweights[i];
      if (w!=0)
	{
	size_t igrid = size_t(pindex[i]);
	pgrid[igrid] += w;
	}
      }
  }
  
  template<typename gridtype>
  void pyGridderWPol(py::array_t<std::complex<gridtype>, py::array::c_style>& np_grid,
		    const py::array_t<std::complex<float>, py::array::c_style>& vis,
		    const py::array_t<double, py::array::c_style>& uvw,
		    const py::array_t<bool, py::array::c_style>& flags,
		    const py::array_t<float, py::array::c_style>& weights,
		    py::array_t<double, py::array::c_style>& sumwt,
		    bool dopsf,
		    const py::list& Lcfs,
		    const py::list& LcfsConj,
		    const py::array_t<double, py::array::c_style>& WInfos,
		    const py::array_t<double, py::array::c_style>& increment,
		    const py::array_t<double, py::array::c_style>& freqs,
		    const py::list& Lmaps,
		    py::list& LJones,
		    const py::array_t<int32_t, py::array::c_style>& SmearMapping,
		    const py::array_t<bool, py::array::c_style>& Sparsification,
		    const py::list& LOptimisation,
		    const py::list& LSmearing,
		    const py::array_t<int32_t, py::array::c_style>& np_ChanMapping,
		    const py::array_t<uint16_t, py::array::c_style>& LDataCorrFormat,
		    const py::array_t<uint16_t, py::array::c_style>& LExpectedOutStokes)
  {
    using svec = vector<string>;
    const svec stokeslookup = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
    size_t ncorr = LDataCorrFormat.shape(0);
    size_t npol = LExpectedOutStokes.shape(0);
    svec inputcorr(ncorr), expstokes(npol);
    for (size_t i=0; i<ncorr; ++i)
      {
      uint16_t corrid = LDataCorrFormat.data<uint16_t>(0)[i];
      if (!(corrid>=5 && corrid<=12 || corrid == 1))
	throw std::invalid_argument("Only accepts I,RR,RL,LR,LL,XX,XY,YX,YY as correlation input type");
      inputcorr[i] = stokeslookup[corrid];
      }
    for (size_t i=0; i<npol; ++i)
      {
      uint16_t polid = LExpectedOutStokes.data<uint16_t>(0)[i];
      if (polid<1 || polid>4)
	throw std::invalid_argument("Only accepts I,Q,U,V as polarization output type");
      expstokes[i] = stokeslookup[polid];
      }
    #define callgridder(stokesgrid, nVisPol) \
      {\
            gridder::gridder<readcorr, mulaccum, stokesgrid, gridtype>(np_grid, vis, uvw, flags, weights, sumwt,\
                                                                       bool(dopsf), Lcfs, LcfsConj, WInfos, increment,\
                                                                       freqs, Lmaps, LJones, SmearMapping, Sparsification,\
                                                                       LOptimisation,LSmearing,np_ChanMapping, expstokes); \
            done=true;\
      }
    using namespace DDF::gridder::policies;
    bool done=false;
    if (inputcorr==svec{"XX", "XY", "YX", "YY"})
      {
      #define readcorr gridder::policies::Read_4Corr
      #define mulaccum gridder::policies::Mulaccum_4Corr
      if (expstokes==svec{"I"})
	callgridder(I_from_XXXYYXYY, 1)
      else if (expstokes==svec{"Q"})
	callgridder(Q_from_XXXYYXYY, 1)
      else if (expstokes==svec{"U"})
	callgridder(U_from_XXXYYXYY, 1)
      else if (expstokes==svec{"V"})
	callgridder(V_from_XXXYYXYY, 1)
      else if (expstokes==svec{"I", "Q"})
	callgridder(IQ_from_XXXYYXYY, 2)
      else if (expstokes==svec{"I", "U"})
	callgridder(IQ_from_XXXYYXYY, 2)
      else if (expstokes==svec{"I", "V"})
	callgridder(IV_from_XXXYYXYY, 2)
      else if (expstokes==svec{"Q", "U"})
	callgridder(QU_from_XXXYYXYY, 2)
      else if (expstokes==svec{"I", "Q", "U"})
	callgridder(IQU_from_XXXYYXYY, 3)
      else if (expstokes==svec{"I", "Q", "U", "V"})
	callgridder(IQUV_from_XXXYYXYY, 4)
      #undef readcorr
      #undef mulaccum
      }
    if (inputcorr==svec{"RR", "RL", "LR", "LL"})
      {
      #define readcorr gridder::policies::Read_4Corr
      #define mulaccum gridder::policies::Mulaccum_4Corr
      if (expstokes==svec{"I"})
	callgridder(I_from_RRRLLRLL, 1)
      else if (expstokes==svec{"I", "Q"})
	callgridder(IQ_from_RRRLLRLL, 2)
      else if (expstokes==svec{"I", "U"})
	callgridder(IU_from_RRRLLRLL, 2)
      else if (expstokes==svec{"I", "V"})
	callgridder(IV_from_RRRLLRLL, 2)
      else if (expstokes==svec{"Q", "U"})
	callgridder(QU_from_RRRLLRLL, 2)
      else if (expstokes==svec{"I", "Q", "U"})
	callgridder(IQU_from_RRRLLRLL, 3)
      else if (expstokes==svec{"I", "Q", "U", "V"})
	callgridder(IQUV_from_RRRLLRLL, 4)
      #undef readcorr
      #undef mulaccum
      }
    else if (inputcorr==svec{"XX", "YY"})
      {
      #define readcorr gridder::policies::Read_2Corr_Pad
      #define mulaccum gridder::policies::Mulaccum_2Corr_Unpad
      if (expstokes==svec{"I"})
	callgridder(I_from_XXYY, 1)
      else if (expstokes==svec{"I", "Q"})
	callgridder(IQ_from_XXYY, 2)
      #undef readcorr
      #undef mulaccum
      }
    else if (inputcorr==svec{"RR", "LL"})
      {
      #define readcorr gridder::policies::Read_2Corr_Pad
      #define mulaccum gridder::policies::Mulaccum_2Corr_Unpad
      if (expstokes==svec{"I"})
	callgridder(I_from_RRLL, 1)
      else if (expstokes==svec{"I", "V"})
	callgridder(IV_from_RRLL, 2)
      #undef readcorr
      #undef mulaccum
      }
    else if (inputcorr==svec{"I"})
      {
      #define readcorr gridder::policies::Read_1Corr_Pad
      #define mulaccum gridder::policies::Mulaccum_1Corr_Unpad
      if (expstokes==svec{"I"})
	callgridder(I_from_I, 1)
      #undef readcorr
      #undef mulaccum
      }
    if (!done)
      throw std::invalid_argument("Cannot convert input correlations to desired output Stokes parameters.");
  }


  py::array_t<std::complex<float>, py::array::c_style> & pyDeGridderWPol(
			    const py::array_t<std::complex<float>, py::array::c_style>& np_grid,
			    py::array_t<std::complex<float>, py::array::c_style>& np_vis,
			    const py::array_t<double, py::array::c_style>& uvw,
			    const py::array_t<bool, py::array::c_style>& flags,
			    py::array_t<double, py::array::c_style>& /*sumwt*/,
			    bool /*dopsf*/,
			    const py::list& Lcfs,
			    const py::list& LcfsConj,
			    const py::array_t<double, py::array::c_style>& WInfos,
			    const py::array_t<double, py::array::c_style>& increment,
			    const py::array_t<double, py::array::c_style>& freqs,
			    const py::list& Lmaps,
			    py::list& LJones,
			    const py::array_t<int32_t, py::array::c_style>& SmearMapping,
			    const py::array_t<bool, py::array::c_style>& /*Sparsification*/,
			    const py::list& LOptimisation,
			    const py::list& LSmear,
			    const py::array_t<int, py::array::c_style>& np_ChanMapping,
			    const py::array_t<uint16_t, py::array::c_style>& LDataCorrFormat,
			    const py::array_t<uint16_t, py::array::c_style>& LExpectedOutStokes
		    )
  {
      using svec = vector<string>;
      const svec stokeslookup = {"undef","I","Q","U","V","RR","RL","LR","LL","XX","XY","YX","YY"};
      size_t ncorr = LDataCorrFormat.shape(0);
      size_t npol = LExpectedOutStokes.shape(0);
      svec inputcorr(ncorr), expstokes(npol);
    for (size_t i=0; i<ncorr; ++i)
      {
      const uint16_t corrid = LDataCorrFormat.data(0)[i];
      if (!(corrid>=5 && corrid<=12 || corrid==1))
	throw std::invalid_argument("Only accepts I,RR,RL,LR,LL,XX,XY,YX,YY as correlation output types");
      inputcorr[i] = stokeslookup[corrid];
      }
    for (size_t i=0; i<npol; ++i)
      {
      const uint16_t polid = LExpectedOutStokes.data(0)[i];
      expstokes[i] = stokeslookup[polid];
      }
    bool done=false;
    #define CALL_DEGRIDDER(STOKES, NVISPOL, NVISCORR)\
      {\
      DDF::degridder::degridder<STOKES, NVISPOL, NVISCORR>(np_grid, np_vis, uvw, flags, Lcfs, LcfsConj, WInfos, increment, freqs, Lmaps, LJones, SmearMapping, LOptimisation, LSmear,np_ChanMapping);\
      done=true;\
      }
    using namespace DDF::degridder::policies;

      
    if (expstokes==svec{"I"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
	CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_I, 1, 4)
      else if (inputcorr==svec{"XX", "YY"})
	CALL_DEGRIDDER(gmode_corr_XXYY_from_I, 1, 2)
      else if (inputcorr==svec{"RR", "RL", "LR", "LL"})
	CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_I, 1, 4)
      else if (inputcorr==svec{"RR", "LL"})
	CALL_DEGRIDDER(gmode_corr_RRLL_from_I, 1, 2)
      else if (inputcorr==svec{"I"})
	CALL_DEGRIDDER(gmode_corr_I_from_I, 1, 1)
      }
    if (expstokes==svec{"Q"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
	CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_Q, 1, 4)
      }
    if (expstokes==svec{"U"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
	CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_U, 1, 4)
      }
    if (expstokes==svec{"V"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
	CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_V, 1, 4)
      }
      // below are new scripts by FG 20210610
    if (expstokes==svec{"I","Q","U","V"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
        CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_IQUV, 4, 4)
      else if (inputcorr==svec{"RR", "RL", "LR", "LL"})
        CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_IQUV, 4, 4)
      }
      // BH 20220222 additional cases for cleaning dual polarized Q or V flux
      // or cases where only Q or V is cleaned for calibrating I on linearly or circularly
      // (respectively) flux
    if (expstokes==svec{"I","Q"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
        CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_IQ, 2, 4)
      else if (inputcorr==svec{"RR", "RL", "LR", "LL"})
        CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_IQ, 2, 4)
      else if (inputcorr==svec{"XX", "YY"})
       CALL_DEGRIDDER(gmode_corr_XXYY_from_IQ, 2, 2)
      }
    if (expstokes==svec{"I","U"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
        CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_IU, 2, 4)
      else if (inputcorr==svec{"RR", "RL", "LR", "LL"})
        CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_IU, 2, 4)
      //else if (inputcorr==svec{"XX", "YY"})
      //  CALL_DEGRIDDER(gmode_corr_XXYY_from_IQ, 2, 2)
      }
    if (expstokes==svec{"I","V"})
      {
      if (inputcorr==svec{"RR", "RL", "LR", "LL"})
        CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_IV, 2, 4)
      else if (inputcorr==svec{"XX", "XY", "YX", "YY"})
        CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_IV, 2, 4)
      else if (inputcorr==svec{"RR", "LL"})
       CALL_DEGRIDDER(gmode_corr_RRLL_from_IV, 2, 2)
      }
    if (expstokes==svec{"Q","U"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
        CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_QU, 2, 4)
      //else if (inputcorr==svec{"XX", "YY"})
      //  CALL_DEGRIDDER(gmode_corr_XXYY_from_IQ, 2, 2)
      }
    if (expstokes==svec{"I","Q","U"})
      {
      if (inputcorr==svec{"XX", "XY", "YX", "YY"})
        CALL_DEGRIDDER(gmode_corr_XXXYYXYY_from_IQU, 3, 4)
      else if (inputcorr==svec{"RR", "RL", "LR", "LL"})
        CALL_DEGRIDDER(gmode_corr_RRRLLRLL_from_IQU, 3, 4)
      //else if (inputcorr==svec{"XX", "YY"})
      //  CALL_DEGRIDDER(gmode_corr_XXYY_from_IQ, 2, 2)
      }
    if (!done)
      {
      throw std::invalid_argument("Cannot convert input Stokes parameter to desired output correlations.");
      }
    return np_vis;
  }
    #if PY_MAJOR_VERSION >= 3
    PYBIND11_MODULE(_pyGridderSmearPols3x, m) {
    #else
    PYBIND11_MODULE(_pyGridderSmearPols27, m) {
    #endif
    m.doc() = "DDFacet Directional Dependent BDA gridding module";
    m.def("pyAccumulateWeightsOntoGrid",
	  &pyAccumulateWeightsOntoGrid);
    m.def("pyAccumulateWeightsOntoGridNoSem",
	  &pyAccumulateWeightsOntoGridNoSem);
    m.def("pyGridderWPol32",
	  &pyGridderWPol<float>);
    m.def("pyGridderWPol64",
	  &pyGridderWPol<double>);
    m.def("pyDeGridderWPol",
	  &pyDeGridderWPol,
	  py::return_value_policy::take_ownership);
    m.def("pySetSemaphores",
	  &pySetSemaphores);
    m.def("pyDeleteSemaphore",
	  &pyDeleteSemaphore);
  }
} // DDF namespace
