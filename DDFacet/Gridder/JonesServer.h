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

#ifndef GRIDDER_JONESSERV_H
#define GRIDDER_JONESSERV_H

#include "common.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace DDF {
  namespace py=pybind11;
  class NotImplemented : public std::logic_error
	{
	public:
    	NotImplemented() : std::logic_error("Function not yet implemented") { };
	};

  template<typename T> class Mat
    {
    private:
      T v[4];

    public:
      Mat() {}
      Mat(T v0, T v1, T v2, T v3)
	: v { v0, v1, v2, v3 } {}
      T &operator[](size_t i) { return v[i]; }
	  Mat<T> operator*(T rhs) const
	{
		throw NotImplemented();
	}
	  Mat<T> operator*(float rhs) const
	{
		return Mat<T>(v[0] * rhs, v[1] * rhs, v[2] * rhs, v[3] * rhs);
	}
	  Mat<T> operator*(double rhs) const
	{
		return Mat<T>(v[0] * rhs, v[1] * rhs, v[2] * rhs, v[3] * rhs);
	}
      const T &operator[](size_t i) const { return v[i]; }
      void setZero()
	{ v[0]=v[1]=v[2]=v[3]=T(0); }
      void setUnity()
	{
	v[0]=v[3]=T(1);
	v[1]=v[2]=T(0);
	}
      void set(T v0, T v1, T v2, T v3)
	{ v = { v0, v1, v2, v3 }; }
      Mat hermitian() const
	{ return Mat(conj(v[0]), conj(v[2]), conj(v[1]), conj(v[3])); }
      void scale(T lambda)
	{ v[0]*=lambda; v[1]*=lambda; v[2]*=lambda; v[3]*=lambda; }
      Mat times(const Mat &b) const
	{
	return Mat(v[0]*b.v[0]+v[1]*b.v[2],
		  v[0]*b.v[1]+v[1]*b.v[3],
		  v[2]*b.v[0]+v[3]*b.v[2],
		  v[2]*b.v[1]+v[3]*b.v[3]);
	}
    };

  using dcMat = Mat<dcmplx>;
  namespace DDEs {
    class JonesServer
      {
      private:
	const fcmplx *ptrJonesMatrices, *ptrJonesMatrices_Beam;
	const int *ptrTimeMappingJonesMatrices, *ptrTimeMappingJonesMatrices_Beam,
		  *ptrVisToJonesChanMapping_killMS,*ptrVisToJonesChanMapping_Beam,
		  *ptrA0, *ptrA1;
	const float *ptrCoefsInterp;
	int i_dir_kMS;

	const float *ptrAlphaReg_killMS;

	int na_AlphaReg;
	int JonesDims_Beam[4];
	bool ApplyJones_Beam=false;
	int i_dir_Beam;
	bool ApplyJones_killMS=false;
	int nt_Jones;

	int JonesDims[4];
	int ModeInterpolation=1;
	bool ApplyAmp, ApplyPhase, DoScaleJones;
	double CalibError, ReWeightSNR;

	double WaveLengthMean;
	bool Has_AlphaReg_killMS=false;

	int CurrentJones_kMS_Time=-1;
	int CurrentJones_kMS_Chan=-1;
	int CurrentJones_Beam_Time=-1;
	int CurrentJones_Beam_Chan=-1;
	int CurrentJones_ant0;
	int CurrentJones_ant1;

	void NormJones(dcMat &J0, const double *uvwPtr) const;
	static dcMat GiveJones(const fcmplx *ptrJonesMatrices, const int *JonesDims,
	  const float *ptrCoefs, int i_t, int i_ant0, int i_dir, int iChJones,
	  int Mode);
      public:
	//BH FIXME: Proper accessors pretty pretty please..
        //MR Some of these quantities are manipulated from the outside.
        //   I guess we need to address this properly first.
	double BB;
	double WeightVaryJJ;
	int DoApplyJones=0;
	dcMat J0, J1, J0H, J1H;

	double *ptrSumJones, *ptrSumJonesChan;

	JonesServer(py::list& LJones, double WaveLengthMeanIn);

	// updates Jones terms for given row and channel. Returns True if something has changed.
	bool updateJones(size_t irow, size_t visChan, const double *uvwPtr, bool EstimateWeight, bool DoApplyAlphaRegIn);
	void resetJonesServerCounter();

      private:
        dcMat J0Beam, J1Beam, J0kMS, J1kMS;
      };
  }
}
#endif /*GRIDDER_JONESSERV_H*/
