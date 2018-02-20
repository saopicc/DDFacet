#pragma once
#include <cmath>
#include "common.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/pytypes.h"
namespace {
  namespace py=pybind11;
  using namespace std;
  class DecorrelationHelper
    {
    private:
      double DT, Dnu, l0, m0;
      const double *uvw_Ptr, *uvw_dt_Ptr;
      bool DoDecorr, TSmear, FSmear;

    public:
      DecorrelationHelper(const py::list& LSmearing, 
			  const py::array_t<double, py::array::c_style>& uvw)
	{
	  
	  DoDecorr=(LSmearing.size() > 0);
	  if (DoDecorr)
	    {
	    uvw_Ptr = uvw.data(0);
	    uvw_dt_Ptr = py::array_t<double, py::array::c_style>(LSmearing(0)).data(0);
	    DT = LSmearing(1).cast<double>();
	    Dnu = LSmearing(2).cast<double>();
	    TSmear = LSmearing(3).cast<bool>();
	    FSmear = LSmearing(4).cast<bool>();
	    l0 = LSmearing(5).cast<double>();
	    m0 = LSmearing(6).cast<double>();
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
}