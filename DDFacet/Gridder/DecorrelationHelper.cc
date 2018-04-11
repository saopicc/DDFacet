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

#include "DecorrelationHelper.h"

namespace DDF {
  namespace DDEs {
    DecorrelationHelper::DecorrelationHelper(const py::list& LSmearing,
					    const py::array_t<double, py::array::c_style>& uvw)
	  {

	    DoDecorr=(LSmearing.size() > 0);
	    if (DoDecorr)
	      {
	      uvw_Ptr = uvw.data(0);
	      uvw_dt_Ptr = py::array_t<double, py::array::c_style>(LSmearing[0]).data(0);
	      DT = LSmearing[1].cast<double>();
	      Dnu = LSmearing[2].cast<double>();
	      TSmear = LSmearing[3].cast<bool>();
	      FSmear = LSmearing[4].cast<bool>();
	      l0 = LSmearing[5].cast<double>();
	      m0 = LSmearing[6].cast<double>();
	      }
	  }
    double DecorrelationHelper::get(double nu, size_t idx)
      {
      if (!DoDecorr) return 1.;

      double n0=sqrt(1.-l0*l0-m0*m0)-1.;
      double DecorrFactor=1.;

      if (FSmear)
	{
	double phase = uvw_Ptr[3*idx+0]*l0 + uvw_Ptr[3*idx+1]*m0 + uvw_Ptr[3*idx+2]*n0;
	double phi=PI*Dnu/C*phase;
	DecorrFactor *= (phi==0.) ? 1.0 : sin(phi)/phi;
	}

      if (TSmear)
	{
	double dphase = (uvw_dt_Ptr[3*idx+0]*l0 + uvw_dt_Ptr[3*idx+1]*m0 + uvw_dt_Ptr[3*idx+2]*n0)*DT;
	double phi=PI*nu/C*dphase;
	DecorrFactor *= (phi==0.) ? 1.0 : sin(phi)/phi;
	}
      return DecorrFactor;
      }
  }
}
