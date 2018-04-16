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

#ifndef GRIDDER_DECORR_H
#define GRIDDER_DECORR_H

#include <cmath>
#include "common.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
namespace DDF{
  namespace py=pybind11;
  using namespace std;
  namespace DDEs {
    class DecorrelationHelper
      {
      private:
	double DT, Dnu, l0, m0;
	const double *uvw_Ptr, *uvw_dt_Ptr;
	bool DoDecorr, TSmear, FSmear;

      public:
	DecorrelationHelper(const py::list& LSmearing,
			    const py::array_t<double, py::array::c_style>& uvw);
	double get(double nu, size_t idx);
      };
  }
}
#endif /*GRIDDER_DECORR_H*/
