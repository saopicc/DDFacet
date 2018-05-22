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

#ifndef GRIDDER_SEMAPHORES_H
#define GRIDDER_SEMAPHORES_H

#include <fcntl.h>           /* For O_* constants */
#include <vector>
#include <semaphore.h>
#include <string>
#include "common.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
namespace DDF {
  const char *GiveSemaphoreName(size_t iS);
  sem_t *GiveSemaphoreFromCell(size_t irow);
  sem_t *GiveSemaphoreFromID(size_t iS);
  void pySetSemaphores(const pybind11::list& LSemaphoreNames);
  void pyDeleteSemaphore();
}

#endif /*GRIDDER_SEMAPHORES_H*/
