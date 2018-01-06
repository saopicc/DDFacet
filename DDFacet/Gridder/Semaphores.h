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

#include <fcntl.h>           /* For O_* constants */
#include <vector>
#include <semaphore.h>
#include "common.h"

static PyObject *LSemaphoreNames;
static size_t NSemaphores;
static std::vector<sem_t *> Tab_SEM;

static const char *GiveSemaphoreName(size_t iS){
  return PyString_AsString(PyList_GetItem(LSemaphoreNames, (ssize_t)iS));
}

static sem_t * GiveSemaphoreFromCell(size_t irow){
  return Tab_SEM[irow % NSemaphores];
}

static sem_t * GiveSemaphoreFromID(size_t iS){
  const char* SemaphoreName=GiveSemaphoreName(iS);
  sem_t * Sem_mutex;
  if ((Sem_mutex = sem_open(SemaphoreName, O_CREAT, 0644, 1)) == SEM_FAILED) {
    perror("semaphore initialization");
    exit(1);
  }
  return Sem_mutex;
}


static PyObject *pySetSemaphores(PyObject */*self*/, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "O!",&PyList_Type, &LSemaphoreNames))  return NULL;
  NSemaphores=(size_t)PyList_Size(LSemaphoreNames);

  Tab_SEM.resize(NSemaphores);
  for(size_t iSemaphore=0; iSemaphore<NSemaphores; iSemaphore++)
    Tab_SEM[iSemaphore]=GiveSemaphoreFromID(iSemaphore);

  Py_INCREF(Py_None);
  return Py_None;
}

// MR FIXME: why pass the lists of name to the destructor?
// This can cause inconsistencies and has no advantage whatsoever
static PyObject *pyDeleteSemaphore(PyObject */*self*/, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "O!",&PyList_Type, &LSemaphoreNames)) return NULL;

  NSemaphores=(size_t)PyList_Size(LSemaphoreNames);

  for(size_t iSemaphore=0; iSemaphore<NSemaphores; iSemaphore++){
    const char* SemaphoreName=GiveSemaphoreName(iSemaphore);
    sem_close(Tab_SEM[iSemaphore]);
    sem_unlink(SemaphoreName);
  }
  Tab_SEM.resize(0);

  Py_INCREF(Py_None);
  return Py_None;
}
