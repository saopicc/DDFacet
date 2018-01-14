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
static std::vector<sem_t *> Tab_SEM;

static const char *GiveSemaphoreName(size_t iS){
  return PyString_AsString(PyList_GetItem(LSemaphoreNames, (ssize_t)iS));
}

static sem_t *GiveSemaphoreFromCell(size_t irow){
  return Tab_SEM[irow % Tab_SEM.size()];
}

static sem_t *GiveSemaphoreFromID(size_t iS){
  const char* SemaphoreName=GiveSemaphoreName(iS);
  sem_t *Sem_mutex = sem_open(SemaphoreName, O_CREAT, 0644, 1);
  if (Sem_mutex != SEM_FAILED) return Sem_mutex;
  perror("semaphore initialization");
  exit(1);
}


static PyObject *pySetSemaphores(PyObject */*self*/, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "O!",&PyList_Type, &LSemaphoreNames))  return NULL;

  Tab_SEM.resize(PyList_Size(LSemaphoreNames));
  for (size_t i=0; i<Tab_SEM.size(); ++i)
    Tab_SEM[i]=GiveSemaphoreFromID(i);
  Py_RETURN_NONE;
}

// MR FIXME: why pass the lists of names to the destructor?
// This can cause inconsistencies and has no advantage whatsoever
static PyObject *pyDeleteSemaphore(PyObject */*self*/, PyObject */*args*/)
{
  for(size_t i=0; i<Tab_SEM.size(); ++i){
    const char* SemaphoreName=GiveSemaphoreName(i);
    sem_close(Tab_SEM[i]);
    sem_unlink(SemaphoreName);
  }
  Tab_SEM.resize(0);
  Tab_SEM.shrink_to_fit();
  Py_RETURN_NONE;
}
