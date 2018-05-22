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

#include "Semaphores.h"
namespace DDF { 
  static std::vector<sem_t *> Tab_SEM;
  static std::vector<std::string> sem_names;
  
  const char *GiveSemaphoreName(size_t iS){
    return sem_names[iS].c_str();
  }

  sem_t *GiveSemaphoreFromCell(size_t irow){
    return Tab_SEM[irow % Tab_SEM.size()];
  }

  sem_t *GiveSemaphoreFromID(size_t iS){
    const char* SemaphoreName=GiveSemaphoreName(iS);
    sem_t *Sem_mutex = sem_open(SemaphoreName, O_CREAT, 0644, 1);
    if (Sem_mutex != SEM_FAILED) 
      return Sem_mutex;
    else
      throw std::runtime_error("Failed to open semaphore");
  }


  void pySetSemaphores(const pybind11::list& LSemaphoreNames)
  {
    Tab_SEM.resize(LSemaphoreNames.size());
    sem_names.resize(LSemaphoreNames.size());
    for (size_t i=0; i<Tab_SEM.size(); ++i){
      sem_names[i] = std::string(pybind11::str(LSemaphoreNames[i]));
      Tab_SEM[i]=GiveSemaphoreFromID(i);
    }
  }

  void pyDeleteSemaphore()
  {
    for(size_t i=0; i<Tab_SEM.size(); ++i){
      const char* SemaphoreName=GiveSemaphoreName(i);
      sem_close(Tab_SEM[i]);
      sem_unlink(SemaphoreName);
    }
    Tab_SEM.resize(0);
    Tab_SEM.shrink_to_fit();
  }
}