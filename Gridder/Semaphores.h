#include <semaphore.h>

PyObject *LSemaphoreNames;
int NSemaphores;
sem_t **Tab_SEM;

const char *GiveSemaphoreName(int iS){
  char*  SemaphoreName=PyString_AsString(PyList_GetItem(LSemaphoreNames, iS));
  return SemaphoreName;
}



sem_t * GiveSemaphoreFromCell(size_t irow){
  sem_t *Sem_mutex;
  int index=irow-NSemaphores*(irow / NSemaphores);
  //printf("fetching sem %i for irow=%i\n",index,irow);
  Sem_mutex=Tab_SEM[index];
  return Sem_mutex;
}


sem_t * GiveSemaphoreFromID(int iS){
  const char* SemaphoreName=GiveSemaphoreName(iS);
  sem_t * Sem_mutex;
  if ((Sem_mutex = sem_open(SemaphoreName, O_CREAT, 0644, 1)) == SEM_FAILED) {
    perror("semaphore initilization");
    exit(1);
  }
  return Sem_mutex;
}


static PyObject *pySetSemaphores(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "O!",&PyList_Type, &LSemaphoreNames))  return NULL;
  NSemaphores=PyList_Size(LSemaphoreNames);


  Tab_SEM=calloc(1,(NSemaphores)*sizeof(sem_t*));
  int iSemaphore=0;
  for(iSemaphore=0; iSemaphore<NSemaphores; iSemaphore++){
    sem_t * SEM=GiveSemaphoreFromID(iSemaphore);
    Tab_SEM[iSemaphore]=SEM;
  }
  Py_INCREF(Py_None);
  return Py_None;

}




static PyObject *pyDeleteSemaphore(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "O!",&PyList_Type, &LSemaphoreNames))  return NULL;

  NSemaphores=PyList_Size(LSemaphoreNames);
  
  int iSemaphore=0;
  for(iSemaphore=0; iSemaphore<NSemaphores; iSemaphore++){
    const char* SemaphoreName=GiveSemaphoreName(iSemaphore);
    //printf("delete %s\n",SemaphoreName);
    int ret=sem_unlink(SemaphoreName);
  }
  free(Tab_SEM);
  

  Py_INCREF(Py_None);
  return Py_None;

}


