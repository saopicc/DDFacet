/* 
 * This file is part of SharedArray.
 * Copyright (C) 2014 Mathieu Mirmont <mat@parad0x.org>
 * 
 * SharedArray is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * SharedArray is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with SharedArray.  If not, see <http://www.gnu.org/licenses/>.
 */

#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL	SHARED_ARRAY_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "shared_array.h"

/*
 * Attach a numpy array from shared memory
 */
static PyObject *do_attach(const char *name)
{
	struct array_meta meta;
	void *data;
	int fd;
	PyObject *ret;
	PyLeonObject *leon;

	/* Open the shm block */
	if ((fd = shm_open(name, O_RDWR, 0)) < 0)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	/* Read the meta data structure */
	if (read(fd, &meta, sizeof (meta)) != sizeof (meta)) {
		close(fd);
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
	}

	/* Check the meta data */
	if (strncmp(meta.magic, SHARED_ARRAY_MAGIC, sizeof (meta.magic))) {
		close(fd);
		PyErr_SetString(PyExc_IOError,
				"No SharedArray at this address");
		return NULL;
	}

	/* Check the number of dimensions */
	if (meta.ndims > SHARED_ARRAY_NDIMS_MAX) {
		close(fd);
		PyErr_SetString(PyExc_ValueError,
				"Too many dimensions, recompile SharedArray!");
		return NULL;
	}
	
	/* Map the array data */
	data = mmap(NULL, meta.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	//data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (data == MAP_FAILED)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	/* Summon Leon */
	leon = PyObject_MALLOC(sizeof (*leon));
	PyObject_INIT((PyObject *) leon, &PyLeonObject_Type);
	leon->data = data;
	leon->size = meta.size;

	/* printf("B %i, %i, %i, %i [",meta.ndims, meta.typenum, meta.size, fd); */
	/* int i; */
	/* for(i=0; i<meta.ndims; i++){printf("%i,",meta.dims[i]);} */
	/* printf("]\n"); */
	/* printf("meta %i\n",sizeof (meta)); */


	/* Create the array object */
	ret = PyArray_SimpleNewFromData(meta.ndims, meta.dims, meta.typenum,
					data + sizeof (meta));

	/* Attach Leon to the array */
	PyArray_SetBaseObject((PyArrayObject *) ret, (PyObject *) leon);
	return ret;
}

/*
 * Method: SharedArray.attach()
 */
PyObject *shared_array_attach(PyObject *self, PyObject *args)
{
	const char *name;

	/* Parse the arguments */
	if (!PyArg_ParseTuple(args, "s", &name))
		return NULL;

	/* Now do the real thing */
	return do_attach(name);
}
