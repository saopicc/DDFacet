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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "shared_array.h"

/*
 * Delete a numpy array from shared memory
 */
static PyObject *do_delete(const char *name)
{
	struct array_meta meta;
	int fd;
	int size;

	/* Open the shm block */
	if ((fd = shm_open(name, O_RDWR, 0)) < 0)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	/* Read the meta data structure */
	size = read(fd, &meta, sizeof (meta));
	close(fd);

	/* Catch read errors */
	if (size <= 0)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	/* Catch short reads */
	if (size != sizeof (meta)) {
		PyErr_SetString(PyExc_IOError, "No SharedArray at this address");
		return NULL;
	}

	/* Check the meta data */
	if (strncmp(meta.magic, SHARED_ARRAY_MAGIC, sizeof (meta.magic))) {
		PyErr_SetString(PyExc_IOError, "No SharedArray at this address");
		return NULL;
	}

	/* Unlink the shm block */
	if (shm_unlink(name) < 0)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	Py_RETURN_NONE;
}

/*
 * Method: SharedArray.delete()
 */
PyObject *shared_array_delete(PyObject *self, PyObject *args)
{
	const char *name;

	/* Parse the arguments */
	if (!PyArg_ParseTuple(args, "s", &name))
		return NULL;

	/* Now do the real thing */
	return do_delete(name);
}
