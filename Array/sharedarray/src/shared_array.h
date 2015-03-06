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

#ifndef __SHARED_ARRAY_H__
#define __SHARED_ARRAY_H__

#include <Python.h>
#include <structseq.h>
#include <numpy/arrayobject.h>

/* Magic header */
#define SHARED_ARRAY_MAGIC	"[SharedArray]"

/* Maximum number of dimensions */
#define SHARED_ARRAY_NDIMS_MAX	16

/* Array metadata */
struct array_meta {
	char	magic[16];
	size_t	size;
	int	typenum;
	int	ndims;
	npy_intp dims[SHARED_ARRAY_NDIMS_MAX];
} __attribute__ ((packed));

/* ArrayDesc object */
extern PyStructSequence_Desc PyArrayDescObject_Desc;
extern PyTypeObject PyArrayDescObject_Type;

/* Leon object */
typedef struct {
	PyObject_HEAD
	void	*data;
	size_t	size;
} PyLeonObject;

extern PyTypeObject PyLeonObject_Type;

/* Module functions */
extern PyObject *shared_array_create(PyObject *self, PyObject *args, PyObject *kw);
extern PyObject *shared_array_attach(PyObject *self, PyObject *args);
extern PyObject *shared_array_delete(PyObject *self, PyObject *args);
extern PyObject *shared_array_list(PyObject *self, PyObject *args);

#endif /* !__SHARED_ARRAY_H__ */
