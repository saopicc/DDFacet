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
#include "shared_array.h"

/*
 * Deallocation function
 */
static void leon_dealloc(PyLeonObject *op)
{
	/* Unmap the data */
	if (munmap(op->data, op->size) < 0)
		PyErr_SetFromErrno(PyExc_RuntimeError);
}

/*
 * SharedArrayObject type definition
 */
PyTypeObject PyLeonObject_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"shared_array.leon",			/* tp_name		*/
	sizeof (PyLeonObject),			/* tp_basicsize		*/
	0,					/* tp_itemsize		*/
	(destructor) leon_dealloc,		/* tp_dealloc		*/
	0,					/* tp_print		*/
	0,					/* tp_getattr		*/
	0,					/* tp_setattr		*/
	0,					/* tp_reserved		*/
	0,					/* tp_repr		*/
	0,					/* tp_as_number		*/
	0,					/* tp_as_sequence	*/
	0,					/* tp_as_mapping	*/
	0,					/* tp_hash		*/
	0,					/* tp_call		*/
	0,					/* tp_str		*/
	0,					/* tp_getattro		*/
	0,					/* tp_setattro		*/
	0,					/* tp_as_buffer		*/
	Py_TPFLAGS_DEFAULT,			/* tp_flags		*/
	0,					/* tp_doc		*/
	0,					/* tp_traverse		*/
	0,					/* tp_clear		*/
	0,					/* tp_richcompare	*/
	0,					/* tp_weaklistoffset	*/
	0,					/* tp_iter		*/
	0,					/* tp_iternext		*/
	0,					/* tp_methods		*/
	0,					/* tp_members		*/
	0,					/* tp_getset		*/
	0,					/* tp_base		*/
	0,					/* tp_dict		*/
	0,					/* tp_descr_get		*/
	0,					/* tp_descr_set		*/
	0,					/* tp_dictoffset	*/
	0,					/* tp_init		*/
	0,					/* tp_alloc		*/
	0,					/* tp_new		*/
	0,					/* tp_free		*/
	0,					/* tp_is_gc		*/
	0,					/* tp_bases		*/
	0,					/* tp_mro		*/
	0,					/* tp_cache		*/
	0,					/* tp_subclasses	*/
	0,					/* tp_weaklist		*/
	0,					/* tp_del		*/
	0,					/* tp_version_tag	*/
};
