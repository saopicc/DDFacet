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

#include <Python.h>
#include <structseq.h>
#include <numpy/arrayobject.h>
#include "shared_array.h"

/* Module name */
static const char module_name[] = "SharedArray";

/* Module documentation string */
static const char module_docstring[] =
	"This module lets you share numpy arrays "
	"between several python interpreters";

/*
 * Module functions
 */
static PyMethodDef module_functions[] = {
	{ "create", (PyCFunction) shared_array_create,
	  METH_VARARGS | METH_KEYWORDS,
	  "Create a numpy array in shared memory" },

	{ "attach", (PyCFunction) shared_array_attach,
	  METH_VARARGS,
	  "Attach an existing numpy array from shared memory" },

	{ "delete", (PyCFunction) shared_array_delete,
	  METH_VARARGS,
	  "Delete an existing numpy array from shared memory" },

	{ "list", (PyCFunction) shared_array_list,
	  METH_VARARGS,
	  "List all existing numpy arrays from shared memory" },

	{ NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3

/*
 * Module definition
 */
static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
	module_name,		/* m_name	*/
	module_docstring,	/* m_doc	*/
        -1,			/* m_size	*/
        module_functions,	/* m_methods	*/
        NULL,			/* m_reload	*/
        NULL,			/* m_traverse	*/
        NULL,			/* m_clear	*/
        NULL,			/* m_free	*/
};

/* Module creation function for python 3 */
#define CREATE_MODULE(NAME, FUNCTIONS, DOCSTRING)	\
	PyModule_Create(&module_def)
#else
/* Module creation function for python 2 */
#define CREATE_MODULE(NAME, FUNCTIONS, DOCSTRING)	\
	Py_InitModule3(NAME, FUNCTIONS, DOCSTRING)
#endif

/*
 * Module initialisation
 */
static PyObject *module_init(void)
{
	PyObject *m;

	/* Import numpy arrays */
	import_array();

	/* Register the module */
	if (!(m = CREATE_MODULE(module_name, module_functions, module_docstring)))
		return NULL;

	/* Register the Leon type */
	PyType_Ready(&PyLeonObject_Type);
	Py_INCREF(&PyLeonObject_Type);
	PyModule_AddObject(m, module_name, (PyObject *) &PyLeonObject_Type);

	/* Register the Descr type */
	PyStructSequence_InitType(&PyArrayDescObject_Type, &PyArrayDescObject_Desc);
	PyType_Ready(&PyArrayDescObject_Type);
	Py_INCREF(&PyArrayDescObject_Type);
	PyModule_AddObject(m, module_name, (PyObject *) &PyArrayDescObject_Type);

	return m;
}

/*
 * Python 2.7 compatibility blob
 */
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_SharedArray(void)
{
	return module_init();
}
#else
PyMODINIT_FUNC initSharedArray(void)
{
	module_init();
}
#endif
