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
#include <structseq.h>
#include <numpy/arrayobject.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include "shared_array.h"

/* SHM directory */
#define SHMDIR	"/dev/shm"

/*
 * List of meta-data
 */
struct list {
	struct list *next;
	PyObject *desc;
};

/*
 * Read the meta data from a shared memory area
 */
static int get_meta(const struct dirent *dirent, struct array_meta *meta)
{
	char path[PATH_MAX];
	int fd;

	/* Only accept regular files */
	if (dirent->d_type != DT_REG)
		return -1;

	/* Construct the file name */
	snprintf(path, sizeof (path), "%s/%s", SHMDIR, dirent->d_name);

	/* Open the file */
	if ((fd = open(path, O_RDONLY, 0666)) < 0)
		return -1;

	/* Read the meta data structure */
	if (read(fd, meta, sizeof (*meta)) != sizeof (*meta)) {
		close(fd);
		return -1;
	}

	/* Close the file */
	close(fd);

	/* Check the meta data */
	if (strncmp(meta->magic, SHARED_ARRAY_MAGIC, sizeof (meta->magic)))
		return -1;

	/* Accept it */
	return 1;
}

/*
 * Convert the dims array into a tuple
 */
static PyObject *convert_dims(int ndims, npy_intp *dims)
{
	PyObject *tuple = PyTuple_New(ndims);
	int i;

	for (i = 0; i < ndims; i++)
		PyTuple_SET_ITEM(tuple, i, PyLong_FromLong(dims[i]));
	return tuple;
}

/*
 * Add an element to the list
 */
static struct list *list_extend(struct list *next, struct array_meta *meta, const char *name)
{
	struct list *list;

	/* Allocate the new list element */
	if (!(list = malloc(sizeof (*list)))) {
		PyErr_NoMemory();
		return next;
	}

	/* Populate the list */
	list->next = next;
	list->desc = PyStructSequence_New(&PyArrayDescObject_Type);

	PyStructSequence_SET_ITEM(list->desc, 0, PyBytes_FromString(name));
	PyStructSequence_SET_ITEM(list->desc, 1, PyArray_TypeObjectFromType(meta->typenum));
	PyStructSequence_SET_ITEM(list->desc, 2, convert_dims(meta->ndims, meta->dims));

	/* Return the new element */
	return list;
}

/*
 * List all arrays created in shared memory
 */
static int build_list(struct list **list)
{
	struct dirent *ent;
	int size = 0;
	DIR *dir;

	/* Start the list */
	*list = NULL;
	
	/* Open the directory */
	if (!(dir = opendir(SHMDIR))) {
		PyErr_SetFromErrnoWithFilename(PyExc_OSError, SHMDIR);
		return -1;
	}

	/* Process each directory entry */
	while ((ent = readdir(dir))) {
		struct array_meta meta;

		/* Add valid entries to the list */
		if (get_meta(ent, &meta) > 0) {
			*list = list_extend(*list, &meta, ent->d_name);
			size++;
		}
	}

	/* Close the directory */
	closedir(dir);

	/* Return the size */
	return size;
}

/*
 * Free the list
 */
static void free_list(struct list *list)
{
	while (list) {
		struct list *next = list->next;

		free(list);
		list = next;
	}
}

/*
 * Build a tuple from the list
 */
static PyObject *build_tuple(struct list *list, int size)
{
	PyObject *tuple = PyTuple_New(size);
	int i;

	for (i = 0; i < size; i++) {
		PyTuple_SetItem(tuple, i, list->desc);
		list = list->next;
	}
	return tuple;
}

/*
 * Method: SharedArray.list()
 */
PyObject *shared_array_list(PyObject *self, PyObject *args)
{
	PyObject *tuple;
	struct list *list;
	int size;

	/* Parse the arguments */
	if (!PyArg_ParseTuple(args, ""))
		return NULL;

	/* Build the list of shared arrays */
	if ((size = build_list(&list)) < 0)
		return NULL;

	/* Convert the list into a tuple */
	tuple = build_tuple(list, size);

	/* Free the list */
	free_list(list);

	/* Return the tuple */
	return tuple;
}
