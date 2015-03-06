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
#include <Python.h>
#include <structseq.h>

/*
 * List of fields
 */
static PyStructSequence_Field fields[] = {
	{ "name",	"Array name"	},
	{ "dtype",	"Data type"	},
	{ "dims",	"Dimensions"	},
	{ NULL, NULL }
};

/*
 * Struct sequence description
 */
PyStructSequence_Desc PyArrayDescObject_Desc = {
	"ArrayDesc",
	"Description of an attachable numpy shared array",
	fields,
	3,
};

/*
 * Type definition
 */
PyTypeObject PyArrayDescObject_Type;
