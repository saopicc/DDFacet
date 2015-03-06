SharedArray python/numpy extension
==================================

This is a simple python extension that lets you share numpy arrays
with other processes on the same computer.  It uses posix shared
memory internally and therefore should work on most operating systems.

Example
-------

Here's a simple example to give an idea of how it works. This example
does everything from a single python interpreter for the sake of
clarity, but the actual intentention is to share arrays between
interpreters.

	import numpy as np
	import SharedArray as sa

	# Create an array in shared memory
	a = sa.create("test1", 10)

	# Attach it as a different array. This can be done from another
	# python interpreter as long as it runs on the same computer.
	b = sa.attach("test1")

	# See how they are actually sharing the same memory block
	a[0] = 42
	print(b[0])

	# Destroying a does not affect b.
	del a
	print(b[0])

	# See how "test1" is still present in shared memory even though we
	# destroyed the array a.
	sa.list()

	# Now destroy the array "test1" from memory.
	sa.delete("test1")

	# The array b is not affected, but once you destroy it then the
	# data are lost.
	print(b[0])

Functions
---------

### `SharedArray.create(name, shape, dtype=float)`

This function creates an array in shared memory identified by `name`.
The `shape` and `dtype` arguments are the same as the numpy function
`numpy.zeros()`.  The returned array is initialized to zero.  The
shared memory block holding the content of the array will not be
deleted when this array is destroyed, either implicitly or explicitly
by calling `del`, it will simply be detached from the current process.
To delete a shared array use the `SharedArray.delete()` function.

### `SharedArray.attach(name)`

This function attaches an array previously created in shared memory
and identified by `name`.  The shared memory block holding the content
of the array will not be deleted when this array is destroyed, either
implicitly or explicitly by calling `del`, it will simply be detached
from the current process.  To delete a shared array use the
`SharedArray.delete()` function.

### `SharedArray.delete(name)`

This function destroys an array previously created in shared memory
and identified by `name`.  After calling `delete`, the array will not
be attachable anymore, but existing attachments will remain valid
until they are themselves destroyed.

### `SharedArray.list()`

This function returns a list of previously created shared arrays,
their name, data type and dimensions.  At the moment this function
only works on Linux because it accesses files exposed under
`/dev/shm`.  There doesn't seem to be a portable method of doing that.

Requirements
------------

* Python 2.7 or 3+
* Numpy 1.8
* Posix shared memory interface

SharedArray uses the posix shm interface (`shm_open` and `shm_unlink`)
and so should work on most operating systems that follow the posix
standards (Linux, *BSD, etc.).

Installation
------------

The extension uses the `distutils` python package that should be
familiar to most python users. To test the extension directly from the
source tree, without installing, type:

	python setup.py build_ext --inplace

To build and install the extension system-wide, type:

	python setup.py build
	sudo python setup.py install
