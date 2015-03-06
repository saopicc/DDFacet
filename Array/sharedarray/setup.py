#!/usr/bin/env python
#
# This file is part of SharedArray.
# Copyright (C) 2014 Mathieu Mirmont <mat@parad0x.org>
#
# SharedArray is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# SharedArray is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SharedArray.  If not, see <http://www.gnu.org/licenses/>.

from distutils.core import setup, Extension
from glob import glob
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(name    = 'SharedArray',
      version = '0.1',

      # Description
      description      = 'Share numpy arrays between processes',
      long_description = long_description,

      # Contact
      author       = 'Mathieu Mirmont',
      author_email = 'mat@parad0x.org',
      url          = 'http://parad0x.org/git/python/shared-array/about',

      # License
      license   = 'https://www.gnu.org/licenses/gpl-2.0.html',

      # Extras for pip
      keywords  = 'numpy array shared memory shm',
      classifiers  = [
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: C',
          'Topic :: Scientific/Engineering'
      ],

      # Compilation
      ext_modules  = [
          Extension('SharedArray',
                    glob(path.join(here, 'src', '*.c')),
                    libraries = [ 'rt' ])
      ])
