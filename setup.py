#!/usr/bin/python
'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import subprocess
import os
import warnings
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from distutils.command.build import build
from os.path import join as pjoin
import sys

pkg='DDFacet'
skymodel_pkg='SkyModel'
__version__ = "0.3.4.1"
build_root=os.path.dirname(__file__)

def backend(compile_options):
    if compile_options is not None:
        print >> sys.stderr, "Compiling extension libraries with user defined options: '%s'"%compile_options
    path = pjoin(build_root, pkg, 'cbuild')
    try:
        subprocess.check_call(["mkdir", path])
    except:
        warnings.warn("%s already exists in your source folder. We will not create a fresh build folder, but you "
                      "may want to remove this folder if the configuration has changed significantly since the "
                      "last time you run setup.py" % path)
    subprocess.check_call(["cd %s && cmake %s .. && make" %
                           (path, compile_options if compile_options is not None else ""), ""], shell=True)

class custom_install(install):
    install.user_options = install.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        install.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        install.run(self)

class custom_build(build):
    build.user_options = build.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        build.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        build.run(self)

class custom_sdist(sdist):
    def run(self):
        bpath = pjoin(build_root, pkg, 'cbuild')
        if os.path.isdir(bpath):
            subprocess.check_call(["rm", "-rf", bpath])
        sdist.run(self)

def define_scripts():
    #these must be relative to setup.py according to setuputils
    return [os.path.join(pkg, script_name) for script_name in ['DDF.py', 'CleanSHM.py', 'MemMonitor.py', 'Restore.py', 'SelfCal.py']]

setup(name=pkg,
      version=__version__,
      description='Facet-based radio astronomy continuum imager',
      url='http://github.com/saopicc/DDFacet',
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Cyril Tasse',
      author_email='cyril.tasse@obspm.fr',
      license='GNU GPL v2',
      cmdclass={'install': custom_install,
                'build': custom_build,
                'sdist': custom_sdist,
               },
      packages=[pkg, skymodel_pkg],
      install_requires=[
            "nose >= 1.3.7",
            "Cython >= 0.25.2",
            "numpy >= 1.11.0",
            "SharedArray >= 2.0.2",
            "Polygon2 >= 2.0.8",
            "pyFFTW >= 0.10.4",
            "astropy >= 1.3.3",
            "deap >= 1.0.1", 
            "ipdb >= 0.10.3",
            "python-casacore >= 2.1.0",
            "pyephem >= 3.7.6.0",
            "numexpr >= 2.6.2",
            "matplotlib >= 2.0.0",
            "scipy >= 0.16.0",
            "astro-kittens >= 0.3.3",
            "meqtrees-cattery >= 1.5.1",
            "owlcat >= 1.4.2",
            "astLib >= 0.8.0",
            "psutil >= 5.2.2",
            "py-cpuinfo >= 3.2.0",
            "tables >= 3.3.0",
            "prettytable >= 0.7.2"
      ],
      include_package_data=True,
      zip_safe=False,
      scripts=define_scripts()
)
