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
import sys
import warnings
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from distutils.command.build import build
from setuptools.command.build_ext import build_ext
from os.path import join as pjoin
import sys

pkg='DDFacet'
skymodel_pkg='SkyModel'
__version__ = "0.6.0.0"
build_root=os.path.dirname(__file__)

try:
    import pybind11
except ImportError as e:
    raise ImportError("Pybind11 not installed. Please install C++ binding package pybind11 before running DDFacet install. "
                      "You should not see this message unless you are not running pip install (19.x) -- run pip install!")

def backend(compile_options):
    if compile_options is not None:
        print("Compiling extension libraries with user defined options: '%s'"%compile_options)
    else:
        compile_options = ""
    
    compile_options += " -DENABLE_PYTHON_2=OFF "
    compile_options += " -DENABLE_PYTHON_3=ON "

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

class custom_build_ext(build_ext):
    build.user_options = build.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        build_ext.run(self)

class custom_sdist(sdist):
    def run(self):
        bpath = pjoin(build_root, pkg, 'cbuild')
        if os.path.isdir(bpath):
            subprocess.check_call(["rm", "-rf", bpath])
        sdist.run(self)

def define_scripts():
    #these must be relative to setup.py according to setuputils
    DDF_scripts = [os.path.join(pkg, script_name) for script_name in ['DDF.py', 'CleanSHM.py', 'MemMonitor.py', 'Restore.py', 'SelfCal.py']]
    SkyModel_scripts = [os.path.join(skymodel_pkg, script_name) for script_name in ['ClusterCat.py', 'dsm.py', 'dsreg.py', 'ExtractPSources.py', 
        'Gaussify.py', 'MakeCatalog.py', 'MakeMask.py', 'MakeModel.py', 'MaskDicoModel.py', 'MyCasapy2bbs.py', 'MaskDicoModel.py']]
    return DDF_scripts + SkyModel_scripts

def readme():
    """ Return README.rst contents """
    with open('README.rst') as f:
        return f.read()

def requirements():
    requirements = ["nose >= 1.3.7; python_version >= '3'", 
                    "Cython >= 0.25.2; python_version >= '3'", 
                    "numpy >= 1.15.1; python_version >= '3'", 
                    "sharedarray >= 3.2.0; python_version >= '3'", 
                    "Polygon3 >= 3.0.8; python_version >= '3'", 
                    "pyFFTW >= 0.10.4; python_version >= '3'", 
                    "astropy >= 3.0; python_version >= '3'", 
                    "deap >= 1.0.1; python_version >= '3'", 
                    "ptyprocess>=0.5; python_version >= '3'", 
                    "ipdb >= 0.10.3; python_version >= '3'", 
                    "python-casacore >= 3.0.0; python_version >= '3'", 
                    "pyephem >= 3.7.6.0; python_version >= '3'", 
                    "numexpr >= 2.6.2; python_version >= '3'", 
                    "matplotlib >= 2.0.0; python_version >= '3'", 
                    "scipy >= 1.3.3; python_version >= '3'", 
                    "astLib >= 0.8.0; python_version >= '3'", 
                    "psutil >= 5.2.2; python_version >= '3'", 
                    "py-cpuinfo >= 3.2.0; python_version >= '3'", 
                    "tables >= 3.6.0; python_version >= '3'", 
                    "prettytable >= 0.7.2; python_version >= '3'", 
                    "pybind11 >= 2.2.2; python_version >= '3'", 
                    "configparser >= 3.7.1; python_version >= '3'", 
                    "pandas >=0.23.3; python_version >= '3'", 
                    "ruamel.yaml >= 0.15.92; python_version >= '3'", 
                    "pylru >= 1.1.0; python_version >= '3'", 
                    "six >= 1.12.0; python_version >= '3'", 
                    "pybind11 >= 2.2.2; python_version >= '3'", 
                    "dask[array] >= 1.1.0; python_version >= '3'", 
                    "codex-africanus[dask] >= 0.2.10; python_version >= '3'", 
                    "regions",
                    "pywavelets",
                    "tqdm",
                    "nenupy >= 2.1.0; python_version>='3'"
                    ] 
    install_requirements = requirements

    return install_requirements

setup(name=pkg,
      version=__version__,
      description='Facet-based radio astronomy continuum imager',
      long_description = readme(),
      url='http://github.com/saopicc/DDFacet',
      classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Cyril Tasse',
      author_email='cyril.tasse@obspm.fr',
      license='GNU GPL v2',
      cmdclass={'install': custom_install,
                'sdist': custom_sdist,
                'build': custom_build,
                'build_ext': custom_build_ext
               },
      python_requires='>=3.0,<3.7',
      packages=[pkg, skymodel_pkg],
      install_requires=requirements(),
      include_package_data=True,
      zip_safe=False,
      long_description_content_type='text/markdown',
      scripts=define_scripts(),
      extras_require={
          'dft-support': ['montblanc >= 0.6.1'],
          'moresane-support': ['pymoresane >= 0.3.0'],
          'testing-requirements': ['nose >= 1.3.7'],
          'fits-beam-support': ['meqtrees-cattery'],
          'kms-support': ['bdsf > 1.8.15']
      }
)
