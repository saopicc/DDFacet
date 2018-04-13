# Find the pybind11 package headers as installed with pip
# The following variables are set:
# PYTHON_PYBIND11_INCLUDE_DIR
# PYTHON_PYBIND11_FOUND
# (C) Benjamin Hugo

cmake_minimum_required(VERSION 2.6)
if(NOT PYTHON_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
    set(_interp_notfound 1)
  endif()
endif()

if (PYTHON_EXECUTABLE)
  # We want to use the first python in the path if it is a virtualenv
  find_program(PYTHONENV NAMES python2.7 PATHS ENV PATH)
  if(NOT PYTHONENV)
    message(FATAL_ERROR "Could not find `python` in PATH")
  endif()
  # write a python script that finds the pybind path
  file(WRITE ${PROJECT_BINARY_DIR}/FindPyBind11Path.py
  "try: import pybind11; print(pybind11.get_include())\nexcept: print('error')\n")
  # execute the find script
  exec_program("${PYTHONENV}" ${PROJECT_BINARY_DIR}
  ARGS "FindPyBind11Path.py"
  OUTPUT_VARIABLE PYBIND11_PATH)
elseif(_interp_notfound)
  message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)


find_path(PYTHON_PYBIND11_INCLUDE_DIR "pybind11/common.h"
PATH "${PYBIND11_PATH}")

if(PYTHON_PYBIND11_INCLUDE_DIR)
  set(PYTHON_PYBIND11_INCLUDE_DIR ${PYBIND11_PATH})
  set(PYTHON_PYBIND11_FOUND 1 CACHE INTERNAL "Pybind11 found")
endif(PYTHON_PYBIND11_INCLUDE_DIR)
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pybind11 DEFAULT_MSG PYTHON_PYBIND_INCLUDE_DIR)
