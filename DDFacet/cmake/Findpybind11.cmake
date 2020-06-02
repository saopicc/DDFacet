# Find the pybind11 package headers as installed with pip <= 9.0.3
# The following variables are set:
# PYTHON_PYBIND11_INCLUDE_DIR
# PYTHON_PYBIND11_FOUND
# User may force an include path by specifying PYBIND11_INCLUDE_PATH_HINT in commandline options
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
  find_program(PYTHONENV NAMES "${PYTHON_EXECUTABLE}" PATHS ENV PATH)
  if(NOT PYTHONENV)
    message(FATAL_ERROR "Could not find `python` in PATH")
  endif()
  # write a python script that finds the pybind path
  file(WRITE ${PROJECT_BINARY_DIR}/FindPyBind11Path.py
  "try: import pybind11; print(pybind11.get_include())\nexcept: pass\n")
  # execute the find script
  exec_program("${PYTHONENV}" ${PROJECT_BINARY_DIR}
  ARGS "FindPyBind11Path.py"
  OUTPUT_VARIABLE PYBIND11_PATH)
  
  # write a python script that finds the pybind path
  file(WRITE ${PROJECT_BINARY_DIR}/FindPyBind11PathUser.py
  "try: import pybind11; print(pybind11.get_include(user=True))\nexcept: pass\n")
  # execute the find script
  exec_program("${PYTHONENV}" ${PROJECT_BINARY_DIR}
  ARGS "FindPyBind11PathUser.py"
  OUTPUT_VARIABLE PYBIND11_PATH_USER)
elseif(_interp_notfound)
  message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

# accept user hint
if(PYBIND11_INCLUDE_PATH_HINT)
  find_path(PYTHON_PYBIND11_INCLUDE_DIR "pybind11/common.h"
  PATH "${PYBIND11_INCLUDE_PATH_HINT}")
  if (NOT PYTHON_PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "Tried user override for pybind11 header include path. This failed. Please check your specified path")
  else()
    set(PYBIND11_PATH ${PYBIND11_INCLUDE_PATH_HINT})
    set(PYBIND11_PATH_USER ${PYBIND11_INCLUDE_PATH_HINT})
  endif ()
else(PYBIND11_INCLUDE_PATH_HINT)
  # otherwise auto
  find_path(PYTHON_PYBIND11_INCLUDE_DIR "pybind11/common.h"
  PATH "${PYBIND11_PATH}"
  PATH "${PYBIND11_PATH_USER}")
endif(PYBIND11_INCLUDE_PATH_HINT)

#if found set return variables
if(PYTHON_PYBIND11_INCLUDE_DIR)
  set(PYTHON_PYBIND11_INCLUDE_DIR "${PYBIND11_PATH};${PYBIND11_PATH_USER}")
  set(PYTHON_PYBIND11_FOUND 1 CACHE INTERNAL "Pybind11 found")
endif()
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pybind11 DEFAULT_MSG PYTHON_PYBIND11_INCLUDE_DIR)
