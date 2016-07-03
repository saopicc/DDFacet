# $Id: FindCasacore.cmake 14552 2009-11-26 16:12:40Z loose $
#
# - Try to find Casacore include dirs and libraries
# Usage:
#   find_package(Casacore [REQUIRED] [COMPONENTS components...])
# Valid components are:
#   casa, scimath_f, scimath, tables, measures, lattices,
#   fits, ms, coordinates, msfits, components, mirlib, images
#
# Note that most components are dependent on other (more basic) components.
# In that case, it suffices to specify the "top-level" components; dependent
# components will be searched for automatically.
#
# Here's the dependency tree:
#   scimath_f    ->  casa
#   scimath      ->  scimath_f
#   tables       ->  casa
#   measures     ->  scimath, tables
#   lattices     ->  scimath, tables
#   fits         ->  measures
#   ms           ->  measures
#   coordinates  ->  fits
#   msfits       ->  fits, ms
#   components   ->  coordinates
#   images       ->  components, mirlib, lattices
#
# Variables used by this module:
#  CASACORE_ROOT_DIR         - Casacore root directory. 
#
# Variables defined by this module:
#  CASACORE_FOUND            - System has Casacore, which means that the
#                              include dir was found, as well as all 
#                              libraries specified (not cached)
#  CASACORE_INCLUDE_DIR      - Casacore include directory (cached)
#  CASACORE_INCLUDE_DIRS     - Casacore include directories (not cached)
#                              identical to CASACORE_INCLUDE_DIR
#  CASACORE_LIBRARIES        - The Casacore libraries (not cached)
#  CASA_${COMPONENT}_LIBRARY - The absolute path of Casacore library 
#                              "component" (cached)
#  HAVE_AIPSPP               - True if system has Casacore (cached)
#                              for backward compatibility with AIPS++
#  HAVE_CASACORE             - True if system has Casacore (cached)
#                              identical to CASACORE_FOUND
#
# ATTENTION: The component names need to be in lower case, just as the
# casacore library names. However, the CMake variables use all upper case.
#
#
#  Copyright (C) 2008-2009
#  ASTRON (Netherlands Foundation for Research in Astronomy)
#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

# - casacore_resolve_dependencies(_result)
#
# Resolve the Casacore library dependencies for the given components. 
# The list of dependent libraries will be returned in the variable result.
# It is sorted from least dependent to most dependent library, so it can be
# directly fed to the linker.
#
#   Usage: casacore_resolve_dependencies(result components...)
#
macro(casacore_resolve_dependencies _result)
  set(${_result} ${ARGN})
  set(_index 0)
  # Do a breadth-first search through the dependency graph; append to the
  # result list the dependent components for each item in that list. 
  # Duplicates will be removed later.
  while(1)
    list(LENGTH ${_result} _length)
    if(NOT _index LESS _length)
      break()
    endif(NOT _index LESS _length)
    list(GET ${_result} ${_index} item)
    list(APPEND ${_result} ${Casacore_${item}_DEPENDENCIES})
    math(EXPR _index "${_index}+1")
  endwhile(1)
  # Remove all duplicates in the current result list, while retaining only the
  # last of each duplicate.
  list(REVERSE ${_result})
  list(REMOVE_DUPLICATES ${_result})
  list(REVERSE ${_result})
endmacro(casacore_resolve_dependencies _result)


# - casacore_find_library(_name)
#
# Search for the library ${_name}. 
# If library is found, add it to CASACORE_LIBRARIES; if not, add ${_name}
# to CASACORE_MISSING_COMPONENTS and set CASACORE_FOUND to false.
#
#   Usage: casacore_find_library(name)
#
macro(casacore_find_library _name)
  string(TOUPPER ${_name} _NAME)
  find_library(${_NAME}_LIBRARY ${_name}
    HINTS ${CASACORE_ROOT_DIR} PATH_SUFFIXES lib)
  mark_as_advanced(${_NAME}_LIBRARY)
  if(${_NAME}_LIBRARY)
    list(APPEND CASACORE_LIBRARIES ${${_NAME}_LIBRARY})
  else(${_NAME}_LIBRARY)
    set(CASACORE_FOUND FALSE)
    list(APPEND CASACORE_MISSING_COMPONENTS ${_name})
  endif(${_NAME}_LIBRARY)
endmacro(casacore_find_library _name)


# - casacore_find_package(_name)
#
# Search for the package ${_name}.
# If the package is found, add the contents of ${_name}_INCLUDE_DIRS to
# CASACORE_INCLUDE_DIRS and ${_name}_LIBRARIES to CASACORE_LIBRARIES.
#
# If Casacore itself is required, then, strictly speaking, the packages it
# requires must be present. However, when linking against static libraries
# they may not be needed. One can override the REQUIRED setting by switching
# CASACORE_MAKE_REQUIRED_EXTERNALS_OPTIONAL to ON. Beware that this might cause
# compile and/or link errors.
#
#   Usage: casacore_find_package(name [REQUIRED])
#
macro(casacore_find_package _name)
  if("${ARGN}" MATCHES "^REQUIRED$" AND
      Casacore_FIND_REQUIRED AND
      NOT CASACORE_MAKE_REQUIRED_EXTERNALS_OPTIONAL)
    find_package(${_name} REQUIRED)
  else()
    find_package(${_name})
  endif()
  if(${_name}_FOUND)
    list(APPEND CASACORE_INCLUDE_DIRS ${${_name}_INCLUDE_DIRS})
    list(APPEND CASACORE_LIBRARIES ${${_name}_LIBRARIES})
  endif(${_name}_FOUND)
endmacro(casacore_find_package _name)


# Define the Casacore components.
set(Casacore_components
  casa
  components
  coordinates
  fits
  images
  lattices
  measures
  mirlib
  ms
  msfits
  scimath
  scimath_f
  tables
)

# Define the Casacore components' inter-dependencies.
set(Casacore_components_DEPENDENCIES  coordinates)
set(Casacore_coordinates_DEPENDENCIES fits)
set(Casacore_fits_DEPENDENCIES        measures)
set(Casacore_images_DEPENDENCIES      components lattices mirlib)
set(Casacore_lattices_DEPENDENCIES    scimath tables)
set(Casacore_measures_DEPENDENCIES    scimath tables)
set(Casacore_ms_DEPENDENCIES          measures)
set(Casacore_msfits_DEPENDENCIES      fits ms)
set(Casacore_scimath_DEPENDENCIES     scimath_f)
set(Casacore_scimath_f_DEPENDENCIES   casa)
set(Casacore_tables_DEPENDENCIES      casa)

# Initialize variables.
set(CASACORE_FOUND FALSE)
set(CASACORE_DEFINITIONS)
set(CASACORE_LIBRARIES)
set(CASACORE_MISSING_COMPONENTS)

# Search for the header file first. Note that casacore installs the header
# files in ${prefix}/include/casacore, instead of ${prefix}/include.
if(NOT CASACORE_INCLUDE_DIR)
  find_path(CASACORE_INCLUDE_DIR casa/aips.h
    HINTS ${CASACORE_ROOT_DIR} PATH_SUFFIXES include/casacore)
  mark_as_advanced(CASACORE_INCLUDE_DIR)
endif(NOT CASACORE_INCLUDE_DIR)

if(NOT CASACORE_INCLUDE_DIR)
  set(CASACORE_ERROR_MESSAGE "Casacore: unable to find the header file casa/aips.h.\nPlease set CASACORE_ROOT_DIR to the root directory containing Casacore.")
else(NOT CASACORE_INCLUDE_DIR)
  # We've found the header file; let's continue.
  set(CASACORE_FOUND TRUE)
  set(CASACORE_INCLUDE_DIRS ${CASACORE_INCLUDE_DIR})

  # If the user specified components explicity, use that list; otherwise we'll
  # assume that the user wants to use all components.
  if(NOT Casacore_FIND_COMPONENTS)
    set(Casacore_FIND_COMPONENTS ${Casacore_components})
  endif(NOT Casacore_FIND_COMPONENTS)

  # Get a list of all dependent Casacore libraries that need to be found.
  casacore_resolve_dependencies(_find_components ${Casacore_FIND_COMPONENTS})

  # Find the library for each component, and handle external dependencies
  foreach(_comp ${_find_components})
    casacore_find_library(casa_${_comp})
    if(${_comp} STREQUAL casa)
      casacore_find_package(HDF5)
      casacore_find_library(m)
      list(APPEND CASACORE_LIBRARIES ${CMAKE_DL_LIBS})
    elseif(${_comp} STREQUAL coordinates)
      casacore_find_package(WcsLib REQUIRED)
    elseif(${_comp} STREQUAL fits)
      casacore_find_package(CfitsIO REQUIRED)
    elseif(${_comp} STREQUAL scimath_f)
      casacore_find_package(LAPACK REQUIRED)
    endif(${_comp} STREQUAL casa)
  endforeach(_comp ${_find_components})
endif(NOT CASACORE_INCLUDE_DIR)

# Set HAVE_CASACORE; and HAVE_AIPSPP (for backward compatibility with AIPS++).
if(CASACORE_FOUND)
  set(HAVE_CASACORE TRUE CACHE INTERNAL "Define if Casacore is installed")
  set(HAVE_AIPSPP TRUE CACHE INTERNAL "Define if AIPS++/Casacore is installed")
endif(CASACORE_FOUND)

# Compose diagnostic message if not all necessary components were found.
if(CASACORE_MISSING_COMPONENTS)
  set(CASACORE_ERROR_MESSAGE "Casacore: the following components could not be found:\n     ${CASACORE_MISSING_COMPONENTS}")
endif(CASACORE_MISSING_COMPONENTS)

# Print diagnostics.
if(CASACORE_FOUND)
  if(NOT Casacore_FIND_QUIETLY)
    message(STATUS "Found the following Casacore components: ")
    foreach(comp ${_find_components})
      message(STATUS "  ${comp}")
    endforeach(comp ${_find_components})
  endif(NOT Casacore_FIND_QUIETLY)
else(CASACORE_FOUND)
  if(Casacore_FIND_REQUIRED)
    message(FATAL_ERROR "${CASACORE_ERROR_MESSAGE}")
  else(Casacore_FIND_REQUIRED)
    message(STATUS "${CASACORE_ERROR_MESSAGE}")
  endif(Casacore_FIND_REQUIRED)
endif(CASACORE_FOUND)

