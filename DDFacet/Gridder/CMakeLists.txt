cmake_minimum_required(VERSION 2.8)
set(BUILD_SHARED_LIBS ON) #we always want shared libs out of this for python extensions
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_NATIVE_TUNING "Enables support for optimized code generation (-march=native)" ON)
option(ENABLE_FAST_MATH "Enables -fast-math option (breaks IEEE754 callbacks)" ON)
option(ENABLE_PYTHON_2 "Enables support for Python 2.7" OFF)
option(ENABLE_PYTHON_3 "Enables support for Python 3.x" ON)

if (${ENABLE_NATIVE_TUNING})
  set(TUNING_FLAGS "-march=native -mtune=native")
  message(WARNING "Instructing gcc to use your native supported instruction set. This will break .so portability and may raise the ILLEGAL_INSTRUCTION signal on incompatible chipsets")
else (${ENABLE_NATIVE_TUNING})
  set(TUNING_FLAGS "")
endif (${ENABLE_NATIVE_TUNING})
if (${ENABLE_FAST_MATH})
  message("-ffast-math is enabled")
  set(FAST_MATH_FLAGS "-ffast-math")
else (${ENABLE_FAST_MATH})
  set(FAST_MATH_FLAGS "")
endif (${ENABLE_FAST_MATH})

#Add more options for debug and release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp -std=c++14 -Wall -fmax-errors=1 -ggdb3 -pedantic -W -Wall -Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -std=c++14 -ggdb3 -fmax-errors=1 -pedantic -W -Wall -Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ${OpenMP_CXX_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS} ${FAST_MATH_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -fopenmp -ggdb3 -std=c99 -pedantic -W -Wall -fmax-errors=1 -Wconversion ${OpenMP_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ggdb3 -fopenmp -fmax-errors=1 -Wconversion -W -Wall -std=c99 ${OpenMP_C_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS} ${FAST_MATH_FLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_RELEASE}")
message("-- Debug CXXFLAGS are ${CMAKE_CXX_FLAGS_DEBUG}")
message("-- Release CXXFLAGS are ${CMAKE_CXX_FLAGS_RELEASE}")

function(build_be PYVER)
  project(backend_${PYVER})
  #Find necessary modules
  if ("${PYVER}" EQUAL "27")
    message("-- Compiling backend against Python 2.7")
    set(PYREQ_VER 2.7)
    set(PYABI .cpython-27)
    find_package(PythonInterp ${PYREQ_VER} EXACT REQUIRED)
    find_package(PythonLibs ${PYREQ_VER} EXACT REQUIRED)
  elseif ("${PYVER}" EQUAL "3x")
    message("-- Compiling backend against Python 3.x")
    set(PYREQ_VER 3.0)
    set(PYABI .cpython-30)
    find_package(PythonInterp ${PYREQ_VER} REQUIRED)    
    find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED) #must match
  else ("${PYVER}" EQUAL "27")
    message(FATAL_ERROR "Version switch ${PYVER} not understood")
  endif("${PYVER}" EQUAL "27")
  
  find_package(NumPy REQUIRED)
  find_package(OpenMP REQUIRED)
  find_package(RT REQUIRED)
  find_package(pybind11 REQUIRED)

  if (NOT ${PYTHON_FOUND})
    message(FATAL_ERROR "Failed to find Python, ensure python-dev and/or python3-dev is installed")
  endif (NOT ${PYTHON_FOUND})
  if (NOT ${NUMPY_FOUND})
    message(FATAL_ERROR "Failed to find Numpy, ensure python-numpy is installed")
  endif (NOT ${NUMPY_FOUND})
  if (NOT ${OPENMP_FOUND})
    message(FATAL_ERROR "Failed to find OpenMP.")
  endif (NOT ${OPENMP_FOUND})
  if (NOT ${HAVE_RT})
    message(FATAL_ERROR "Failed to find librt and header.")
  endif (NOT ${HAVE_RT})
  if (NOT ${PYTHON_PYBIND11_FOUND})
    message(FATAL_ERROR "Failed to find Pybind11, ensure pybind11 is installed from pip. If it still can't find it downgrade to pip 9.0.3 and redo the installation.")
  endif (NOT ${PYTHON_PYBIND11_FOUND})

  #Update include paths with found includes above
  include_directories(${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR}/numpy ${RT_INCLUDES} ${CASACORE_INCLUDE_DIR} ${PYTHON_PYBIND11_INCLUDE_DIR})

  set(PYBIND11_CPP_STANDARD -std=c++14)

  #Ensure __init__.py is added to out-of-source build directory
  execute_process(COMMAND touch __init__.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  #compile and link _pyArrays.so
  add_library(_pyArrays${PYVER} Arrays.cc)
  set_target_properties(_pyArrays${PYVER} PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
  target_link_libraries(_pyArrays${PYVER} ${RT_LIBRARIES} ${PYTHON_LIBRARY})

  #compile and link _pyGridderSmearPols.so
  add_library(_pyGridderSmearPols${PYVER} GridderSmearPols.cc Semaphores.cc JonesServer.cc DecorrelationHelper.cc)
  set_target_properties(_pyGridderSmearPols${PYVER} PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
  target_link_libraries(_pyGridderSmearPols${PYVER} ${RT_LIBRARIES} ${PYTHON_LIBRARY})

  # set install path
  install(TARGETS _pyArrays${PYVER} LIBRARY DESTINATION DDFacet/cbuild/Gridder)
  install(TARGETS _pyGridderSmearPols${PYVER} LIBRARY DESTINATION DDFacet/cbuild/Gridder)
  
  # install the init.py
  install(FILES __init__.py DESTINATION DDFacet/cbuild/Gridder)
  add_subdirectory(old_c_gridder)
endfunction(build_be)

if (${ENABLE_PYTHON_2})
  build_be("27")
endif (${ENABLE_PYTHON_2})

if (${ENABLE_PYTHON_3})
  build_be("3x")
endif (${ENABLE_PYTHON_3})

