cmake_minimum_required(VERSION 2.8)

#As per http://permalink.gmane.org/gmane.comp.programming.tools.cmake.user/15952
IF(DEFINED CMAKE_BUILD_TYPE)
   IF((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo") OR (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
      MESSAGE("-- Build type is defined as '${CMAKE_BUILD_TYPE}'")
   ELSE()
      SET(CMAKE_BUILD_TYPE "Release")   
      MESSAGE("-- Invalid option for BUILD TYPE, defaulted to '${CMAKE_BUILD_TYPE}' build")
   ENDIF()
ELSE()
   SET(CMAKE_BUILD_TYPE "Release")
   MESSAGE("-- Build type wasn't defined, defaulted to '${CMAKE_BUILD_TYPE}' build")
ENDIF()
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  MESSAGE(WARNING "You are compiling the backend in debugging mode... things may be slower than usual. Overriding all optimizations to 'OFF'")
ENDIF()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(Gridder)
