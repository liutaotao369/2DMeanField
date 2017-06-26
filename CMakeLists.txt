#Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(2DMeanField)

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -fPIC -O3 -march=native")
endif(MSVC_IDE)

find_package(OpenCV REQUIRED)

#Taken from InfiniTAM - https://github.com/victorprad/InfiniTAM.
macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()

OPTION(WITH_CUDA "Compile with CUDA?" ${CUDA_FOUND})
find_package(CUDA QUIET)
CANONIFY_BOOL(CUDA_FOUND)
if(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(WITH_CUDA TRUE)
  add_definitions(-DWITH_CUDA)
else(WITH_CUDA)
    add_definitions(-DCPU_ONLY)
endif(WITH_CUDA)

#Determine if we can use OpenMP and append to compiler flags accordingly.
option(USE_OpenMP "Use OpenMP" ON)
if(USE_OpenMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	add_definitions(-DWITH_OPENMP)
  endif(OPENMP_FOUND)
endif(USE_OpenMP)

option(PERMUTOHEDRAL "Use Permutohedral Lattice?" OFF)
if(PERMUTOHEDRAL)
include(${PROJECT_SOURCE_DIR}/cmake/UsePermutohedral.cmake)
add_definitions(-DWITH_PERMUTOHEDRAL)
endif(PERMUTOHEDRAL)

#Add shared library.
add_subdirectory(src/lib)

#Add demo.
add_subdirectory(src/app)
