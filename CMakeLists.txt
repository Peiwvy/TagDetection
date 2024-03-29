cmake_minimum_required(VERSION 3.5)
project(cpptemplates)

# ##############################################################################
# Version
# ##############################################################################
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)

# ##############################################################################
# C++ Standard
# ##############################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ##############################################################################
# Compiler Defines
# ##############################################################################
add_compile_options(-DWORK_SPACE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

# ##############################################################################
# Compiler Options
# ##############################################################################
add_compile_options(-Wall)
add_compile_options(-Werror=return-type)
# default static library for better performance
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dynamically-linked binaries")
# configure link time optimization
option(WITH_LTO "enable link time optimization" OFF)
if(WITH_LTO)
  add_compile_options(-flto)
endif()

# ##############################################################################
# Dependencies
# ##############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(cmake/cmake_tools.cmake)
enable_testing()

set(CUSTOM_TBB_DIR "/opt/tbb")

if(CUSTOM_TBB_DIR)
  set(TBB_INCLUDE_DIR "${CUSTOM_TBB_DIR}/include")
  set(TBB_LIBRARY_DIR "${CUSTOM_TBB_DIR}/lib/intel64/gcc4.8")
  include_directories(${TBB_INCLUDE_DIR})
  link_directories(${TBB_LIBRARY_DIR})
endif()

message(STATUS "Hello, CMake!  ${TBB_INCLUDE_DIR}  ")
message(STATUS "Hello, CMake!  ${TBB_LIBRARY_DIR}  ")

# ##############################################################################
# Source files
# ##############################################################################
auto_subdirectory_recursive(.)
