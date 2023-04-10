cmake_minimum_required(VERSION 3.12)
project(logger)

find_package(Glog REQUIRED)

add_library(logger STATIC logger.cc)
add_library(common::logger ALIAS logger)
target_include_directories(logger INTERFACE ${GLOG_INCLUDE_DIRS})
target_link_libraries(logger PUBLIC ${GLOG_LIBRARIES})

option(BUILD_EXAMPLE "Build example" OFF)
if(BUILD_EXAMPLE)
  add_executable(logger_example logger_example.cpp)
  target_link_libraries(logger_example logger)
endif()
