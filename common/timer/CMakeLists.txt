cmake_minimum_required(VERSION 3.12)
project(timers)

set(CMAKE_CXX_STANDARD 17)

find_package(Glog REQUIRED)

add_library(timer STATIC timer.cc)
add_library(common::timer ALIAS timer)
target_include_directories(
  timer INTERFACE ${GLOG_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
  )
target_link_libraries(timer PUBLIC ${GLOG_LIBRARIES})

option(BUILD_EXAMPLE "Build example" OFF)
if(BUILD_EXAMPLE)
  add_executable(timer_record timerrecord_test.cpp)
  target_link_libraries(timer_record timer)
endif()
