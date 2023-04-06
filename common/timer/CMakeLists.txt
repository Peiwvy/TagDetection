cmake_minimum_required(VERSION 3.12)

project(timers)
set(CMAKE_CXX_STANDARD 17)

find_package(Glog REQUIRED)

add_library(timer STATIC timer.cc)
add_library(common::timer ALIAS timer)
target_include_directories(timer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(timer glog)

option(TIMER_BUILD_EXAMPLE "timer example" ON)
if(TIMER_BUILD_EXAMPLE)
  add_executable(timer_record timerrecord_test.cpp)
  target_link_libraries(timer_record timer)
endif()
