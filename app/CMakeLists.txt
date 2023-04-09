find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(yaml-cpp 0.5.2 REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})
link_libraries(${YAML_CPP_LIBRARIES})

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(tag_detection tagdetection/tag_detection.cc)
add_library(app::tag_detection ALIAS tag_detection)

target_include_directories(
  tag_detection
  PUBLIC tagdetection
         ${PCL_INCLUDE_DIRS}
         ${YAML_CPP_INCLUDE_DIR}
         ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(
  tag_detection
  modules::ApriltagManager
  common::timer
  reflcpp::reflcpp
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  tbb
  )
