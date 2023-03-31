add_library(app_example example/app_example.cc)
add_library(app::example ALIAS app_example)

target_include_directories(app_example PUBLIC example)
target_link_libraries(app_example modules::example)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# yaml ###
find_package(yaml-cpp 0.5.2 REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})
link_libraries(${YAML_CPP_LIBRARIES})

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
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  )
