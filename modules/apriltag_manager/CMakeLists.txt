project(apriltag_manager)

file(GLOB TAG_FILES apriltag/tag*.c)
aux_source_directory(apriltag/common COMMON_SRC)
set(APRILTAG_SRCS apriltag/apriltag.c apriltag/apriltag_pose.c
                  apriltag/apriltag_quad_thresh.c
  )

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}) include_directories(src)
# include_directories(./src) include_directories(".")
add_library(ApriltagManager STATIC
            ${APRILTAG_SRCS}
            ${COMMON_SRC}
            ${TAG_FILES}
            apriltag_manager.cc
  )
add_library(modules::ApriltagManager ALIAS ApriltagManager)
target_include_directories(ApriltagManager PUBLIC apriltag)
target_include_directories(ApriltagManager PUBLIC include)
# include_directories(${HEADER_FILES} ${OpenCV_INCLUDE_DIRS})
# include_directories("./src")

# add_gtest(module_example_test example_test.cc)
# target_link_libraries(module_example_test modules::example)
