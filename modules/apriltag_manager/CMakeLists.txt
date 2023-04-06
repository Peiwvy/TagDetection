project(apriltag_manager)

file(GLOB TAG_FILES apriltag/tag*.c)
aux_source_directory(apriltag/common COMMON_SRC)
set(APRILTAG_SRCS apriltag/apriltag.c apriltag/apriltag_pose.c
                  apriltag/apriltag_quad_thresh.c
  )

add_library(ApriltagManager STATIC
            ${APRILTAG_SRCS}
            ${COMMON_SRC}
            ${TAG_FILES}
            apriltag_manager.cc
  )
add_library(modules::ApriltagManager ALIAS ApriltagManager)
target_include_directories(
  ApriltagManager PUBLIC apriltag ${CMAKE_CURRENT_SOURCE_DIR}
  )
