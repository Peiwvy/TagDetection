project(modules_example)

add_library(modules_example example.cc)
add_library(modules::example ALIAS modules_example)
target_include_directories(modules_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_gtest(module_example_test example_test.cc)
target_link_libraries(module_example_test modules::example)
