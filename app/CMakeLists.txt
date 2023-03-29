add_library(app_example example/app_example.cc)
add_library(app::example ALIAS app_example)

target_include_directories(app_example PUBLIC example)
target_link_libraries(app_example modules::example)
