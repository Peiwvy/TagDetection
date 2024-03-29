function(auto_subdirectory directory)
    set(excludes ${ARGN})
    file(GLOB filelist ${directory}/*)
    foreach(pathname ${filelist})
        if(IS_DIRECTORY ${pathname})
            set(skip OFF)
            foreach(exclude ${excludes})
                if(${pathname} MATCHES ".*${exclude}$")
                    set(skip ON)
                    break()
                endif()
            endforeach()
            if(NOT skip)
                if(EXISTS ${pathname}/CMakeLists.txt)
                    message("add_subdirectory(${pathname})")
                    add_subdirectory(${pathname})
                endif()
            endif()
        endif()
    endforeach()
endfunction()

function(auto_subdirectory_recursive directory)
    set(excludes ${ARGN})
    file(GLOB filelist ${directory}/*)
    foreach(pathname ${filelist})
        if(IS_DIRECTORY ${pathname})
            set(skip OFF)
            foreach(exclude ${excludes})
                if(${pathname} MATCHES ".*${exclude}$")
                    set(skip ON)
                    break()
                endif()
            endforeach()
            if(NOT skip)
                if(EXISTS ${pathname}/CMakeLists.txt)
                    message("add_subdirectory(${pathname})")
                    add_subdirectory(${pathname})
                else()
                    auto_subdirectory_recursive(${pathname} ${excludes})
                endif()                
            endif()
        endif()
    endforeach()
endfunction()

function(add_gtest target)
    set(src ${ARGN})
    find_package(GTest REQUIRED)
    add_executable(${target} ${src})
    target_link_libraries(${target} GTest::gtest GTest::gtest_main) 
    gtest_discover_tests(${target})
endfunction()