cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "dl"
        LANGUAGES CXX)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Catch2/CMakeLists.txt")
    message(FATAL_ERROR "GIT_SUBMODULE was turned off or failed. Please update Catch2 and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/eigen/CMakeLists.txt")
    message(FATAL_ERROR "GIT_SUBMODULE was turned off or failed. Please update eigen and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "GIT_SUBMODULE was turned off or failed. Please update pybind11 and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt")
    message(FATAL_ERROR "GIT_SUBMODULE was turned off or failed. Please update spdlog and try again.")
endif()
# When is it exactly that someone else has to do submodule init and stuff like that?
# Build an interface library target ourselves or use the CMake from the library
# Maybe all of this could go in a CMakeLists in the extern folder?
# Catch2 maybe this one could go to test directly?
add_subdirectory(extern/Catch2)
# eigen
add_subdirectory(extern/eigen)
# pybind
add_subdirectory(extern/pybind11)
# spdlog
add_subdirectory(extern/spdlog)


# STATIC or what?
# Also if header only, what is the point of adding one of the headers? (In the guide they talk about fictional target)
# should it instead be INTERFACE? 
add_library(libdl STATIC include/libdl/hello.h)
target_include_directories(libdl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)

target_link_libraries(libdl PUBLIC eigen)
target_link_libraries(libdl PUBLIC pybind11)
target_link_libraries(libdl PUBLIC spdlog)

# because header only so far:
set_target_properties(libdl PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(libdl PUBLIC cxx_std_11)

# test target
add_subdirectory(test)


