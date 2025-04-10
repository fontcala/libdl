cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "dl"
        LANGUAGES CXX)

# Submodules
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
    message(FATAL_ERROR "Please update Catch2 and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/eigen/CMakeLists.txt")
    message(FATAL_ERROR "Please update eigen and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "Please update pybind11 and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/spdlog/CMakeLists.txt")
    message(FATAL_ERROR "Please update spdlog and try again.")
endif()
# When is it exactly that someone else has to do submodule init and stuff like that?
# Build an interface library target ourselves or use the CMake from the library
# Maybe all of this could go in a CMakeLists in the extern folder?
# Catch2 maybe this one could go to test directly?
add_subdirectory(extern/Catch2)
# eigen
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen/)
message(STATUS "eigenpath = ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen/")
# add_subdirectory(extern/eigen)
# pybind (directories not being found?)
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/pybind11/tools")
# message(STATUS "pytools = ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/tools/pybind11Tools")
# include(pybind11Tools)
# add_library(pybind11 INTERFACE)
# target_include_directories(pybind11 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/include/)
# target_compile_options(pybind11 INTERFACE -std=c++1z)
# message(STATUS "compiler")
# message(STATUS "${CMAKE_CXX_COMPILER_ID}")
# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     message(STATUS "applying -fsized-deallocation -std=c++1z")
#     target_compile_options(pybind11 INTERFACE -fsized-deallocation -std=c++1z)
#  else()
#     target_compile_options(pybind11 INTERFACE -std=c++1z)
# endif()
#set(PYBIND11_CPP_STANDARD -std=c++11) # Experimental C++17 support
add_subdirectory(extern/pybind11)
# spdlog
add_subdirectory(extern/spdlog)


# Also if header only, what is the point of adding one of the headers? (In the guide they talk about fictional target)
# should it instead be INTERFACE? 
# # STATIC or what? STATIC/SHARED can be controled via variable BUILD_SHARED_LIBS (ON)
add_library(libdl INTERFACE)
target_include_directories(libdl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/)

# PUBLIC/ PRIVATE/INTERFACE?
target_link_libraries(libdl INTERFACE eigen)
target_link_libraries(libdl INTERFACE spdlog)

# because header only so far:
target_compile_features(libdl INTERFACE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "applying -fsized-deallocation -std=c++1z")
    target_compile_options(libdl INTERFACE -fsized-deallocation -std=c++1z)
 else()
    target_compile_options(libdl INTERFACE -std=c++1z)
endif()

# test target
enable_testing()
add_subdirectory(test)
add_subdirectory(doc EXCLUDE_FROM_ALL)
add_subdirectory(python)
add_subdirectory(experimental)
add_subdirectory(examples)


