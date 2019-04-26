cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "dl"
        LANGUAGES CXX)


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
target_compile_features(libdl PUBLIC cxx_std_17)

# test target
add_subdirectory(test)


