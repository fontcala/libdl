cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "dl"
        LANGUAGES CXX)

# STATIC or what?
# Also if header only, what is the point of adding one of the headers? (In the guide they talk about fictional target)
# should it instead be INTERFACE? 
add_library(libdl STATIC include/libdl/hello.h)
target_include_directories(libdl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
# because header only so far:
set_target_properties(libdl PROPERTIES LINKER_LANGUAGE CXX)

# test target
add_subdirectory(test)


