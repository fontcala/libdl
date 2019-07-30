# The main program
add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE libdl)
target_compile_features(xor PRIVATE cxx_std_17)

add_executable(images images.cpp)
target_link_libraries(images PRIVATE libdl)
target_compile_features(images PRIVATE cxx_std_17)

add_executable(reshapes reshapes.cpp)
target_link_libraries(reshapes PRIVATE libdl)
target_compile_features(reshapes PRIVATE cxx_std_17)

add_executable(skip skip.cpp)
target_link_libraries(skip PRIVATE libdl)
target_compile_features(skip PRIVATE cxx_std_17)