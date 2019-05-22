# The main program
add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE libdl)
target_compile_features(xor PRIVATE cxx_std_17)

add_executable(xor_bigger_network xor_bigger_network.cpp)
target_link_libraries(xor_bigger_network PRIVATE libdl)
target_compile_features(xor_bigger_network PRIVATE cxx_std_17)
