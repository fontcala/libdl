# The main program
add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE libdl)
target_compile_features(xor PRIVATE cxx_std_17)
