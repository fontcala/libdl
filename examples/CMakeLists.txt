# The main program
add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE libdl)
target_compile_features(xor PRIVATE cxx_std_17)

add_executable(xor_exp xor_exp.cpp)
target_link_libraries(xor_exp PRIVATE libdl)
target_compile_features(xor_exp PRIVATE cxx_std_17)

add_executable(xor_exp_dfa xor_exp_dfa.cpp)
target_link_libraries(xor_exp_dfa PRIVATE libdl)
target_compile_features(xor_exp_dfa PRIVATE cxx_std_17)

add_executable(images images.cpp)
target_link_libraries(images PRIVATE libdl)
target_compile_features(images PRIVATE cxx_std_17)

add_executable(reshapes reshapes.cpp)
target_link_libraries(reshapes PRIVATE libdl)
target_compile_features(reshapes PRIVATE cxx_std_17)