# The main program
add_executable(firstTest firstTest.cpp)
target_link_libraries(firstTest PRIVATE libdl)
target_link_libraries(firstTest PRIVATE Catch2)
target_compile_features(firstTest PRIVATE cxx_std_17)
add_test(NAME firstTest COMMAND firstTest)

add_executable(reshapings reshapings.cpp)
target_link_libraries(reshapings PRIVATE libdl)
target_link_libraries(reshapings PRIVATE Catch2)
target_compile_features(reshapings PRIVATE cxx_std_17)
add_test(NAME reshapings COMMAND reshapings)

add_executable(conv conv.cpp)
target_link_libraries(conv PRIVATE libdl)
target_link_libraries(conv PRIVATE Catch2)
target_compile_features(conv PRIVATE cxx_std_17)
add_test(NAME conv COMMAND conv)

