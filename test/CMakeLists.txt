# The main program
add_executable(firstTest firstTest.cpp)
target_link_libraries(firstTest PRIVATE libdl)
target_link_libraries(firstTest PRIVATE Catch2)
target_compile_features(firstTest PRIVATE cxx_std_17)
add_test(NAME firstTest COMMAND firstTest)

add_executable(im2col im2col.cpp)
target_link_libraries(im2col PRIVATE libdl)
target_link_libraries(im2col PRIVATE Catch2)
target_compile_features(im2col PRIVATE cxx_std_17)
add_test(NAME im2col COMMAND im2col)

add_executable(conv conv.cpp)
target_link_libraries(conv PRIVATE libdl)
target_link_libraries(conv PRIVATE Catch2)
target_compile_features(conv PRIVATE cxx_std_17)
add_test(NAME conv COMMAND conv)
