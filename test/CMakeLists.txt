# The main program
add_executable(firstTest firstTest.cpp)
target_link_libraries(firstTest PRIVATE libdl)
target_link_libraries(firstTest PRIVATE Catch2)
