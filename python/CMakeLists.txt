project(pybindings)
pybind11_add_module(pybindings main.cpp)
target_link_libraries(pybindings PRIVATE libdl)
target_compile_features(pybindings PRIVATE cxx_std_17)