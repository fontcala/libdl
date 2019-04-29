project(pybindings)
pybind11_add_module(pybindings main.cpp)
target_link_libraries(pybindings PRIVATE libdl)