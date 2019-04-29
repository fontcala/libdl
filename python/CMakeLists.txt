project(pybindings)
pybind11_add_module(pybindings main.cpp)
target_link_libraries(pybindings PRIVATE libdl)

enable_testing()
add_test(NAME pybindings_libdl COMMAND python3 bindings.py)
set_tests_properties (pybindings_libdl PROPERTIES ENVIRONMENT "PYTHONPATH=${pybindings_BINARY_DIR}")