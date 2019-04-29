#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <libdl/hello.h>



// Wrapper around (cpp17)
int hello_py(std::string const& name)
{
  return hello(name);
}
namespace py = pybind11;
PYBIND11_MODULE(pybindings, m) {
    m.def("hello", &hello_py, "py");
    m.def("subtract", [](int i, int j) { return i - j; }, "substract");
}