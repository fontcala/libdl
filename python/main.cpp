#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <libdl/hello.h>



// Wrapper around (cpp17)
int hello_py(std::string const& name)
{
  return hello(name);
}
using Eigen::MatrixXd;
class MyClass {
    MatrixXd someMat;
public:
    void setMatrix(const MatrixXd & aInput){someMat = aInput;}
    const Eigen::MatrixXd &viewMatrix() { return someMat; }
    
};
namespace py = pybind11;
PYBIND11_MODULE(pybindings, m) {
    m.def("hello", &hello_py, "py");
    m.def("subtract", [](int i, int j) { return i - j; }, "substract");
    py::class_<MyClass>(m, "MyClass")
    .def(py::init<>())
    .def("view_matrix", &MyClass::viewMatrix, py::return_value_policy::reference_internal)
    .def("set_matrix", &MyClass::setMatrix,"set")
    ;
}