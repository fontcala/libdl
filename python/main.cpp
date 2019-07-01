#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "pythonexamples.h"

namespace py = pybind11;
PYBIND11_MODULE(pybindings, m)
{
    m.def("subtract", [](int i, int j) { return i - j; }, "substract");
    py::class_<CNNClassificationExampleModel>(m, "CNNClassificationExampleModel")
        .def(py::init<>())
        .def("setTrainInputs", &CNNClassificationExampleModel::setTrainInputs, "set In")
        .def("setTrainLabels", &CNNClassificationExampleModel::setTrainLabels, "set Lb")
        .def("setTestInputs", &CNNClassificationExampleModel::setTestInputs, "set Int")
        .def("setTestLabels", &CNNClassificationExampleModel::setTestLabels, "set Lbt")
        .def("setLearningRate", &CNNClassificationExampleModel::setLearningRate, "set Lr")
        .def("runExample", &CNNClassificationExampleModel::runExample,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());

    py::class_<SegmentationExample>(m, "SegmentationExample")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int>(),
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Test", &SegmentationExample::Test, "test")
        .def("Train", &SegmentationExample::Train,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());
}