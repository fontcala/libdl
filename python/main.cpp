#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "ClassificationExamples.h"
#include "AutoEncoderExamples.h"

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
     
    py::class_<CNNClassificationExampleModel2>(m, "CNNClassificationExampleModel2")
        .def(py::init<>())
        .def("setTrainInputs", &CNNClassificationExampleModel2::setTrainInputs, "set In")
        .def("setTrainLabels", &CNNClassificationExampleModel2::setTrainLabels, "set Lb")
        .def("setTestInputs", &CNNClassificationExampleModel2::setTestInputs, "set Int")
        .def("setTestLabels", &CNNClassificationExampleModel2::setTestLabels, "set Lbt")
        .def("setLearningRate", &CNNClassificationExampleModel2::setLearningRate, "set Lr")
        .def("runExample", &CNNClassificationExampleModel2::runExample,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());

    py::class_<AutoEncoderExample>(m, "AutoEncoderExample")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int>(),
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Test", &AutoEncoderExample::Test, 
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Train", &AutoEncoderExample::Train,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());

    py::class_<AutoEncoderExample2>(m, "AutoEncoderExample2")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int>(),
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Test", &AutoEncoderExample2::Test, 
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Train", &AutoEncoderExample2::Train,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());

     py::class_<AutoEncoderExample3>(m, "AutoEncoderExample3")
        .def(py::init<int, int, int, int>(),
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Test", &AutoEncoderExample3::Test, 
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Train", &AutoEncoderExample3::Train,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());
                            
     py::class_<AutoEncoderExample4>(m, "AutoEncoderExample4")
        .def(py::init<int, int, int, int>(),
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Test", &AutoEncoderExample4::Test, 
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>())
        .def("Train", &AutoEncoderExample4::Train,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());
}
