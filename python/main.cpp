#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "ClassificationExamples.h"
#include "SegmentationExamples.h"
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

     py::class_<CNNClassificationExampleModel3>(m, "CNNClassificationExampleModel3")
         .def(py::init<>())
         .def("setTrainInputs", &CNNClassificationExampleModel3::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationExampleModel3::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationExampleModel3::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationExampleModel3::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationExampleModel3::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationExampleModel3::runExample,
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

     py::class_<AutoEncoderExample5>(m, "AutoEncoderExample5")
         .def(py::init<int, int, int, int, int, int, int, int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &AutoEncoderExample5::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &AutoEncoderExample5::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<AutoEncoderExample6>(m, "AutoEncoderExample6")
         .def(py::init<int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &AutoEncoderExample6::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &AutoEncoderExample6::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<SegmentationExample1>(m, "SegmentationExample1")
         .def(py::init<int, int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &SegmentationExample1::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &SegmentationExample1::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<SegmentationExample2>(m, "SegmentationExample2")
         .def(py::init<int, int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &SegmentationExample2::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &SegmentationExample2::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<SegmentationExample3>(m, "SegmentationExample3")
         .def(py::init<int, int, int, int, int, int, int, int, int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &SegmentationExample3::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &SegmentationExample3::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<SegmentationExample6>(m, "SegmentationExample6")
         .def(py::init<int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &SegmentationExample6::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &SegmentationExample6::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<SegmentationExample7>(m, "SegmentationExample7")
         .def(py::init<int, int, int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &SegmentationExample7::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &SegmentationExample7::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<SegmentationExample8>(m, "SegmentationExample8")
         .def(py::init<int, int, int, int>(),
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Test", &SegmentationExample8::Test,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>())
         .def("Train", &SegmentationExample8::Train,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());
}
