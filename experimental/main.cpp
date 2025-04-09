#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include "ExperimentalTests.h"


namespace py = pybind11;

template <class classifierType>
void addClassificationExperiment(const std::string& classifierName, py::module& m)
{
     py::class_<classifierType>(m, classifierName.c_str())
     .def(py::init<int, int, int, int, int, int, int, int, int, int, int>())
     .def("setTrainInputs", &classifierType::setTrainInputs, "set In")
     .def("setTrainLabels", &classifierType::setTrainLabels, "set Lb")
     .def("setTestInputs", &classifierType::setTestInputs, "set Int")
     .def("setTestLabels", &classifierType::setTestLabels, "set Lbt")
     .def("setLearningRate", &classifierType::setLearningRate, "set Lr")
     // .def("setApproximationRatios", &classifierType::setApproximationRatios, "set Lr")
     .def("train", &classifierType::train,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>())
     .def("test", &classifierType::test,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>());
}

template <class classifierType>
void addMLPExperiment(const std::string& classifierName, py::module& m)
{
     py::class_<classifierType>(m, classifierName.c_str())
     .def(py::init<int, int, int, int, int, int>())
     .def("setTrainInputs", &classifierType::setTrainInputs, "set In")
     .def("setTrainLabels", &classifierType::setTrainLabels, "set Lb")
     .def("setTestInputs", &classifierType::setTestInputs, "set Int")
     .def("setTestLabels", &classifierType::setTestLabels, "set Lbt")
     .def("setLearningRate", &classifierType::setLearningRate, "set Lr")
     // .def("setApproximationRatios", &classifierType::setApproximationRatios, "set Lr")
     .def("train", &classifierType::train,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>())
     .def("test", &classifierType::test,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>());
}

PYBIND11_MODULE(pybindings_experimental, m)
{
     addClassificationExperiment<CNNClassificationExperimentalTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>>("CNNClassificationFA", m);
     addClassificationExperiment<CNNClassificationExperimentalTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>>("CNNClassificationBaseline", m);

     //addClassificationExperiment<CNNClassificationExperimentalDFATest<ConvLayer<ReLUActivation>,FullyConnectedDirectAlignmentLayer<ReLUActivation>,FullyConnectedDirectAlignmentLayer<LinearActivation>>>("CNNClassificationDirectFeedbackAlignment", m);

     addMLPExperiment<MLPTest<FullyConnectedDirectAlignmentLayer<ReLUActivation>,FullyConnectedDirectAlignmentLayer<LinearActivation>>>("MLPDirectFeedbackAlignment", m);
     addMLPExperiment<MLPTestBP<FullyConnectedLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>>("MLPBackpropagation", m);



}
