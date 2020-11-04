#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "ClassificationExamples.h"
#include "SegmentationExamples.h"
#include "AutoEncoderExamples.h"
#include "AlignmentTests.h"

namespace py = pybind11;
PYBIND11_MODULE(pybindings, m)
{
     m.def("subtract", [](int i, int j) { return i - j; }, "substract");

     py::class_<CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>>(m, "CNNClassificationAlignmentTestMom")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAlignmentTest<ConvExperimentalLayer<ReLUActivation>,FullyConnectedExperimentalLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>>(m, "CNNClassificationAlignmentTestBP")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAlignmentTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());
     

     py::class_<CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>>(m, "CNNClassificationAlignmentTestFA")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAlignmentTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());
     
     py::class_<CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>>(m, "CNNClassificationAlignmentTestSFA")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAlignmentTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());


     py::class_<CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>>(m, "CNNClassificationAltTestBP")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAltTest<ConvLayer<ReLUActivation>,FullyConnectedLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>>(m, "CNNClassificationAltTestFA")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAltTest<ConvAlignmentLayer<ReLUActivation>,FullyConnectedAlignmentLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());

     py::class_<CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>>(m, "CNNClassificationAltTestSFA")
         .def(py::init<int, int, int, int, int, int>())
         .def("setTrainInputs", &CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationAltTest<ConvSignedAlignmentLayer<ReLUActivation>,FullyConnectedSignedAlignmentLayer<LinearActivation>>::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());     

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
     
     py::class_<CNNClassificationExampleModelMirror>(m, "CNNClassificationExampleModelMirror")
         .def(py::init<>())
         .def("setTrainInputs", &CNNClassificationExampleModelMirror::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationExampleModelMirror::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationExampleModelMirror::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationExampleModelMirror::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationExampleModelMirror::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationExampleModelMirror::runExample,
              py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>());
     
     py::class_<CNNClassificationExampleModelSignedMirror>(m, "CNNClassificationExampleModelSignedMirror")
         .def(py::init<>())
         .def("setTrainInputs", &CNNClassificationExampleModelSignedMirror::setTrainInputs, "set In")
         .def("setTrainLabels", &CNNClassificationExampleModelSignedMirror::setTrainLabels, "set Lb")
         .def("setTestInputs", &CNNClassificationExampleModelSignedMirror::setTestInputs, "set Int")
         .def("setTestLabels", &CNNClassificationExampleModelSignedMirror::setTestLabels, "set Lbt")
         .def("setLearningRate", &CNNClassificationExampleModelSignedMirror::setLearningRate, "set Lr")
         .def("runExample", &CNNClassificationExampleModelSignedMirror::runExample,
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
