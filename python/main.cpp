#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <libdl/hello.h>
#include <libdl/dlfunctions.h>
#include <libdl/ConvLayer.h>
#include <libdl/SigmoidActivationLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/FullyConnectedLayer.h>

int hello_py(std::string const &name)
{
    return hello(name);
}
using Eigen::MatrixXd;
class MyClass
{
    MatrixXd someMat;

public:
    void setMatrix(const MatrixXd &aInput) { someMat = aInput; }
    const Eigen::MatrixXd &viewMatrix() { return someMat; }
};

class ExampleModel
{
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;
    size_t mInputDepth;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mNumCategories;
    double mLearningRate;

public:
    void setInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        mInputDepth = aInputDepth;
        mInputHeight = aInputHeight;
        mInputWidth = aInputWidth;
    }
    void setLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        mNumCategories = aNumCategories;
    }
    void setLearningRate(const double aLearningRate)
    {
        mLearningRate = aLearningRate;
    }
    void train();
};
void ExampleModel::train()
{
    const size_t vInputSampleNumber = 1;

    // CONV 1
    const size_t vFilterHeight1 = 2;
    const size_t vFilterWidth1 = 3;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 2;
    const size_t vOutputDepth1 = 6;

    ConvLayer firstConvLayer(vFilterHeight1,
                             vFilterWidth1,
                             vPaddingHeight1,
                             vPaddingWidth1,
                             vStride1,
                             mInputDepth,
                             mInputHeight,
                             mInputWidth,
                             vOutputDepth1,
                             vInputSampleNumber);

    // Sigmoid.
    SigmoidActivationLayer firstSigmoidLayer;

    // Conv 2
    const size_t vFilterHeight2 = 3;
    const size_t vFilterWidth2 = 2;
    const size_t vPaddingHeight2 = 1;
    const size_t vPaddingWidth2 = 1;
    const size_t vStride2 = 2;
    const size_t vOutputDepth2 = 7;
    ConvLayer secondConvLayer(vFilterHeight2,
                              vFilterWidth2,
                              vPaddingHeight2,
                              vPaddingWidth2,
                              vStride2,
                              firstConvLayer.GetOutputDims(),
                              vOutputDepth2,
                              vInputSampleNumber);

    // Sigmoid
    SigmoidActivationLayer secondSigmoidLayer;

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    FullyConnectedLayer fcLayer(flattenLayer.GetOutputDims(), mNumCategories);

    // losslayer
    SoftmaxLossLayer lossLayer;

    // Connect
    firstSigmoidLayer.SetInput(firstConvLayer.GetOutput());
    secondConvLayer.SetInput(firstSigmoidLayer.GetOutput());
    secondSigmoidLayer.SetInput(secondConvLayer.GetOutput());
    flattenLayer.SetInput(secondSigmoidLayer.GetOutput());
    fcLayer.SetInput(flattenLayer.GetOutput());
    lossLayer.SetInput(fcLayer.GetOutput());

    firstConvLayer.SetBackpropInput(firstSigmoidLayer.GetBackpropOutput());
    firstSigmoidLayer.SetBackpropInput(secondConvLayer.GetBackpropOutput());
    secondConvLayer.SetBackpropInput(secondSigmoidLayer.GetBackpropOutput());
    secondSigmoidLayer.SetBackpropInput(flattenLayer.GetBackpropOutput());
    flattenLayer.SetBackpropInput(fcLayer.GetBackpropOutput());
    fcLayer.SetBackpropInput(lossLayer.GetBackpropOutput());

    for (size_t i = 0; i < 5; i++)
    {
        MatrixXd Input = mTrainInput.block(0, 10, mInputHeight * mInputWidth, 1);
        MatrixXd Label = mTrainLabels.block(10, 0, 1, mNumCategories);
        // std::cout << "Input" << std::endl;
        // std::cout << Input.rows() << " " << Input.cols() << std::endl;
        // std::cout << "Label" << std::endl;
        // std::cout << Label << std::endl;
        std::cout << "---------start forward ---------" << std::endl;
        firstConvLayer.SetInput(Input);
        lossLayer.SetLabels(Label);
        std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
        firstConvLayer.ForwardPass();
        std::cout << "firstSigmoidLayer.ForwardPass()  ------" << std::endl;
        firstSigmoidLayer.ForwardPass();
        std::cout << "secondConvLayer.ForwardPass()  ------" << std::endl;
        secondConvLayer.ForwardPass();
        std::cout << "secondSigmoidLayer.ForwardPass()  ------" << std::endl;
        secondSigmoidLayer.ForwardPass();
        std::cout << "flattenLayer.ForwardPass()  ------" << std::endl;
        flattenLayer.ForwardPass();
        std::cout << "fcLayer.ForwardPass()  ------" << std::endl;
        fcLayer.ForwardPass();
        std::cout << "lossLayer.ForwardPass()  ------" << std::endl;
        lossLayer.ForwardPass();
        std::cout << "lossLayer.GetLoss() ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << lossLayer.GetLoss() << std::endl;

        std::cout << "---------start backward ---------" << std::endl;
        std::cout << "lossLayer.BackwardPass()  ------" << std::endl;
        lossLayer.BackwardPass();
        std::cout << "fcLayer.BackwardPass()  ------" << std::endl;
        fcLayer.BackwardPass();
        std::cout << "flattenLayer.BackwardPass()  ------" << std::endl;
        flattenLayer.BackwardPass();
        std::cout << "secondSigmoidLayer.BackwardPass()  ------" << std::endl;
        secondSigmoidLayer.BackwardPass();
        std::cout << "secondConv.BackwardPass()  ------" << std::endl;
        secondConvLayer.BackwardPass();
        std::cout << "firstSigmoidLayer.BackwardPass()  ------" << std::endl;
        firstSigmoidLayer.BackwardPass();
        std::cout << "firstConvLayer.BackwardPass()  ------" << std::endl;
        firstConvLayer.BackwardPass();
    }
}

namespace py = pybind11;
PYBIND11_MODULE(pybindings, m)
{
    m.def("hello", &hello_py, "py");
    m.def("subtract", [](int i, int j) { return i - j; }, "substract");
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("view_matrix", &MyClass::viewMatrix, py::return_value_policy::reference_internal)
        .def("set_matrix", &MyClass::setMatrix, "set");
    py::class_<ExampleModel>(m, "ExampleModel")
        .def(py::init<>())
        .def("setInputs", &ExampleModel::setInputs, "set In")
        .def("setLabels", &ExampleModel::setLabels, "set Lb")
        .def("setLearningRate", &ExampleModel::setLearningRate, "set Lr")
        .def("trainz", &ExampleModel::train,
             py::call_guard<py::scoped_ostream_redirect,
                            py::scoped_estream_redirect>());
}

// .def("train", []() {
//             py::scoped_ostream_redirect stream(
//                 std::cout,                               // std::ostream&
//                 py::module::import("sys").attr("stdout") // Python output
//             );
//             &ExampleModel::train;

// m.def("noisy_func", &call_noisy_function,
//       py::call_guard<py::scoped_ostream_redirect,
//                      py::scoped_estream_redirect>());
