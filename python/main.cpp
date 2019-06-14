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
#include <algorithm> // std::random_shuffle
#include <vector>    // std::vector

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

// Super stupid class to quickly debug achieve the milestone, a better wrapper will come.
class ExampleModel
{
    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;
    size_t mInputDepth;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mNumCategories;
    double mLearningRate = 0.01;

public:
    void setTrainInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        mInputDepth = aInputDepth;
        mInputHeight = aInputHeight;
        mInputWidth = aInputWidth;
    }
    void setTrainLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        mNumCategories = aNumCategories;
    }
    void setTestInputs(const MatrixXd &aInput)
    {
        mTestInput = aInput;
    }
    void setTestLabels(const MatrixXd &aInput)
    {
        mTestLabels = aInput;
    }
    void setLearningRate(const double aLearningRate)
    {
        mLearningRate = aLearningRate;
    }
    void runExample(const size_t aBatchNum);
};
void ExampleModel::runExample(const size_t aBatchNum)
{
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 5;
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
    const size_t vFilterWidth2 = 3;
    const size_t vPaddingHeight2 = 1;
    const size_t vPaddingWidth2 = 1;
    const size_t vStride2 = 2;
    const size_t vOutputDepth2 = 8;
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

    // Init Params
    firstConvLayer.mLearningRate = mLearningRate;
    secondConvLayer.mLearningRate = mLearningRate;
    fcLayer.mLearningRate = mLearningRate;

    // TRAIN
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    for (size_t vBatch = 0; vBatch < aBatchNum; vBatch++)
    {
        std::random_shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end());
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            // std::cout << "Input" << std::endl;
            // std::cout << Input.rows() << " " << Input.cols() << std::endl;
            // std::cout << "Label" << std::endl;
            // std::cout << Label << std::endl;
            //std::cout << "---------start forward ---------" << std::endl;
            firstConvLayer.SetInput(Input);
            lossLayer.SetLabels(Label);
            //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
            firstConvLayer.ForwardPass();
            //std::cout << "firstSigmoidLayer.ForwardPass()  ------" << std::endl;
            firstSigmoidLayer.ForwardPass();
            //std::cout << "secondConvLayer.ForwardPass()  ------" << std::endl;
            secondConvLayer.ForwardPass();
            //std::cout << "secondSigmoidLayer.ForwardPass()  ------" << std::endl;
            secondSigmoidLayer.ForwardPass();
            //std::cout << "flattenLayer.ForwardPass()  ------" << std::endl;
            flattenLayer.ForwardPass();
            //std::cout << "fcLayer.ForwardPass()  ------" << std::endl;
            fcLayer.ForwardPass();
            //std::cout << "lossLayer.ForwardPass()  ------" << std::endl;
            lossLayer.ForwardPass();
            // std::cout << "lossLayer.GetLoss() ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            // std::cout << lossLayer.GetLoss() << std::endl;
            //std::cout << "---------start backward ---------" << std::endl;
            //std::cout << "lossLayer.BackwardPass()  ------" << std::endl;
            lossLayer.BackwardPass();
            //std::cout << "fcLayer.BackwardPass()  ------" << std::endl;
            fcLayer.BackwardPass();
            //std::cout << "flattenLayer.BackwardPass()  ------" << std::endl;
            flattenLayer.BackwardPass();
            //std::cout << "secondSigmoidLayer.BackwardPass()  ------" << std::endl;
            secondSigmoidLayer.BackwardPass();
            //std::cout << "secondConv.BackwardPass()  ------" << std::endl;
            secondConvLayer.BackwardPass();
            //std::cout << "firstSigmoidLayer.BackwardPass()  ------" << std::endl;
            firstSigmoidLayer.BackwardPass();
            //std::cout << "firstConvLayer.BackwardPass()  ------" << std::endl;
            firstConvLayer.BackwardPass();
        }
        if (vBatch % 1 == 0)
        {
            std::cout << "lossLayer.GetLoss() of any givenn sample ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            std::cout << lossLayer.GetLoss() << std::endl;
            std::cout << "Batch " << vBatch << std::endl;
        }
    }

    //TEST
    const size_t vTotalTestSamples = mTestInput.cols();
    size_t vNumCorrectlyClassified = 0;
    std::vector<size_t> vIndexTestVector(vTotalTestSamples);
    std::iota(std::begin(vIndexTestVector), std::end(vIndexTestVector), 0); // Fill with 0, 1, ..., N.
    for (const auto &vIndex : vIndexTestVector)
        {
            MatrixXd Input = mTestInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTestLabels.block(vIndex, 0, 1, mNumCategories);
            // std::cout << "Input" << std::endl;
            // std::cout << Input.rows() << " " << Input.cols() << std::endl;
            // std::cout << "Label" << std::endl;
            // std::cout << Label << std::endl;
            //std::cout << "---------start forward ---------" << std::endl;
            firstConvLayer.SetInput(Input);
            lossLayer.SetLabels(Label);
            //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
            firstConvLayer.ForwardPass();
            //std::cout << "firstSigmoidLayer.ForwardPass()  ------" << std::endl;
            firstSigmoidLayer.ForwardPass();
            //std::cout << "secondConvLayer.ForwardPass()  ------" << std::endl;
            secondConvLayer.ForwardPass();
            //std::cout << "secondSigmoidLayer.ForwardPass()  ------" << std::endl;
            secondSigmoidLayer.ForwardPass();
            //std::cout << "flattenLayer.ForwardPass()  ------" << std::endl;
            flattenLayer.ForwardPass();
            //std::cout << "fcLayer.ForwardPass()  ------" << std::endl;
            fcLayer.ForwardPass();
            //std::cout << "lossLayer.ForwardPass()  ------" << std::endl;
            lossLayer.ForwardPass();

            // Compare with labels
            MatrixXd vScores = *(lossLayer.GetOutput());
            MatrixXd::Index maxColScores,maxRowScores;
            const double maxScores = vScores.maxCoeff(&maxRowScores, &maxColScores);
            MatrixXd::Index maxColLabel,maxRowLabel;
            const double maxLabels = Label.maxCoeff(&maxRowLabel, &maxColLabel);
            if(maxColScores == maxColLabel)
            {
                ++vNumCorrectlyClassified;
            }
            // std::cout << "Cat scores" << std::endl;
            // std::cout << maxColScores << std::endl;
            // std::cout << "Cat label" << std::endl;
            // std::cout << maxColLabel << std::endl;
        }
        const double vAccuracy = static_cast<double>(vNumCorrectlyClassified)/static_cast<double>(vTotalTestSamples);
        std::cout << "test Accuuracy is " << vAccuracy << std::endl;
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
        .def("setTrainInputs", &ExampleModel::setTrainInputs, "set In")
        .def("setTrainLabels", &ExampleModel::setTrainLabels, "set Lb")
        .def("setTestInputs", &ExampleModel::setTestInputs, "set Int")
        .def("setTestLabels", &ExampleModel::setTestLabels, "set Lbt")
        .def("setLearningRate", &ExampleModel::setLearningRate, "set Lr")
        .def("runExample", &ExampleModel::runExample,
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
