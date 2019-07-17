#include <iostream>
#include <libdl/NetworkHelper.h>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/L2LossLayer.h>
#include <libdl/FullyConnectedLayer.h>

using Eigen::MatrixXd;
int main()
{
    const size_t vInputSampleNumber = 1;
    const size_t vNumCategories = 2;
    // Input
    const size_t vInputDepth1 = 3;
    const size_t vInputHeight1 = 10;
    const size_t vInputWidth1 = 9;
    MatrixXd Input = MatrixXd::Random(vInputHeight1 * vInputWidth1, vInputDepth1);
    MatrixXd Label(vInputSampleNumber, vNumCategories);
    Label << 0, 1;
    //std::cout << "Input" << std::endl;
    //std::cout << Input << std::endl;

    // CONV 1
    const size_t vFilterHeight1 = 3;
    const size_t vFilterWidth1 = 3;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 2;
    const size_t vOutputDepth1 = 6;

    ConvLayer<ReLUActivation> firstConvLayer(vFilterHeight1,
                                             vFilterWidth1,
                                             vPaddingHeight1,
                                             vPaddingWidth1,
                                             vStride1,
                                             vInputDepth1,
                                             vInputHeight1,
                                             vInputWidth1,
                                             vOutputDepth1,
                                             vInputSampleNumber);
    double vSomeConstantValue = 0.45;
    MatrixXd vCustomWeights = MatrixXd::Constant(3 * 3 * 3, 6, vSomeConstantValue);
    MatrixXd vCustomBiases = MatrixXd::Zero(1, 6);
    firstConvLayer.SetCustomParams(vCustomWeights, vCustomBiases, 0.005);
    // Connect
    firstConvLayer.SetInput(Input);
    firstConvLayer.ForwardPass();

    MatrixXd vOutput = *(firstConvLayer.GetOutput());

    std::cout << "firstConvLayer.GetOutput()  ------" << std::endl;
    std::cout << vOutput.block<25,1>(0,0) << std::endl;
    std::cout << vOutput.block<25,1>(0,1) << std::endl;
}
