#include <iostream>
#include <type_traits>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FullyConnectedLayer.h>
#include <math.h>

using Eigen::MatrixXd;
int main()
{
    MatrixXd Input = MatrixXd::Random(16, 3);
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = 3;
    const size_t vFilterWidth1 = 3;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 1;
    const size_t vOutputDepth1 = 3;

    ConvLayer<ReLUActivation> firstConvLayer(vFilterHeight1,
                                             vFilterWidth1,
                                             vPaddingHeight1,
                                             vPaddingWidth1,
                                             vStride1,
                                             3,
                                             4,
                                             4,
                                             vOutputDepth1,
                                             vInputSampleNumber);
    firstConvLayer.SetData(Input);

    // Conv 2
    const size_t vFilterHeight2 = 3;
    const size_t vFilterWidth2 = 3;
    const size_t vPaddingHeight2 = 1;
    const size_t vPaddingWidth2 = 1;
    const size_t vStride2 = 1;
    const size_t vOutputDepth2 = 3;
    ConvLayer<ReLUActivation> secondConvLayer(vFilterHeight2,
                                              vFilterWidth2,
                                              vPaddingHeight2,
                                              vPaddingWidth2,
                                              vStride2,
                                              firstConvLayer.GetOutputDims(),
                                              vOutputDepth2,
                                              vInputSampleNumber);

    // Conv 3
    const size_t vFilterHeight3 = 3;
    const size_t vFilterWidth3 = 3;
    const size_t vPaddingHeight3 = 1;
    const size_t vPaddingWidth3 = 1;
    const size_t vStride3 = 1;
    const size_t vOutputDepth3 = 3;
    ConvLayer<ReLUActivation> thirdConvLayer(vFilterHeight3,
                                             vFilterWidth3,
                                             vPaddingHeight3,
                                             vPaddingWidth3,
                                             vStride3,
                                             secondConvLayer.GetOutputDims(),
                                             vOutputDepth3,
                                             vInputSampleNumber);

    // Connect
    secondConvLayer.SetInput(firstConvLayer.GetOutput());
    thirdConvLayer.SetInput(secondConvLayer.GetOutput());

    firstConvLayer.SetBackpropInput(secondConvLayer.GetBackpropOutput());
    secondConvLayer.SetBackpropInput(thirdConvLayer.GetBackpropOutput());

    // RUN
    firstConvLayer.ForwardPass();
    secondConvLayer.ForwardPass();
    thirdConvLayer.ForwardPass();

    std::cout << "*firstConvLayer.GetOutput()" << std::endl;

    std::cout << *firstConvLayer.GetOutput() << std::endl;

    std::cout << "*secondConvLayer.GetOutput()" << std::endl;

    std::cout << *secondConvLayer.GetOutput() << std::endl;

    std::cout << "*thirdConvLayer.GetOutput()" << std::endl;

    std::cout << *thirdConvLayer.GetOutput() << std::endl;

    std::cout << "sum" << std::endl;

    std::cout << *thirdConvLayer.GetOutput() + *secondConvLayer.GetOutput() + *firstConvLayer.GetOutput() << std::endl; 

    thirdConvLayer.ForwardAdditionSkipConnection(firstConvLayer,secondConvLayer);
    std::cout << "*thirdConvLayer.GetOutput() skip" << std::endl;

    std::cout << *thirdConvLayer.GetOutput() << std::endl;
}
