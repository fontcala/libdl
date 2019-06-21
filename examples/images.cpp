#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/FullyConnectedLayer.h>

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
  const size_t vFilterHeight1 = 2;
  const size_t vFilterWidth1 = 3;
  const size_t vPaddingHeight1 = 1;
  const size_t vPaddingWidth1 = 1;
  const size_t vStride1 = 2;
  const size_t vOutputDepth1 = 6;

  ConvLayer<SigmoidActivation, MatrixXd> firstConvLayer(vFilterHeight1,
                                                        vFilterWidth1,
                                                        vPaddingHeight1,
                                                        vPaddingWidth1,
                                                        vStride1,
                                                        vInputDepth1,
                                                        vInputHeight1,
                                                        vInputWidth1,
                                                        vOutputDepth1,
                                                        vInputSampleNumber);

  // Conv 2
  const size_t vFilterHeight2 = 3;
  const size_t vFilterWidth2 = 2;
  const size_t vPaddingHeight2 = 1;
  const size_t vPaddingWidth2 = 1;
  const size_t vStride2 = 2;
  const size_t vOutputDepth2 = 7;
  ConvLayer<SigmoidActivation, MatrixXd> secondConvLayer(vFilterHeight2,
                                                         vFilterWidth2,
                                                         vPaddingHeight2,
                                                         vPaddingWidth2,
                                                         vStride2,
                                                         firstConvLayer.GetOutputDims(),
                                                         vOutputDepth2,
                                                         vInputSampleNumber);


  // flatten layer
  FlattenLayer<MatrixXd> flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

  // fullyconnectedlayer
  FullyConnectedLayer<LinearActivation, MatrixXd> fcLayer(flattenLayer.GetOutputDims(), vNumCategories);

  // losslayer
  SoftmaxLossLayer<MatrixXd> lossLayer;
  lossLayer.SetLabels(Label);

  // Connect
  firstConvLayer.SetInput(Input);
  secondConvLayer.SetInput(firstConvLayer.GetOutput());
  flattenLayer.SetInput(secondConvLayer.GetOutput());
  fcLayer.SetInput(flattenLayer.GetOutput());
  lossLayer.SetInput(fcLayer.GetOutput());

  firstConvLayer.SetBackpropInput(secondConvLayer.GetBackpropOutput());
  secondConvLayer.SetBackpropInput(flattenLayer.GetBackpropOutput());
  flattenLayer.SetBackpropInput(fcLayer.GetBackpropOutput());
  fcLayer.SetBackpropInput(lossLayer.GetBackpropOutput());

  for (size_t i = 0; i < 5; i++)
  {
    std::cout << "---------start forward ---------" << std::endl;
    std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
    firstConvLayer.ForwardPass();
    std::cout << "secondConvLayer.ForwardPass()  ------" << std::endl;
    secondConvLayer.ForwardPass();
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
    std::cout << "secondConv.BackwardPass()  ------" << std::endl;
    secondConvLayer.BackwardPass();
    std::cout << "firstConvLayer.BackwardPass()  ------" << std::endl;
    firstConvLayer.BackwardPass();
  }
}
