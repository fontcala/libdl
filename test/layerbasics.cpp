#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/FullyConnectedLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <type_traits>

using Eigen::MatrixXd;
TEST_CASE("network overfit a single noise sample, without maxpool", "network")
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

  // Conv 2
  const size_t vFilterHeight2 = 3;
  const size_t vFilterWidth2 = 2;
  const size_t vPaddingHeight2 = 1;
  const size_t vPaddingWidth2 = 1;
  const size_t vStride2 = 2;
  const size_t vOutputDepth2 = 7;
  ConvLayer<ReLUActivation> secondConvLayer(vFilterHeight2,
                                            vFilterWidth2,
                                            vPaddingHeight2,
                                            vPaddingWidth2,
                                            vStride2,
                                            firstConvLayer.GetOutputDims(),
                                            vOutputDepth2,
                                            vInputSampleNumber);

  // flatten layer
  FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

  // fullyconnectedlayer
  FullyConnectedLayer<LinearActivation> fcLayer(flattenLayer.GetOutputDims(), vNumCategories);

  // losslayer
  SoftmaxLossLayer lossLayer{};
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

  double vPreviousLoss = std::numeric_limits<double>::max();
  const double cTolerance = 0.000000000000001;
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
    REQUIRE(vPreviousLoss - lossLayer.GetLoss() > -cTolerance);
    vPreviousLoss = lossLayer.GetLoss();
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

TEST_CASE("network overfit one single example, with max pool", "network")
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

  const size_t vPoolSize = 2;
  const size_t vStridePool = 2;
  MaxPoolLayer mpLayer(firstConvLayer.GetOutputDims(),
                       vPoolSize,
                       vStridePool,
                       vInputSampleNumber);

  // Conv 2
  const size_t vFilterHeight2 = 3;
  const size_t vFilterWidth2 = 2;
  const size_t vPaddingHeight2 = 1;
  const size_t vPaddingWidth2 = 1;
  const size_t vStride2 = 2;
  const size_t vOutputDepth2 = 7;
  ConvLayer<ReLUActivation> secondConvLayer(vFilterHeight2,
                                            vFilterWidth2,
                                            vPaddingHeight2,
                                            vPaddingWidth2,
                                            vStride2,
                                            mpLayer.GetOutputDims(),
                                            vOutputDepth2,
                                            vInputSampleNumber);

  // flatten layer
  FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

  // fullyconnectedlayer
  FullyConnectedLayer<LinearActivation> fcLayer(flattenLayer.GetOutputDims(), vNumCategories);

  // losslayer
  SoftmaxLossLayer lossLayer{};
  lossLayer.SetLabels(Label);

  // Connect
  firstConvLayer.SetInput(Input);
  mpLayer.SetInput(firstConvLayer.GetOutput());
  secondConvLayer.SetInput(mpLayer.GetOutput());
  flattenLayer.SetInput(secondConvLayer.GetOutput());
  fcLayer.SetInput(flattenLayer.GetOutput());
  lossLayer.SetInput(fcLayer.GetOutput());

  firstConvLayer.SetBackpropInput(mpLayer.GetBackpropOutput());
  mpLayer.SetBackpropInput(secondConvLayer.GetBackpropOutput());
  secondConvLayer.SetBackpropInput(flattenLayer.GetBackpropOutput());
  flattenLayer.SetBackpropInput(fcLayer.GetBackpropOutput());
  fcLayer.SetBackpropInput(lossLayer.GetBackpropOutput());

  double vPreviousLoss = std::numeric_limits<double>::max();
  const double cTolerance = 0.000000000000001;
  for (size_t i = 0; i < 5; i++)
  {
    std::cout << "---------start forward ---------" << std::endl;
    std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
    firstConvLayer.ForwardPass();
    std::cout << "mpLayer.ForwardPass()  ------" << std::endl;
    mpLayer.ForwardPass();
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
    REQUIRE(vPreviousLoss - lossLayer.GetLoss() > -cTolerance);
    vPreviousLoss = lossLayer.GetLoss();
    std::cout << "---------start backward ---------" << std::endl;
    std::cout << "lossLayer.BackwardPass()  ------" << std::endl;
    lossLayer.BackwardPass();
    std::cout << "fcLayer.BackwardPass()  ------" << std::endl;
    fcLayer.BackwardPass();
    std::cout << "flattenLayer.BackwardPass()  ------" << std::endl;
    flattenLayer.BackwardPass();
    std::cout << "secondConv.BackwardPass()  ------" << std::endl;
    secondConvLayer.BackwardPass();
    std::cout << "mpLayer.BackwardPass()  ------" << std::endl;
    mpLayer.BackwardPass();
    std::cout << "firstConvLayer.BackwardPass()  ------" << std::endl;
    firstConvLayer.BackwardPass();
  }
}

TEST_CASE("maxpool checks", "network")
{
  MatrixXd InputVol1(16, 1);
  MatrixXd InputVol2(16, 1);
  MatrixXd InputVol3(16, 1);
  InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316;
  InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
  InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216;
  MatrixXd InputVol(16, 3);
  InputVol << InputVol1, InputVol2, InputVol3;
  std::cout << "InputVol" << std::endl;
  std::cout << InputVol << std::endl;

  // Parameters
  const size_t tFILTERSIZE = 2;
  size_t vNumChannelsVol = 3;
  size_t vImageHeightVol = 4;
  size_t vImageWidthVol = 4;
  size_t vStrideVol = 2;
  size_t vPaddingVol = 0;
  size_t vOutHeightVol = (vImageHeightVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
  size_t vOutWidthVol = (vImageWidthVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
  size_t vOutFieldsVol = tFILTERSIZE * tFILTERSIZE;
  size_t vNumSamples = 1;

  MaxPoolLayer maxP(vNumChannelsVol, vImageHeightVol, vImageWidthVol, tFILTERSIZE, vStrideVol, vNumSamples);
  maxP.SetInput(InputVol);
  maxP.ForwardPass();
  std::cout << "*(maxP.GetOutput())" << std::endl;
  std::cout << *(maxP.GetOutput()) << std::endl;
  maxP.SetBackpropInput(maxP.GetOutput());
  maxP.BackwardPass();
  std::cout << "*(maxP.GetBackpropOutput())" << std::endl;
  std::cout << *(maxP.GetBackpropOutput()) << std::endl;

  MatrixXd ShouldVol1(16, 1);
  MatrixXd ShouldVol2(16, 1);
  MatrixXd ShouldVol3(16, 1);
  ShouldVol3 << 0, 0, 0, 0, 0, 306, 0, 308, 0, 0, 0, 0, 0, 314, 0, 316;
  ShouldVol1 << 0, 0, 0, 0, 0, 106, 0, 108, 0, 0, 0, 0, 0, 114, 0, 116;
  ShouldVol2 << 0, 0, 0, 0, 0, 206, 0, 208, 0, 0, 0, 0, 0, 214, 0, 216;
  MatrixXd ShouldVol(16, 3);
  ShouldVol << ShouldVol1, ShouldVol2, ShouldVol3;

  REQUIRE(*(maxP.GetBackpropOutput()) == ShouldVol);
}