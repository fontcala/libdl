#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/FullyConnectedLayer.h>

using Eigen::MatrixXd;
int main()
{
  const size_t vInputSampleNumber = 1;
  const size_t vNumCategories = 2;
  const size_t vInputDepth1 = 3;
  const size_t vInputHeight1 = 4;
  const size_t vInputWidth1 = 4;
  MatrixXd Input = MatrixXd::Random(vInputHeight1 * vInputWidth1, vInputDepth1);
  MatrixXd Label(vInputSampleNumber, vNumCategories);
  Label << 0, 1;

  // CONV 1
  const size_t vFilterHeight1 = 3;
  const size_t vFilterWidth1 = 3;
  const size_t vPaddingHeight1 = 1;
  const size_t vPaddingWidth1 = 1;
  const size_t vStride1 = 1;
  const size_t vOutputDepth1 = 2;

  TransposedConvLayer<ReLUActivation> someLayer(vFilterHeight1,
                                                vFilterWidth1,
                                                vPaddingHeight1,
                                                vPaddingWidth1,
                                                vStride1,
                                                vInputDepth1,
                                                vInputHeight1,
                                                vInputWidth1,
                                                vOutputDepth1,
                                                vInputSampleNumber);
someLayer.SetInput(Input);
someLayer.ForwardPass();
}
