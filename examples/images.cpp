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

  std::cout << "-----------------normal Convolution------------" << std::endl;
  const size_t vInputDepth2 = 3;
  const size_t vInputHeight2 = 5;
  const size_t vInputWidth2 = 5;
  MatrixXd Input2 = MatrixXd::Random(vInputHeight2 * vInputWidth2, vInputDepth2);

  const size_t vBackpropInputDepth2= 6;
  const size_t vBackpropInputHeight2 = 3;
  const size_t vBackpropInputWidth2 = 3;
  MatrixXd vBackpropInput2 = MatrixXd::Random(vBackpropInputHeight2 * vBackpropInputWidth2, vBackpropInputDepth2);


  // CONV 2

  const size_t vFilterHeight2 = 3;
  const size_t vFilterWidth2 = 3;
  const size_t vPaddingHeight2 = 1;
  const size_t vPaddingWidth2 = 1;
  const size_t vStride2 = 2;
  const size_t vOutputDepth2 = 6;

  ConvLayer<SigmoidActivation> someLayer2(vFilterHeight2,
                                      vFilterWidth2,
                                      vPaddingHeight2,
                                      vPaddingWidth2,
                                      vStride2,
                                      vInputDepth2,
                                      vInputHeight2,
                                      vInputWidth2,
                                      vOutputDepth2,
                                      1);
  someLayer2.SetInput(Input2);
  someLayer2.ForwardPass();
  someLayer2.SetBackpropInput(&vBackpropInput2);
  someLayer2.BackwardPass();

  std::cout << "-----------------transposed Convolution------------" << std::endl;
  const size_t vInputDepth1 = 6;
  const size_t vInputHeight1 = 3;
  const size_t vInputWidth1 = 3;
  MatrixXd Input = MatrixXd::Random(vInputHeight1 * vInputWidth1, vInputDepth1);

  const size_t vBackpropInputDepth1 = 3;
  const size_t vBackpropInputHeight1 = 5;
  const size_t vBackpropInputWidth1 = 5;
  MatrixXd vBackpropInput1 = MatrixXd::Random(vBackpropInputHeight1 * vBackpropInputWidth1, vBackpropInputDepth1);


  // CONV 1
  const size_t vFilterHeight1 = 3;
  const size_t vFilterWidth1 = 3;
  const size_t vPaddingHeight1 = 1;
  const size_t vPaddingWidth1 = 1;
  const size_t vStride1 = 2;
  const size_t vOutputDepth1 = 3;

  TransposedConvLayer<ReLUActivation> someLayer(vFilterHeight1,
                                                vFilterWidth1,
                                                vPaddingHeight1,
                                                vPaddingWidth1,
                                                vStride1,
                                                vInputDepth1,
                                                vInputHeight1,
                                                vInputWidth1,
                                                vOutputDepth1,
                                                1);
  someLayer.SetInput(Input);
  someLayer.ForwardPass();
  someLayer.SetBackpropInput(&vBackpropInput1);
  someLayer.BackwardPass();

}
