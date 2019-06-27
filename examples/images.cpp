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
  const size_t vInputDepth1 = 3;
  const size_t vInputHeight1 = 4;
  const size_t vInputWidth1 = 4;
  MatrixXd Input = MatrixXd::Random(vInputHeight1 * vInputWidth1, vInputDepth1);


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
                                                1);
  someLayer.SetInput(Input);
  someLayer.ForwardPass();
  someLayer.SetBackpropInput(someLayer.GetOutput());
  someLayer.BackwardPass();

  std::cout << "-----------------someLayer2------------" << std::endl;
  const size_t vInputDepth2 = 2;
  const size_t vInputHeight2 = 4;
  const size_t vInputWidth2 = 4;
  MatrixXd Input2 = MatrixXd::Random(vInputHeight2 * vInputWidth2, vInputDepth2);



  // CONV 2

  const size_t vFilterHeight2 = 3;
  const size_t vFilterWidth2 = 3;
  const size_t vPaddingHeight2 = 1;
  const size_t vPaddingWidth2 = 1;
  const size_t vStride2 = 1;
  const size_t vOutputDepth2 = 3;

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
  someLayer2.SetBackpropInput(someLayer2.GetOutput());
  someLayer2.BackwardPass();
}
