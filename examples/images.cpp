#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/ConvLayer.h>
#include <libdl/SigmoidActivationLayer.h>

int main()
{
  const size_t vNumSamples = 2;
  MatrixXd someMat(2,3);
  someMat << 1,0,1,2,2,5;
  MatrixXd labels(2,3);
  labels << 1,0,0,0,0,1;
  std::cout << someMat << std::endl;
  MatrixXd exp = someMat.array().exp();
  MatrixXd probs = exp.array().colwise() / exp.rowwise().sum().array();
  MatrixXd logprobs = -probs.array().log();
  MatrixXd filtered = logprobs.cwiseProduct(labels);
  double lossdata = filtered.array().sum() / static_cast<double>(vNumSamples); 
  std::cout << exp << std::endl;
  std::cout << probs << std::endl;
  std::cout << logprobs << std::endl;
  std::cout << lossdata << " lossdata" << std::endl;
  MatrixXd dscores = probs - labels;
  std::cout << dscores << " dscores" << std::endl;
  // const size_t vInputSampleNumber = 1;

  // // Input
  // const size_t vInputDepth1 = 3;
  // const size_t vInputHeight1 = 10;
  // const size_t vInputWidth1 = 9;
  // MatrixXd Input = MatrixXd::Random(vInputHeight1 * vInputWidth1, vInputDepth1);
  // std::cout << "Input" << std::endl;
  // std::cout << Input << std::endl;

  // // CONV 1
  // const size_t vFilterHeight1 = 2;
  // const size_t vFilterWidth1 = 3;
  // const size_t vPaddingHeight1 = 1;
  // const size_t vPaddingWidth1 = 1;
  // const size_t vStride1 = 2;
  // const size_t vOutputDepth1 = 6;

  // ConvLayer firstConvLayer(vFilterHeight1,
  //                          vFilterWidth1,
  //                          vPaddingHeight1,
  //                          vPaddingWidth1,
  //                          vStride1,
  //                          vInputDepth1,
  //                          vInputHeight1,
  //                          vInputWidth1,
  //                          vOutputDepth1,
  //                          vInputSampleNumber);

  // // Sigmoid.
  // SigmoidActivationLayer firstSigmoidLayer;

  // // Conv 2
  // const size_t vFilterHeight2 = 3;
  // const size_t vFilterWidth2 = 2;
  // const size_t vPaddingHeight2 = 1;
  // const size_t vPaddingWidth2 = 1;
  // const size_t vStride2 = 2;
  // const size_t vOutputDepth2 = 7;
  // ConvLayer secondConvLayer(vFilterHeight2,
  //                           vFilterWidth2,
  //                           vPaddingHeight2,
  //                           vPaddingWidth2,
  //                           vStride2,
  //                           firstConvLayer.GetOutputDims(),
  //                           vOutputDepth2,
  //                           vInputSampleNumber);

  // // Sigmoid
  // SigmoidActivationLayer secondSigmoidLayer;

  // // flatten layer

  // // Connect
  // firstConvLayer.SetInput(Input);
  // firstSigmoidLayer.SetInput(firstConvLayer.GetOutput());
  // secondConvLayer.SetInput(firstSigmoidLayer.GetOutput());
  // secondSigmoidLayer.SetInput(secondConvLayer.GetOutput());

  // firstConvLayer.ForwardPass();
  // firstSigmoidLayer.ForwardPass();
  // secondConvLayer.ForwardPass();
  // secondSigmoidLayer.ForwardPass();


  // MatrixXd BackpropInput = 0.1 * MatrixXd::Random(9, 7);
  // firstConvLayer.SetBackpropInput(firstSigmoidLayer.GetBackpropOutput());
  // firstSigmoidLayer.SetBackpropInput(secondConvLayer.GetBackpropOutput());
  // secondConvLayer.SetBackpropInput(secondSigmoidLayer.GetBackpropOutput());
  // secondSigmoidLayer.SetBackpropInput(&BackpropInput);
  
  // std::cout << *(secondSigmoidLayer.GetOutput()) << std::endl;
  // std::cout << "secondSigmoidLayer.BackwardPass()" << std::endl;
  // secondSigmoidLayer.BackwardPass();
  // std::cout << "secondConv.BackwardPass()" << std::endl;
  // secondConvLayer.BackwardPass();
  // std::cout << "firstSigmoidLayer.BackwardPass()" << std::endl;
  // firstSigmoidLayer.BackwardPass();
  // std::cout << "firstConvLayer.BackwardPass()" << std::endl;
  // firstConvLayer.BackwardPass();
}
