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
class SegmentationExample
{
  const size_t mInputHeight;
  const size_t mInputWidth;
  ConvLayer<ReLUActivation> conv1;
  ConvLayer<ReLUActivation> conv2;
  TransposedConvLayer<ReLUActivation> tran3;
  TransposedConvLayer<SigmoidActivation> tran4;
  L2LossLayer<> l2;
  NetworkHelper<> net;

public:
  SegmentationExample(const size_t aInputHeight,
                      const size_t aInputWidth,
                      const size_t aInputDepth,
                      const size_t aOutputDepth1,
                      const size_t aOutputDepth2) : mInputHeight(aInputHeight),
                                                    mInputWidth(aInputWidth),
                                                    conv1{3, 3, 1, 1, 2, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                    conv2{3, 3, 1, 1, 1, conv1.GetOutputDims(), aOutputDepth2, 1},
                                                    tran3{3, 3, 1, 1, 1, conv2.GetOutputDims(), conv2.GetInputDims(), 1},
                                                    tran4{3, 3, 1, 1, 2, tran3.GetOutputDims(), conv1.GetInputDims(), 1},
                                                    l2{},
                                                    net{{&conv1,
                                                         &conv2,
                                                         &tran3,
                                                         &tran4,
                                                         &l2}}
  {
  }
  void Train(const MatrixXd &aInput, const MatrixXd &aLabels, double aLearningRate)
  {
    conv1.mLearningRate = aLearningRate;
    conv2.mLearningRate = aLearningRate;
    tran3.mLearningRate = aLearningRate;
    tran4.mLearningRate = aLearningRate;

    for (size_t i = 0; i < 18; i++)
    {

      conv1.SetInput(aInput);
      l2.SetLabels(aLabels);

      net.FullForwardPass();
      std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
      std::cout << l2.GetLoss() << std::endl;
      net.FullBackwardPass();
    }
  }

  const MatrixXd Test(MatrixXd aInput)
  {
    conv1.SetInput(aInput);
    net.FullForwardPass();
    return *(tran4.GetOutput());
  }
};

using Eigen::MatrixXd;
int main()
{

  const size_t vInputDepth2 = 3;
  const size_t vInputHeight2 = 5;
  const size_t vInputWidth2 = 5;
  MatrixXd Input = MatrixXd::Random(vInputHeight2 * vInputWidth2, vInputDepth2);

  SegmentationExample unet(5, 5, 3, 6, 16);
  unet.Train(Input, Input,0.0005);
}
