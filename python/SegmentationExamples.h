#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/NetworkHelper.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/L2LossLayer.h>
#include <libdl/FullyConnectedLayer.h>
using Eigen::MatrixXd;

class SegmentationExample1
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<SigmoidActivation> conv1;
    TransposedConvLayer<SigmoidActivation> tonv2;
    SoftmaxLossLayer<> l2;
    NetworkHelper<> net;

public:
    SegmentationExample1(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aOutputDepth1,
                         const size_t aLabelDepth) : mInputHeight(aInputHeight),
                                                       mInputWidth(aInputWidth),
                                                       conv1{3, 3, 0, 0, 1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                       tonv2{3, 3, 0, 0, 1, conv1.GetOutputDims(), aLabelDepth, 1},
                                                       l2{},
                                                       net{{&conv1,
                                                            &tonv2,
                                                            &l2}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aNumber)
    {
        conv1.mLearningRate = aLearningRate;
        tonv2.mLearningRate = aLearningRate;

        for (size_t i = 0; i < aNumber; i++)
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
        std::cout << "Test Sample:" << std::endl;
        conv1.SetInput(aInput);
        return net.FullForwardTestPass();
    }
};


