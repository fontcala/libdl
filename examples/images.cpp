#include <iostream>
#include <libdl/NetworkHelper.h>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/L2LossLayer.h>
using Eigen::MatrixXd;
// Autoencoder 
class Example
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<ReLUActivation> conv1;
    MaxPoolLayer<> maxp1;
    ConvLayer<ReLUActivation> conv2;
    MaxPoolLayer<> maxp2;
    TransposedConvLayer<ReLUActivation> tran3;
    TransposedConvLayer<SigmoidActivation> tran4;
    L2LossLayer<> loss;
    NetworkHelper<> net;

public:
    Example(const size_t aInputHeight,
            const size_t aInputWidth,
            const size_t aInputDepth,
            const size_t aFirstDepth,
            const size_t aSecondDepth) : mInputHeight(aInputHeight),
                                         mInputWidth(aInputWidth),
                                         conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, aFirstDepth, 1, UpdateMethod::ADAM},
                                         maxp1{conv1.GetOutputDims(), 2, 2, 1},
                                         conv2{3, 3, 1, 1, 1, maxp1.GetOutputDims(), aSecondDepth, 1, UpdateMethod::ADAM},
                                         maxp2{conv2.GetOutputDims(), 2, 2, 1},
                                         tran3{2, 2, 0, 0, 2, maxp2.GetOutputDims(), aFirstDepth, 1, UpdateMethod::ADAM},
                                         tran4{2, 2, 0, 0, 2, tran3.GetOutputDims(), aInputDepth, 1, UpdateMethod::ADAM},
                                         loss{10},
                                         net{{&conv1,
                                              &maxp1,
                                              &conv2,
                                              &maxp2,
                                              &tran3,
                                              &tran4,
                                              &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const double aLearningRate, const size_t aEpochNum)
    {
        // This example assumes a user wanting the same learning rate for all layers
        conv1.SetLearningRate(aLearningRate);
        conv2.SetLearningRate(aLearningRate);
        tran3.SetLearningRate(aLearningRate);
        tran4.SetLearningRate(aLearningRate);

        // We need to randomly pick sample indices from the input in each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        const size_t vTotalTrainSamples = aInput.cols();
        std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
        std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0);
        for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
        {
            double vLoss = 0;
            std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
            for (const auto &vIndex : vIndexTrainVector)
            {
                // take a sample from the data and feed it to the network (in this autoencoder example the label is also the input)
                const MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
                net.SetInputData(Input);
                net.SetLabelData(Input);
                net.FullForwardPass();
                net.FullBackwardPass();
                vLoss = vLoss + loss.GetLoss();
            }
            std::cout << vLoss / aInput.cols() << std::endl;
        }
    }

    const MatrixXd Test(MatrixXd &aInput)
    {
        net.SetInputData(aInput);
        return net.FullForwardTestPass();
    }
};

int main()
{
    // Example with noise
    const size_t vInputHeight = 28;
    const size_t vInputWidth = 28;
    const size_t vInputDepth = 1;
    MatrixXd someImage = MatrixXd::Random(vInputHeight * vInputWidth, vInputDepth);

    // Construct with desired parameters and test with the same input
    Example AutoEncoder(vInputHeight,vInputWidth,vInputDepth,16,4);
    AutoEncoder.Train(someImage,0.0000001,10);
    AutoEncoder.Test(someImage);
}
