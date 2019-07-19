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
    SoftmaxLossLayer<> loss;
    NetworkHelper<> net;

public:
    SegmentationExample1(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aOutputDepth1,
                         const size_t aLabelDepth) : mInputHeight(aInputHeight),
                                                     mInputWidth(aInputWidth),
                                                     conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                     tonv2{3, 3, 1, 1, 1, conv1.GetOutputDims(), aLabelDepth, 1},
                                                     loss{},
                                                     net{{&conv1,
                                                          &tonv2,
                                                          &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aNumber)
    {
        conv1.mLearningRate = aLearningRate;
        tonv2.mLearningRate = aLearningRate;

        for (size_t i = 0; i < aNumber; i++)
        {

            conv1.SetData(aInput);
            loss.SetData(aLabels);

            net.FullForwardPass();
            std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            std::cout << loss.GetLoss() << std::endl;
            net.FullBackwardPass();
        }
    }

    const MatrixXd Test(MatrixXd aInput)
    {
        std::cout << "Test Sample:" << std::endl;
        conv1.SetData(aInput);
        return net.FullForwardTestPass();
    }
};

class SegmentationExample2
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;
    ConvLayer<ReLUActivation> conv1;
    ConvLayer<ReLUActivation> conv2;
    MaxPoolLayer<> maxp1;
    ConvLayer<ReLUActivation> conv3;
    ConvLayer<ReLUActivation> conv4;
    TransposedConvLayer<SigmoidActivation> tran1;
    ConvLayer<ReLUActivation> conv5;
    ConvLayer<ReLUActivation> conv6;
    ConvLayer<ReLUActivation> conv7;
    SoftmaxLossLayer<> loss;
    NetworkHelper<> net;

public:
    SegmentationExample2(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aOutputDepth1,
                         const size_t aLabelDepth) : mInputHeight(aInputHeight),
                                                     mInputWidth(aInputWidth),
                                                     mInputDepth(aInputDepth),
                                                     mLabelDepth(aLabelDepth),
                                                     conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                     conv2{3, 3, 1, 1, 1, conv1.GetOutputDims(), aOutputDepth1, 1},
                                                     maxp1{conv2.GetOutputDims(), 2, 2, 1},
                                                     conv3{3, 3, 1, 1, 1, maxp1.GetOutputDims(), aOutputDepth1, 1},
                                                     conv4{3, 3, 1, 1, 1, conv3.GetOutputDims(), aOutputDepth1 * 2, 1},
                                                     tran1{2, 2, 0, 0, 2, conv4.GetOutputDims(), aOutputDepth1 * 2, 1},
                                                     conv5{3, 3, 1, 1, 1, tran1.GetOutputDims(), aOutputDepth1 * 2, 1},
                                                     conv6{3, 3, 1, 1, 1, conv5.GetOutputDims(), aOutputDepth1, 1},
                                                     conv7{3, 3, 1, 1, 1, conv6.GetOutputDims(), aLabelDepth, 1},
                                                     loss{},
                                                     net{{&conv1,
                                                          &conv2,
                                                          &maxp1,
                                                          &conv3,
                                                          &conv4,
                                                          &tran1,
                                                          &conv5,
                                                          &conv6,
                                                          &conv7,
                                                          &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aEpochNum)
    {
        conv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;
        conv3.mLearningRate = aLearningRate;
        conv4.mLearningRate = aLearningRate;
        conv5.mLearningRate = aLearningRate;
        conv6.mLearningRate = aLearningRate;
        conv7.mLearningRate = aLearningRate;
        tran1.mLearningRate = aLearningRate;

        conv1.mMomentumUpdateParam = 0.99;
        conv2.mMomentumUpdateParam = 0.99;
        conv3.mMomentumUpdateParam = 0.99;
        conv4.mMomentumUpdateParam = 0.99;
        conv5.mMomentumUpdateParam = 0.99;
        conv6.mMomentumUpdateParam = 0.99;
        conv7.mMomentumUpdateParam = 0.99;
        tran1.mMomentumUpdateParam = 0.99;

        std::random_device rd;
        std::mt19937 g(rd());
        const size_t vTotalTrainSamples = aInput.cols() / mInputDepth;
        std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
        std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
        for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
        {
            std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
            for (const auto &vIndex : vIndexTrainVector)
            {
                std::cout << vIndex << " vIndex" << std::endl;
                MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, mInputDepth);
                MatrixXd Labels = aLabels.block(0, vIndex * mLabelDepth, mInputHeight * mInputWidth, mLabelDepth);

                conv1.SetData(Input);
                loss.SetData(Labels);
                net.FullForwardPass();
                net.FullBackwardPass();
                std::cout << "Loss of a given sample at epoch: " << vEpoch << std::endl;
                std::cout << loss.GetLoss() << std::endl;
            }
        }
    }

    const MatrixXd
    Test(MatrixXd aInput)
    {
        std::cout << "Test Sample:" << std::endl;
        conv1.SetData(aInput);
        return net.FullForwardTestPass();
    }
};

class SegmentationExample3
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;
    ConvLayer<SigmoidActivation> conv1;
    ConvLayer<SigmoidActivation> conv2;
    ConvLayer<SigmoidActivation> conv3;
    ConvLayer<SigmoidActivation> conv4;
    ConvLayer<SigmoidActivation> conv5;
    ConvLayer<SigmoidActivation> conv6;
    ConvLayer<SigmoidActivation> conv7;
    ConvLayer<SigmoidActivation> conv8;
    TransposedConvLayer<SigmoidActivation> tran8;
    TransposedConvLayer<SigmoidActivation> tran7;
    TransposedConvLayer<SigmoidActivation> tran6;
    TransposedConvLayer<SigmoidActivation> tran5;
    TransposedConvLayer<SigmoidActivation> tran4;
    TransposedConvLayer<SigmoidActivation> tran3;
    TransposedConvLayer<SigmoidActivation> tran2;
    TransposedConvLayer<SigmoidActivation> tran1;
    SoftmaxLossLayer<> loss;
    NetworkHelper<> net;

public:
    SegmentationExample3(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aLabelDepth,
                         const size_t aFilterSize1,
                         const size_t aFilterSize2,
                         const size_t aFilterSize3,
                         const size_t aFilterSize4,
                         const size_t aFilterSize5,
                         const size_t aFilterSize6,
                         const size_t aFilterSize7,
                         const size_t aFilterSize8) : mInputHeight(aInputHeight),
                                                      mInputWidth(aInputWidth),
                                                      mInputDepth(aInputDepth),
                                                      mLabelDepth(aLabelDepth),
                                                      conv1{aFilterSize1, aFilterSize1, 0, 0, 1, aInputDepth, aInputHeight, aInputWidth, 2, 1},
                                                      conv2{aFilterSize2, aFilterSize2, 0, 0, 1, conv1.GetOutputDims(), 2, 1},
                                                      conv3{aFilterSize3, aFilterSize3, 0, 0, 1, conv2.GetOutputDims(), 2, 1},
                                                      conv4{aFilterSize4, aFilterSize4, 0, 0, 1, conv3.GetOutputDims(), 2, 1},
                                                      conv5{aFilterSize5, aFilterSize5, 0, 0, 1, conv4.GetOutputDims(), 4, 1},
                                                      conv6{aFilterSize6, aFilterSize6, 0, 0, 1, conv5.GetOutputDims(), 4, 1},
                                                      conv7{aFilterSize7, aFilterSize7, 0, 0, 1, conv6.GetOutputDims(), 4, 1},
                                                      conv8{aFilterSize8, aFilterSize8, 0, 0, 1, conv7.GetOutputDims(), 5, 1},
                                                      tran8{aFilterSize8, aFilterSize8, 0, 0, 1, conv8.GetOutputDims(), conv8.GetInputDims(), 1},
                                                      tran7{aFilterSize7, aFilterSize7, 0, 0, 1, tran8.GetOutputDims(), conv7.GetInputDims(), 1},
                                                      tran6{aFilterSize6, aFilterSize6, 0, 0, 1, tran7.GetOutputDims(), conv6.GetInputDims(), 1},
                                                      tran5{aFilterSize5, aFilterSize5, 0, 0, 1, tran6.GetOutputDims(), conv5.GetInputDims(), 1},
                                                      tran4{aFilterSize4, aFilterSize4, 0, 0, 1, tran5.GetOutputDims(), conv4.GetInputDims(), 1},
                                                      tran3{aFilterSize3, aFilterSize3, 0, 0, 1, tran4.GetOutputDims(), conv3.GetInputDims(), 1},
                                                      tran2{aFilterSize2, aFilterSize2, 0, 0, 1, tran3.GetOutputDims(), conv2.GetInputDims(), 1},
                                                      tran1{aFilterSize1, aFilterSize1, 0, 0, 1, tran2.GetOutputDims(), aLabelDepth, 1},
                                                      loss{},
                                                      net{{&conv1,
                                                           &conv2,
                                                           &conv3,
                                                           &conv4,
                                                           &conv5,
                                                           &conv6,
                                                           &conv7,
                                                           &conv8,
                                                           &tran8,
                                                           &tran7,
                                                           &tran6,
                                                           &tran5,
                                                           &tran4,
                                                           &tran3,
                                                           &tran2,
                                                           &tran1,
                                                           &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aEpochNum)
    {
        conv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;
        conv3.mLearningRate = aLearningRate;
        conv4.mLearningRate = aLearningRate;
        conv5.mLearningRate = aLearningRate;
        conv6.mLearningRate = aLearningRate;
        conv7.mLearningRate = aLearningRate;
        conv8.mLearningRate = aLearningRate;
        tran8.mLearningRate = aLearningRate;
        tran7.mLearningRate = aLearningRate;
        tran6.mLearningRate = aLearningRate;
        tran5.mLearningRate = aLearningRate;
        tran4.mLearningRate = aLearningRate;
        tran3.mLearningRate = aLearningRate;
        tran2.mLearningRate = aLearningRate;
        tran1.mLearningRate = aLearningRate;

        conv1.mMomentumUpdateParam = 0.99;
        conv2.mMomentumUpdateParam = 0.99;
        conv3.mMomentumUpdateParam = 0.99;
        conv4.mMomentumUpdateParam = 0.99;
        conv5.mMomentumUpdateParam = 0.99;
        conv6.mMomentumUpdateParam = 0.99;
        conv7.mMomentumUpdateParam = 0.99;
        conv8.mMomentumUpdateParam = 0.99;
        tran8.mMomentumUpdateParam = 0.99;
        tran7.mMomentumUpdateParam = 0.99;
        tran6.mMomentumUpdateParam = 0.99;
        tran5.mMomentumUpdateParam = 0.99;
        tran4.mMomentumUpdateParam = 0.99;
        tran3.mMomentumUpdateParam = 0.99;
        tran2.mMomentumUpdateParam = 0.99;
        tran1.mMomentumUpdateParam = 0.99;

        std::random_device rd;
        std::mt19937 g(rd());
        const size_t vTotalTrainSamples = aInput.cols() / mInputDepth;
        std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
        std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
        for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
        {
            std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
            for (const auto &vIndex : vIndexTrainVector)
            {
                MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, mInputDepth);
                MatrixXd Labels = aLabels.block(0, vIndex * mLabelDepth, mInputHeight * mInputWidth, mLabelDepth);

                conv1.SetData(Input);
                loss.SetData(Labels);
                net.FullForwardPass();
                net.FullBackwardPass();
            }
            std::cout << "Loss of a given sample at epoch: " << vEpoch << std::endl;
            std::cout << loss.GetLoss() << std::endl;
        }
    }

    const MatrixXd Test(MatrixXd aInput)
    {
        std::cout << "testing:" << std::endl;
        conv1.SetData(aInput);
        return net.FullForwardTestPass();
    }
};

class SegmentationExample6
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;
    ConvLayer<ReLUActivation> conv1;
    MaxPoolLayer<> maxp1;
    ConvLayer<ReLUActivation> conv2;
    MaxPoolLayer<> maxp2;
    TransposedConvLayer<ReLUActivation> tran3;
    TransposedConvLayer<SigmoidActivation> tran4;
    L2LossLayer<> loss;
    NetworkHelper<> net;

public:
    SegmentationExample6(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aLabelDepth) : mInputHeight(aInputHeight),
                                                     mInputWidth(aInputWidth),
                                                     mInputDepth(aInputDepth),
                                                     mLabelDepth(aLabelDepth),
                                                     conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, 16, 1, UpdateMethod::ADAM},
                                                     maxp1{conv1.GetOutputDims(), 2, 2, 1},
                                                     conv2{3, 3, 1, 1, 1, maxp1.GetOutputDims(), 32, 1, UpdateMethod::ADAM},
                                                     maxp2{conv2.GetOutputDims(), 2, 2, 1},
                                                     tran3{2, 2, 0, 0, 2, maxp2.GetOutputDims(), 16, 1, UpdateMethod::ADAM},
                                                     tran4{2, 2, 0, 0, 2, tran3.GetOutputDims(), 2, 1, UpdateMethod::ADAM},
                                                     loss{},
                                                     net{{&conv1,
                                                          &maxp1,
                                                          &conv2,
                                                          &maxp2,
                                                          &tran3,
                                                          &tran4,
                                                          &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aEpochNum)
    {
        conv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;
        tran4.mLearningRate = aLearningRate;
        tran3.mLearningRate = aLearningRate;

        std::random_device rd;
        std::mt19937 g(rd());
        const size_t vTotalTrainSamples = aInput.cols() / mInputDepth;
        std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
        std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
        for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
        {
            std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
            for (const auto &vIndex : vIndexTrainVector)
            {
                MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, mInputDepth);
                MatrixXd Labels = aLabels.block(0, vIndex * mLabelDepth, mInputHeight * mInputWidth, mLabelDepth);

                conv1.SetData(Input);
                loss.SetData(Labels);
                net.FullForwardPass();
                net.FullBackwardPass();
            }
            std::cout << "Loss of a given sample at epoch: " << vEpoch << std::endl;
            std::cout << loss.GetLoss() << std::endl;
        }
    }

    const MatrixXd Test(MatrixXd aInput)
    {
        std::cout << "testing:" << std::endl;
        conv1.SetData(aInput);
        return net.FullForwardTestPass();
    }
};

class SegmentationExample7
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<SigmoidActivation> conv1;
    ConvLayer<SigmoidActivation> conv2;
    TransposedConvLayer<SigmoidActivation> tonv2;
    TransposedConvLayer<SigmoidActivation> tonv1;
    SoftmaxLossLayer<> l2;
    NetworkHelper<> net;

public:
    SegmentationExample7(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aOutputDepth1,
                         const size_t aOutputDepth2,
                         const size_t aLabelDepth) : mInputHeight(aInputHeight),
                                                     mInputWidth(aInputWidth),
                                                     conv1{3, 3, 0, 0, 1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                     conv2{3, 3, 0, 0, 1, conv1.GetOutputDims(), aOutputDepth2, 1},
                                                     tonv2{3, 3, 0, 0, 1, conv2.GetOutputDims(), aOutputDepth1, 1},
                                                     tonv1{3, 3, 0, 0, 1, tonv2.GetOutputDims(), aLabelDepth, 1},
                                                     l2{},
                                                     net{{&conv1,
                                                          &conv2,
                                                          &tonv2,
                                                          &tonv1,
                                                          &l2}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aNumber)
    {
        conv1.mLearningRate = aLearningRate;
        tonv2.mLearningRate = aLearningRate;
        tonv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;

        for (size_t i = 0; i < aNumber; i++)
        {

            conv1.SetData(aInput);
            l2.SetData(aLabels);

            net.FullForwardPass();
            std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            std::cout << l2.GetLoss() << std::endl;
            net.FullBackwardPass();
        }
    }

    const MatrixXd Test(MatrixXd aInput)
    {
        std::cout << "Test Sample:" << std::endl;
        conv1.SetData(aInput);
        return net.FullForwardTestPass();
    }
};

class SegmentationExample8
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;
    ConvLayer<ReLUActivation> conv1;
    ConvLayer<ReLUActivation> conv1s;
    MaxPoolLayer<> maxp1;
    ConvLayer<ReLUActivation> conv2;
    ConvLayer<ReLUActivation> conv2s;
    MaxPoolLayer<> maxp2;
    TransposedConvLayer<ReLUActivation> tran3;
    ConvLayer<ReLUActivation> conv3s;
    TransposedConvLayer<ReLUActivation> tran4;
    ConvLayer<SigmoidActivation> conv4s;
    L2LossLayer<> loss;
    NetworkHelper<> net;

public:
    SegmentationExample8(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aLabelDepth) : mInputHeight(aInputHeight),
                                                     mInputWidth(aInputWidth),
                                                     mInputDepth(aInputDepth),
                                                     mLabelDepth(aLabelDepth),
                                                     conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, 8, 1},
                                                     conv1s{3, 3, 1, 1, 1, conv1.GetOutputDims(), 16, 1},
                                                     maxp1{conv1s.GetOutputDims(), 2, 2, 1},
                                                     conv2{3, 3, 1, 1, 1, maxp1.GetOutputDims(), 32, 1},
                                                     conv2s{3, 3, 1, 1, 1, conv2.GetOutputDims(), 32, 1},
                                                     maxp2{conv2s.GetOutputDims(), 2, 2, 1},
                                                     tran3{2, 2, 0, 0, 2, maxp2.GetOutputDims(), 32, 1},
                                                     conv3s{3, 3, 1, 1, 1, tran3.GetOutputDims(), 16, 1},
                                                     tran4{2, 2, 0, 0, 2, conv3s.GetOutputDims(), 8, 1},
                                                     conv4s{3, 3, 1, 1, 1, tran4.GetOutputDims(), 2, 1},
                                                     loss{},
                                                     net{{&conv1,
                                                          &conv1s,
                                                          &maxp1,
                                                          &conv2,
                                                          &conv2s,
                                                          &maxp2,
                                                          &tran3,
                                                          &conv3s,
                                                          &tran4,
                                                          &conv4s,
                                                          &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aEpochNum)
    {
        conv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;
        conv1s.mLearningRate = aLearningRate;
        conv2s.mLearningRate = aLearningRate;
        conv3s.mLearningRate = aLearningRate;
        conv4s.mLearningRate = aLearningRate;
        tran4.mLearningRate = aLearningRate;
        tran3.mLearningRate = aLearningRate;

        std::random_device rd;
        std::mt19937 g(rd());
        const size_t vTotalTrainSamples = aInput.cols() / mInputDepth;
        std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
        std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
        for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
        {
            std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
            for (const auto &vIndex : vIndexTrainVector)
            {
                MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, mInputDepth);
                MatrixXd Labels = aLabels.block(0, vIndex * mLabelDepth, mInputHeight * mInputWidth, mLabelDepth);

                conv1.SetData(Input);
                loss.SetData(Labels);
                net.FullForwardPass();
                net.FullBackwardPass();
            }
            std::cout << "Loss of a given sample at epoch: " << vEpoch << std::endl;
            std::cout << loss.GetLoss() << std::endl;
        }
    }

    const MatrixXd Test(MatrixXd aInput)
    {
        std::cout << "testing:" << std::endl;
        conv1.SetData(aInput);
        return net.FullForwardTestPass();
    }
};
