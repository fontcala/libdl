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

class AutoEncoderExample2
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<ReLUActivation> conv1;
    ConvLayer<ReLUActivation> conv2;
    ConvLayer<ReLUActivation> conv3;
    ConvLayer<ReLUActivation> conv4;
    ConvLayer<ReLUActivation> conv5;
    ConvLayer<ReLUActivation> conv6;
    ConvLayer<ReLUActivation> conv7;
    ConvLayer<ReLUActivation> conv8;
    TransposedConvLayer<ReLUActivation> tran8;
    TransposedConvLayer<ReLUActivation> tran7;
    TransposedConvLayer<ReLUActivation> tran6;
    TransposedConvLayer<ReLUActivation> tran5;
    TransposedConvLayer<ReLUActivation> tran4;
    TransposedConvLayer<ReLUActivation> tran3;
    TransposedConvLayer<ReLUActivation> tran2;
    TransposedConvLayer<ReLUActivation> tran1;
    L2LossLayer<> l2;
    NetworkHelper<> net;

public:
    AutoEncoderExample2(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aFilterSize1,
                         const size_t aFilterSize2,
                         const size_t aFilterSize3,
                         const size_t aFilterSize4,
                         const size_t aFilterSize5,
                         const size_t aFilterSize6,
                         const size_t aFilterSize7,
                         const size_t aFilterSize8) : mInputHeight(aInputHeight),
                                                      mInputWidth(aInputWidth),
                                                      conv1{aFilterSize1, aFilterSize1, 0, 0, 1, aInputDepth, aInputHeight, aInputWidth, 2, 1},
                                                      conv2{aFilterSize2, aFilterSize2, 0, 0, 1, conv1.GetOutputDims(), 4, 1},
                                                      conv3{aFilterSize3, aFilterSize3, 0, 0, 1, conv2.GetOutputDims(), 4, 1},
                                                      conv4{aFilterSize4, aFilterSize4, 0, 0, 1, conv3.GetOutputDims(), 8, 1},
                                                      conv5{aFilterSize5, aFilterSize5, 0, 0, 1, conv4.GetOutputDims(), 8, 1},
                                                      conv6{aFilterSize6, aFilterSize6, 0, 0, 1, conv5.GetOutputDims(), 12, 1},
                                                      conv7{aFilterSize7, aFilterSize7, 0, 0, 1, conv6.GetOutputDims(), 12, 1},
                                                      conv8{aFilterSize8, aFilterSize8, 0, 0, 1, conv7.GetOutputDims(), 16, 1},
                                                      tran8{aFilterSize8, aFilterSize8, 0, 0, 1, conv8.GetOutputDims(), conv8.GetInputDims(), 1},
                                                      tran7{aFilterSize7, aFilterSize7, 0, 0, 1, tran8.GetOutputDims(), conv7.GetInputDims(), 1},
                                                      tran6{aFilterSize6, aFilterSize6, 0, 0, 1, tran7.GetOutputDims(), conv6.GetInputDims(), 1},
                                                      tran5{aFilterSize5, aFilterSize5, 0, 0, 1, tran6.GetOutputDims(), conv5.GetInputDims(), 1},
                                                      tran4{aFilterSize4, aFilterSize4, 0, 0, 1, tran5.GetOutputDims(), conv4.GetInputDims(), 1},
                                                      tran3{aFilterSize3, aFilterSize3, 0, 0, 1, tran4.GetOutputDims(), conv3.GetInputDims(), 1},
                                                      tran2{aFilterSize2, aFilterSize2, 0, 0, 1, tran3.GetOutputDims(), conv2.GetInputDims(), 1},
                                                      tran1{aFilterSize1, aFilterSize1, 0, 0, 1, tran2.GetOutputDims(), conv1.GetInputDims(), 1},
                                                      l2{},
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
                                                           &l2}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aNumber)
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
        std::cout << "testing:" << std::endl;
        conv1.SetInput(aInput);
        net.FullForwardPass();
        std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << l2.GetLoss() << std::endl;
        return *(tran1.GetOutput());
    }
};

class AutoEncoderExample
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<SigmoidActivation> conv1;
    MaxPoolLayer<> maxp1;
    ConvLayer<SigmoidActivation> conv2;
    TransposedConvLayer<SigmoidActivation> tran3;
    TransposedConvLayer<SigmoidActivation> trmp1;
    TransposedConvLayer<SigmoidActivation> tran4;
    L2LossLayer<> l2;
    NetworkHelper<> net;

public:
    AutoEncoderExample(const size_t aInputHeight,
                        const size_t aInputWidth,
                        const size_t aInputDepth,
                        const size_t aOutputDepth1,
                        const size_t aOutputDepth2,
                        const size_t aFilterSize1,
                        const size_t aFilterSize2,
                        const size_t aPadding1,
                        const size_t aPadding2,
                        const size_t aStride1,
                        const size_t aStride2) : mInputHeight(aInputHeight),
                                                 mInputWidth(aInputWidth),
                                                 conv1{aFilterSize1, aFilterSize1, aPadding1, aPadding1, aStride1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                 maxp1{conv1.GetOutputDims(), 2, 2, 1},
                                                 conv2{aFilterSize2, aFilterSize2, aPadding2, aPadding2, aStride2, maxp1.GetOutputDims(), aOutputDepth2, 1},
                                                 tran3{aFilterSize2, aFilterSize2, aPadding2, aPadding2, aStride2, conv2.GetOutputDims(), conv2.GetInputDims(), 1},
                                                 trmp1{2, 2, 0, 0, 2, tran3.GetOutputDims(), maxp1.GetInputDims(), 1},
                                                 tran4{aFilterSize1, aFilterSize1, aPadding1, aPadding1, aStride1, trmp1.GetOutputDims(), conv1.GetInputDims(), 1},
                                                 l2{},
                                                 net{{&conv1,
                                                      &maxp1,
                                                      &conv2,
                                                      &tran3,
                                                      &trmp1,
                                                      &tran4,
                                                      &l2}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aNumber)
    {
        conv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;
        tran3.mLearningRate = aLearningRate;
        trmp1.mLearningRate = aLearningRate;
        tran4.mLearningRate = aLearningRate;

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
        std::cout << "testing:" << std::endl;
        conv1.SetInput(aInput);
        net.FullForwardPass();
        std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << l2.GetLoss() << std::endl;
        return *(tran4.GetOutput());
    }
};

class AutoEncoderExample3
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<ReLUActivation> conv1;
    ConvLayer<ReLUActivation> conv2;
    L2LossLayer<> l2;
    NetworkHelper<> net;

public:
    AutoEncoderExample3(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aOutputDepth1) : mInputHeight(aInputHeight),
                                                       mInputWidth(aInputWidth),
                                                       conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                       conv2{3, 3, 1, 1, 1, conv1.GetOutputDims(), 1, 1},
                                                       l2{},
                                                       net{{&conv1,
                                                            &conv2,
                                                            &l2}}
    {
    }
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aNumber)
    {
        conv1.mLearningRate = aLearningRate;
        conv2.mLearningRate = aLearningRate;

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
        std::cout << "testing:" << std::endl;
        conv1.SetInput(aInput);
        net.FullForwardPass();
        std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << l2.GetLoss() << std::endl;
        return *(conv2.GetOutput());
    }
};

class AutoEncoderExample4
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<SigmoidActivation> conv1;
    TransposedConvLayer<SigmoidActivation> tonv2;
    L2LossLayer<> l2;
    NetworkHelper<> net;

public:
    AutoEncoderExample4(const size_t aInputHeight,
                         const size_t aInputWidth,
                         const size_t aInputDepth,
                         const size_t aOutputDepth1) : mInputHeight(aInputHeight),
                                                       mInputWidth(aInputWidth),
                                                       conv1{3, 3, 0, 0, 1, aInputDepth, aInputHeight, aInputWidth, aOutputDepth1, 1},
                                                       tonv2{3, 3, 0, 0, 1, conv1.GetOutputDims(), conv1.GetInputDims(), 1},
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
        l2.SetLabels(aInput); // Not used, just in case we want to do just a full forward pass
        conv1.SetInput(aInput);
        net.FullForwardPass();
        std::cout << "GetLoss(): +++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << l2.GetLoss() << std::endl;
        return *(tonv2.GetOutput());
    }
};


class CNNClassificationExampleModel
{
    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;
    size_t mInputDepth;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mNumCategories;
    double mLearningRate = 0.01;

public:
    void setTrainInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        mInputDepth = aInputDepth;
        mInputHeight = aInputHeight;
        mInputWidth = aInputWidth;
    }
    void setTrainLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        mNumCategories = aNumCategories;
    }
    void setTestInputs(const MatrixXd &aInput)
    {
        mTestInput = aInput;
    }
    void setTestLabels(const MatrixXd &aInput)
    {
        mTestLabels = aInput;
    }
    void setLearningRate(const double aLearningRate)
    {
        mLearningRate = aLearningRate;
    }
    std::vector<size_t> runExample(const size_t aEpochNum);
};
std::vector<size_t> CNNClassificationExampleModel::runExample(const size_t aEpochNum)
{
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 5;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 2;
    const size_t vOutputDepth1 = 6;

    ConvLayer<ReLUActivation> firstConvLayer(vFilterHeight1,
                                             vFilterWidth1,
                                             vPaddingHeight1,
                                             vPaddingWidth1,
                                             vStride1,
                                             mInputDepth,
                                             mInputHeight,
                                             mInputWidth,
                                             vOutputDepth1,
                                             vInputSampleNumber);

    // Conv 2
    const size_t vFilterHeight2 = 3;
    const size_t vFilterWidth2 = 3;
    const size_t vPaddingHeight2 = 1;
    const size_t vPaddingWidth2 = 1;
    const size_t vStride2 = 2;
    const size_t vOutputDepth2 = 8;
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
    FullyConnectedLayer<LinearActivation> fcLayer(flattenLayer.GetOutputDims(), mNumCategories);

    // losslayer
    SoftmaxLossLayer lossLayer{};

    // Connect
    secondConvLayer.SetInput(firstConvLayer.GetOutput());
    flattenLayer.SetInput(secondConvLayer.GetOutput());
    fcLayer.SetInput(flattenLayer.GetOutput());
    lossLayer.SetInput(fcLayer.GetOutput());

    firstConvLayer.SetBackpropInput(secondConvLayer.GetBackpropOutput());
    secondConvLayer.SetBackpropInput(flattenLayer.GetBackpropOutput());
    flattenLayer.SetBackpropInput(fcLayer.GetBackpropOutput());
    fcLayer.SetBackpropInput(lossLayer.GetBackpropOutput());

    // Init Params
    firstConvLayer.mLearningRate = mLearningRate;
    secondConvLayer.mLearningRate = mLearningRate;
    fcLayer.mLearningRate = mLearningRate;

    // TRAIN
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            // std::cout << "Input" << std::endl;
            // std::cout << Input.rows() << " " << Input.cols() << std::endl;
            // std::cout << "Label" << std::endl;
            // std::cout << Label << std::endl;
            //std::cout << "---------start forward ---------" << std::endl;
            firstConvLayer.SetInput(Input);
            lossLayer.SetLabels(Label);
            //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
            firstConvLayer.ForwardPass();
            //std::cout << "secondConvLayer.ForwardPass()  ------" << std::endl;
            secondConvLayer.ForwardPass();
            //std::cout << "flattenLayer.ForwardPass()  ------" << std::endl;
            flattenLayer.ForwardPass();
            //std::cout << "fcLayer.ForwardPass()  ------" << std::endl;
            fcLayer.ForwardPass();
            //std::cout << "lossLayer.ForwardPass()  ------" << std::endl;
            lossLayer.ForwardPass();
            // std::cout << "lossLayer.GetLoss() ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            // std::cout << lossLayer.GetLoss() << std::endl;
            // std::cout << "---------start backward ---------" << std::endl;
            // std::cout << "lossLayer.BackwardPass()  ------" << std::endl;
            lossLayer.BackwardPass();
            //std::cout << "fcLayer.BackwardPass()  ------" << std::endl;
            fcLayer.BackwardPass();
            //std::cout << "flattenLayer.BackwardPass()  ------" << std::endl;
            flattenLayer.BackwardPass();
            //std::cout << "secondConv.BackwardPass()  ------" << std::endl;
            secondConvLayer.BackwardPass();
            //std::cout << "firstConvLayer.BackwardPass()  ------" << std::endl;
            firstConvLayer.BackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "lossLayer.GetLoss() of any given sample ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            std::cout << lossLayer.GetLoss() << std::endl;
            std::cout << "Epoch " << vEpoch << std::endl;
        }
    }

    //TEST
    std::vector<size_t> vTestResults;
    const size_t vTotalTestSamples = mTestInput.cols();
    size_t vNumCorrectlyClassified = 0;
    std::vector<size_t> vIndexTestVector(vTotalTestSamples);
    std::iota(std::begin(vIndexTestVector), std::end(vIndexTestVector), 0); // Fill with 0, 1, ..., N.
    for (const auto &vIndex : vIndexTestVector)
    {
        MatrixXd Input = mTestInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
        MatrixXd Label = mTestLabels.block(vIndex, 0, 1, mNumCategories);
        // std::cout << "Input" << std::endl;
        // std::cout << Input.rows() << " " << Input.cols() << std::endl;
        // std::cout << "Label" << std::endl;
        // std::cout << Label << std::endl;
        //std::cout << "---------start forward ---------" << std::endl;
        firstConvLayer.SetInput(Input);
        lossLayer.SetLabels(Label);
        //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
        firstConvLayer.ForwardPass();
        //std::cout << "secondConvLayer.ForwardPass()  ------" << std::endl;
        secondConvLayer.ForwardPass();
        //std::cout << "flattenLayer.ForwardPass()  ------" << std::endl;
        flattenLayer.ForwardPass();
        //std::cout << "fcLayer.ForwardPass()  ------" << std::endl;
        fcLayer.ForwardPass();
        //std::cout << "lossLayer.ForwardPass()  ------" << std::endl;
        lossLayer.ForwardPass();

        // Compare with labels
        MatrixXd vScores = *(lossLayer.GetOutput());
        MatrixXd::Index maxColScores, maxRowScores;
        const double maxScores = vScores.maxCoeff(&maxRowScores, &maxColScores);
        MatrixXd::Index maxColLabel, maxRowLabel;
        vTestResults.push_back(maxColScores);
        const double maxLabels = Label.maxCoeff(&maxRowLabel, &maxColLabel);
        if (maxColScores == maxColLabel)
        {
            ++vNumCorrectlyClassified;
        }
        // std::cout << "Cat scores" << std::endl;
        // std::cout << maxColScores << std::endl;
        // std::cout << "Cat label" << std::endl;
        // std::cout << maxColLabel << std::endl;
    }
    const double vAccuracy = static_cast<double>(vNumCorrectlyClassified) / static_cast<double>(vTotalTestSamples);
    std::cout << "test Accuracy is " << vAccuracy << std::endl;
    return vTestResults;
}