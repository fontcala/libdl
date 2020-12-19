#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/NetworkHelper.h>
#include <libdl/ConvLayer.h>
#include <libdl/ConvExperimentalLayer.h>
#include <libdl/ConvFA.h>
#include <libdl/ConvSFA.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/L2LossLayer.h>
#include <libdl/FullyConnectedLayer.h>
#include <libdl/FullyConnectedExperimentalLayer.h>
#include <libdl/FullyConnectedFA.h>
#include <libdl/FullyConnectedSFA.h>
using Eigen::MatrixXd;

template <class convType, class fcType>
class CNNClassificationAlignmentTest
{
    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;
    size_t mInputDepth;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mNumCategories;
    double mLearningRate;

    size_t mFilterSize1;
    size_t mFilterSize2;
    size_t mPadding1;
    size_t mPadding2;
    size_t mFilterDepth1;
    size_t mFilterDepth2;

public:
    CNNClassificationAlignmentTest(const int aFilterSize1, const int aFilterSize2, const int aPadding1, const int aPadding2, const int aDepth1, const int aDepth2) : mFilterSize1(aFilterSize1), mFilterSize2(aFilterSize2), mPadding1(aPadding1), mPadding2(aPadding2), mFilterDepth1(aDepth1), mFilterDepth2(aDepth2){};
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
template <class convType, class fcType>
std::vector<size_t> CNNClassificationAlignmentTest<convType, fcType>::runExample(const size_t aEpochNum)
{
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = mFilterSize1; //5
    const size_t vFilterWidth1 = mFilterSize1;
    const size_t vPaddingHeight1 = mPadding1;
    const size_t vPaddingWidth1 = mPadding1;
    const size_t vStride1 = 1;
    const size_t vOutputDepth1 = mFilterDepth1; //6

    convType firstConvLayer(vFilterHeight1,
                            vFilterWidth1,
                            vPaddingHeight1,
                            vPaddingWidth1,
                            vStride1,
                            mInputDepth,
                            mInputHeight,
                            mInputWidth,
                            vOutputDepth1,
                            vInputSampleNumber, UpdateMethod::ADAM);
    // MaxPool
    const size_t vPoolSize = 2;
    const size_t vStridePool = 2;
    MaxPoolLayer maxp1(firstConvLayer.GetOutputDims(),
                       vPoolSize,
                       vStridePool,
                       vInputSampleNumber);
    // Conv 2
    const size_t vFilterHeight2 = mFilterSize2;
    const size_t vFilterWidth2 = mFilterSize2;
    const size_t vPaddingHeight2 = mPadding2;
    const size_t vPaddingWidth2 = mPadding2;
    const size_t vStride2 = 1;
    const size_t vOutputDepth2 = mFilterDepth2;
    convType secondConvLayer(vFilterHeight2,
                             vFilterWidth2,
                             vPaddingHeight2,
                             vPaddingWidth2,
                             vStride2,
                             maxp1.GetOutputDims(),
                             vOutputDepth2,
                             vInputSampleNumber, UpdateMethod::ADAM);

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    fcType fcLayer(flattenLayer.GetOutputDims(), mNumCategories, UpdateMethod::ADAM);

    // losslayer
    SoftmaxLossLayer lossLayer{};

    // Connect
    maxp1.SetInput(firstConvLayer.GetOutput());
    secondConvLayer.SetInput(maxp1.GetOutput());
    flattenLayer.SetInput(secondConvLayer.GetOutput());
    fcLayer.SetInput(flattenLayer.GetOutput());
    lossLayer.SetInput(fcLayer.GetOutput());

    firstConvLayer.SetBackpropInput(maxp1.GetBackpropOutput());
    maxp1.SetBackpropInput(secondConvLayer.GetBackpropOutput());
    secondConvLayer.SetBackpropInput(flattenLayer.GetBackpropOutput());
    flattenLayer.SetBackpropInput(fcLayer.GetBackpropOutput());
    fcLayer.SetBackpropInput(lossLayer.GetBackpropOutput());

    // Init Params
    firstConvLayer.SetLearningRate(mLearningRate);
    secondConvLayer.SetLearningRate(mLearningRate);
    fcLayer.SetLearningRate(mLearningRate);

    // TRAIN
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);

            // forward
            firstConvLayer.ForwardPass();
            maxp1.ForwardPass();
            secondConvLayer.ForwardPass();
            flattenLayer.ForwardPass();
            fcLayer.ForwardPass();
            lossLayer.ForwardPass();

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + lossLayer.GetLoss();

            //backward
            lossLayer.BackwardPass();
            fcLayer.BackwardPass();
            flattenLayer.BackwardPass();
            secondConvLayer.BackwardPass();
            maxp1.BackwardPass();
            firstConvLayer.BackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                      << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
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
        firstConvLayer.SetData(Input);
        lossLayer.SetData(Label);

        firstConvLayer.ForwardPass();
        maxp1.ForwardPass();
        secondConvLayer.ForwardPass();
        flattenLayer.ForwardPass();
        fcLayer.ForwardPass();
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
    }
    const double vAccuracy = static_cast<double>(vNumCorrectlyClassified) / static_cast<double>(vTotalTestSamples);
    std::cout << "test Accuracy is " << vAccuracy << std::endl;
    return vTestResults;
}

template <class convType, class fcType>
class CNNClassificationAltTest
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

    size_t mFilterSize1;
    size_t mFilterSize2;
    size_t mPadding1;
    size_t mPadding2;
    size_t mFilterDepth1;
    size_t mFilterDepth2;

public:
    CNNClassificationAltTest(const int aFilterSize1, const int aFilterSize2, const int aPadding1, const int aPadding2, const int aDepth1, const int aDepth2) : mFilterSize1(aFilterSize1), mFilterSize2(aFilterSize2), mPadding1(aPadding1), mPadding2(aPadding2), mFilterDepth1(aDepth1), mFilterDepth2(aDepth2){};
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
template <class convType, class fcType>
std::vector<size_t> CNNClassificationAltTest<convType, fcType>::runExample(const size_t aEpochNum)
{
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = mFilterSize1; //5
    const size_t vFilterWidth1 = mFilterSize1;
    const size_t vPaddingHeight1 = mPadding1;
    const size_t vPaddingWidth1 = mPadding1;
    const size_t vStride1 = 1;
    const size_t vOutputDepth1 = mFilterDepth1; //6

    convType firstConvLayer(vFilterHeight1,
                            vFilterWidth1,
                            vPaddingHeight1,
                            vPaddingWidth1,
                            vStride1,
                            mInputDepth,
                            mInputHeight,
                            mInputWidth,
                            vOutputDepth1,
                            vInputSampleNumber, UpdateMethod::ADAM);

    convType firstConvLayerAlt(vFilterHeight1,
                               vFilterWidth1,
                               vPaddingHeight1,
                               vPaddingWidth1,
                               vStride1,
                               mInputDepth,
                               mInputHeight,
                               mInputWidth,
                               vOutputDepth1,
                               vInputSampleNumber, UpdateMethod::ADAM);
    // MaxPool
    const size_t vPoolSize = 2;
    const size_t vStridePool = 2;
    MaxPoolLayer maxp1(firstConvLayer.GetOutputDims(),
                       vPoolSize,
                       vStridePool,
                       vInputSampleNumber);

    MaxPoolLayer maxp1Alt(firstConvLayer.GetOutputDims(),
                          vPoolSize,
                          vStridePool,
                          vInputSampleNumber);
    // Conv 2
    const size_t vFilterHeight2 = mFilterSize2;
    const size_t vFilterWidth2 = mFilterSize2;
    const size_t vPaddingHeight2 = mPadding2;
    const size_t vPaddingWidth2 = mPadding2;
    const size_t vStride2 = 1;
    const size_t vOutputDepth2 = mFilterDepth2;
    convType secondConvLayer(vFilterHeight2,
                             vFilterWidth2,
                             vPaddingHeight2,
                             vPaddingWidth2,
                             vStride2,
                             maxp1.GetOutputDims(),
                             vOutputDepth2,
                             vInputSampleNumber, UpdateMethod::ADAM);

    convType secondConvLayerAlt(vFilterHeight2,
                                vFilterWidth2,
                                vPaddingHeight2,
                                vPaddingWidth2,
                                vStride2,
                                maxp1.GetOutputDims(),
                                vOutputDepth2,
                                vInputSampleNumber, UpdateMethod::ADAM);

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    FlattenLayer flattenLayerAlt(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    fcType fcLayer(flattenLayer.GetOutputDims(), mNumCategories, UpdateMethod::ADAM);

    fcType fcLayerAlt(flattenLayer.GetOutputDims(), mNumCategories, UpdateMethod::ADAM);

    // losslayer
    SoftmaxLossLayer lossLayer{};

    SoftmaxLossLayer lossLayerAlt{};

    // Connect
    maxp1.SetInput(firstConvLayer.GetOutput());
    secondConvLayer.SetInput(maxp1.GetOutput());
    flattenLayer.SetInput(secondConvLayer.GetOutput());
    fcLayer.SetInput(flattenLayer.GetOutput());
    lossLayer.SetInput(fcLayer.GetOutput());

    maxp1Alt.SetInput(firstConvLayerAlt.GetOutput());
    secondConvLayerAlt.SetInput(maxp1Alt.GetOutput());
    flattenLayerAlt.SetInput(secondConvLayerAlt.GetOutput());
    fcLayerAlt.SetInput(flattenLayerAlt.GetOutput());
    lossLayerAlt.SetInput(fcLayerAlt.GetOutput());

    firstConvLayer.SetBackpropInput(maxp1.GetBackpropOutput());
    maxp1.SetBackpropInput(secondConvLayer.GetBackpropOutput());
    secondConvLayer.SetBackpropInput(flattenLayer.GetBackpropOutput());
    flattenLayer.SetBackpropInput(fcLayer.GetBackpropOutput());
    fcLayer.SetBackpropInput(lossLayer.GetBackpropOutput());

    firstConvLayerAlt.SetBackpropInput(maxp1Alt.GetBackpropOutput());
    maxp1Alt.SetBackpropInput(secondConvLayerAlt.GetBackpropOutput());
    secondConvLayerAlt.SetBackpropInput(flattenLayerAlt.GetBackpropOutput());
    flattenLayerAlt.SetBackpropInput(fcLayerAlt.GetBackpropOutput());
    fcLayerAlt.SetBackpropInput(lossLayerAlt.GetBackpropOutput());

    // Init Params
    firstConvLayer.SetLearningRate(mLearningRate);
    secondConvLayer.SetLearningRate(mLearningRate);
    fcLayer.SetLearningRate(mLearningRate);

    firstConvLayerAlt.SetLearningRate(mLearningRate);
    secondConvLayerAlt.SetLearningRate(mLearningRate);
    fcLayerAlt.SetLearningRate(mLearningRate);

    // TRAIN
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    double currentLossAggregatorAlt;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        currentLossAggregatorAlt = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);

            firstConvLayerAlt.SetData(Input);
            lossLayerAlt.SetData(Label);

            // forward
            firstConvLayer.ForwardPass();
            maxp1.ForwardPass();
            secondConvLayer.ForwardPass();
            flattenLayer.ForwardPass();
            fcLayer.ForwardPass();
            lossLayer.ForwardPass();
            // forward ALT
            firstConvLayerAlt.ForwardPass();
            maxp1Alt.ForwardPass();
            secondConvLayerAlt.ForwardPass();
            flattenLayerAlt.ForwardPass();
            fcLayerAlt.ForwardPass();
            lossLayerAlt.ForwardPass();

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + lossLayer.GetLoss();
            currentLossAggregatorAlt = currentLossAggregatorAlt + lossLayerAlt.GetLoss();

            //backward
            lossLayer.BackwardPass();
            lossLayerAlt.BackwardPass();
            fcLayer.BackwardPass();
            fcLayerAlt.BackwardPass();
            flattenLayer.BackwardPass();
            flattenLayerAlt.BackwardPass();
            secondConvLayer.BackwardPass();
            secondConvLayerAlt.BackwardPass();
            maxp1.BackwardPass();
            maxp1Alt.BackwardPass();
            firstConvLayer.BackwardPass();
            firstConvLayerAlt.BackwardPass();
        }
        if (vEpoch == 10)
        {
            firstConvLayer.SetBackpropInput(maxp1Alt.GetBackpropOutput());
            maxp1.SetBackpropInput(secondConvLayerAlt.GetBackpropOutput());
            secondConvLayer.SetBackpropInput(flattenLayerAlt.GetBackpropOutput());
            flattenLayer.SetBackpropInput(fcLayerAlt.GetBackpropOutput());
            fcLayer.SetBackpropInput(lossLayerAlt.GetBackpropOutput());

            firstConvLayerAlt.SetBackpropInput(maxp1.GetBackpropOutput());
            maxp1Alt.SetBackpropInput(secondConvLayer.GetBackpropOutput());
            secondConvLayerAlt.SetBackpropInput(flattenLayer.GetBackpropOutput());
            flattenLayerAlt.SetBackpropInput(fcLayer.GetBackpropOutput());
            fcLayerAlt.SetBackpropInput(lossLayer.GetBackpropOutput());
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                      << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
            std::cout << "trainingLoss[" << vEpoch << "]"
                      << "=" << currentLossAggregatorAlt / static_cast<double>(vTotalTrainSamples) << std::endl;
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
        firstConvLayer.SetData(Input);
        lossLayer.SetData(Label);

        firstConvLayer.ForwardPass();
        maxp1.ForwardPass();
        secondConvLayer.ForwardPass();
        flattenLayer.ForwardPass();
        fcLayer.ForwardPass();
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
    }
    const double vAccuracy = static_cast<double>(vNumCorrectlyClassified) / static_cast<double>(vTotalTestSamples);
    std::cout << "test Accuracy is " << vAccuracy << std::endl;
    return vTestResults;
}

template <class convType, class transType, class fcType>
class AutoencoderExp
{
public:

    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;
    convType conv1;
    convType conv1s;
    MaxPoolLayer<> maxp1;
    convType conv2;
    convType conv2s;
    MaxPoolLayer<> maxp2;
    convType convBottleneck1;
    convType convBottleneck2;
    transType tran3;
    convType conv3s;
    transType tran4;
    ConvLayer<SigmoidActivation> conv4s;
    L2LossLayer<> loss;

    AutoencoderExp(const size_t aInputHeight,
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
                                                     convBottleneck1{3, 3, 1, 1, 1, maxp2.GetOutputDims(), 64, 1},
                                                     convBottleneck2{3, 3, 1, 1, 1, convBottleneck1.GetOutputDims(), 64, 1},
                                                     tran3{2, 2, 0, 0, 2, convBottleneck2.GetOutputDims(), 32, 1},
                                                     conv3s{3, 3, 1, 1, 1, tran3.GetOutputDims(), 16, 1},
                                                     tran4{2, 2, 0, 0, 2, conv3s.GetOutputDims(), 8, 1},
                                                     conv4s{3, 3, 1, 1, 1, tran4.GetOutputDims(), 2, 1},
                                                     loss {}{};
void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aEpochNum)
{
    conv1.SetLearningRate(aLearningRate);
    conv2.SetLearningRate(aLearningRate);
    conv1s.SetLearningRate(aLearningRate);
    conv2s.SetLearningRate(aLearningRate);
    conv3s.SetLearningRate(aLearningRate);
    convBottleneck1.SetLearningRate(aLearningRate);
    convBottleneck2.SetLearningRate(aLearningRate);
    conv4s.SetLearningRate(aLearningRate);
    tran4.SetLearningRate(aLearningRate);
    tran3.SetLearningRate(aLearningRate);

    conv1s.SetInput(conv1.GetOutput());
    maxp1.SetInput(conv1s.GetOutput());
    conv2.SetInput(maxp1.GetOutput());
    conv2s.SetInput(conv2.GetOutput());
    maxp2.SetInput(conv2s.GetOutput());
    convBottleneck1.SetInput(maxp2.GetOutput());
    convBottleneck2.SetInput(convBottleneck1.GetOutput());
    tran3.SetInput(convBottleneck2.GetOutput());
    conv3s.SetInput(tran3.GetOutput());
    tran4.SetInput(conv3s.GetOutput());
    conv4s.SetInput(tran4.GetOutput());
    loss.SetInput(conv4s.GetOutput());

    conv1.SetBackpropInput(conv1s.GetBackpropOutput());
    conv1s.SetBackpropInput(maxp1.GetBackpropOutput());
    maxp1.SetBackpropInput(conv2.GetBackpropOutput());
    conv2.SetBackpropInput(conv2s.GetBackpropOutput());
    conv2s.SetBackpropInput(maxp2.GetBackpropOutput());
    maxp2.SetBackpropInput(convBottleneck1.GetBackpropOutput());
    convBottleneck1.SetBackpropInput(convBottleneck2.GetBackpropOutput());
    convBottleneck2.SetBackpropInput(tran3.GetBackpropOutput());
    tran3.SetBackpropInput(conv3s.GetBackpropOutput());
    conv3s.SetBackpropInput(tran4.GetBackpropOutput());
    tran4.SetBackpropInput(conv4s.GetBackpropOutput());
    conv4s.SetBackpropInput(loss.GetBackpropOutput());

    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = aInput.cols() / mInputDepth;
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, mInputDepth);
            MatrixXd Labels = aLabels.block(0, vIndex * mLabelDepth, mInputHeight * mInputWidth, mLabelDepth);
            conv1.SetData(Input);
            loss.SetData(Labels);

            // forward
            conv1.ForwardPass();
            conv1s.ForwardPass();
            maxp1.ForwardPass();
            conv2.ForwardPass();
            conv2s.ForwardPass();
            maxp2.ForwardPass();
            convBottleneck1.ForwardPass();
            convBottleneck2.ForwardPass();
            tran3.ForwardPass();
            conv3s.ForwardPass();
            tran4.ForwardPass();
            conv4s.ForwardPass();
            loss.ForwardPass();     

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + loss.GetLoss();

            //backward
            loss.BackwardPass();
            conv4s.BackwardPass();
            tran4.BackwardPass();
            conv3s.BackwardPass();
            tran3.BackwardPass();
            convBottleneck2.BackwardPass();
            convBottleneck1.BackwardPass();
            maxp2.BackwardPass();
            conv2s.BackwardPass();
            conv2.BackwardPass();
            maxp1.BackwardPass();
            conv1s.BackwardPass();
            conv1.BackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                      << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
        }
    }
}

const MatrixXd Test(MatrixXd aInput)
{
    std::cout << "testing:" << std::endl;
    conv1.SetData(aInput);
    // forward
    conv1.ForwardPass();
    conv1s.ForwardPass();
    maxp1.ForwardPass();
    conv2.ForwardPass();
    conv2s.ForwardPass();
    maxp2.ForwardPass();
    convBottleneck1.ForwardPass();
    convBottleneck2.ForwardPass();
    tran3.ForwardPass();
    conv3s.ForwardPass();
    tran4.ForwardPass();
    conv4s.ForwardPass();
    
    return *(conv4s.GetOutput());
}
};

template <class convType, class transType, class fcType>
class AutoencoderExpSkip
{
public:

    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;
    convType conv1;
    convType conv1s;
    MaxPoolLayer<> maxp1;
    convType conv2;
    convType conv2s;
    MaxPoolLayer<> maxp2;
    convType convBottleneck1;
    convType convBottleneck2;
    transType tran3;
    convType conv3s;
    transType tran4;
    ConvLayer<SigmoidActivation> conv4s;
    L2LossLayer<> loss;

    AutoencoderExpSkip(const size_t aInputHeight,
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
                                                     convBottleneck1{3, 3, 1, 1, 1, maxp2.GetOutputDims(), 64, 1},
                                                     convBottleneck2{3, 3, 1, 1, 1, convBottleneck1.GetOutputDims(), 64, 1},
                                                     tran3{2, 2, 0, 0, 2, convBottleneck2.GetOutputDims(), 32, 1},
                                                     conv3s{3, 3, 1, 1, 1, tran3.GetOutputDims(), 16, 1},
                                                     tran4{2, 2, 0, 0, 2, conv3s.GetOutputDims(), 8, 1},
                                                     conv4s{3, 3, 1, 1, 1, tran4.GetOutputDims(), 2, 1},
                                                     loss {}{};
void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const size_t aEpochNum)
{
    conv1.SetLearningRate(aLearningRate);
    conv2.SetLearningRate(aLearningRate);
    conv1s.SetLearningRate(aLearningRate);
    conv2s.SetLearningRate(aLearningRate);
    conv3s.SetLearningRate(aLearningRate);
    convBottleneck1.SetLearningRate(aLearningRate);
    convBottleneck2.SetLearningRate(aLearningRate);
    conv4s.SetLearningRate(aLearningRate);
    tran4.SetLearningRate(aLearningRate);
    tran3.SetLearningRate(aLearningRate);

    conv1s.SetInput(conv1.GetOutput());
    maxp1.SetInput(conv1s.GetOutput());
    conv2.SetInput(maxp1.GetOutput());
    conv2s.SetInput(conv2.GetOutput());
    maxp2.SetInput(conv2s.GetOutput());
    convBottleneck1.SetInput(maxp2.GetOutput());
    convBottleneck2.SetInput(convBottleneck1.GetOutput());
    tran3.SetInput(convBottleneck2.GetOutput());
    conv3s.SetInput(tran3.GetOutput());
    tran4.SetInput(conv3s.GetOutput());
    conv4s.SetInput(tran4.GetOutput());
    loss.SetInput(conv4s.GetOutput());

    conv1.SetBackpropInput(conv4s.GetBackpropOutput());
    conv1s.SetBackpropInput(maxp1.GetBackpropOutput());
    maxp1.SetBackpropInput(conv2.GetBackpropOutput());
    conv2.SetBackpropInput(conv3s.GetBackpropOutput());
    conv2s.SetBackpropInput(maxp2.GetBackpropOutput());
    maxp2.SetBackpropInput(convBottleneck1.GetBackpropOutput());
    convBottleneck1.SetBackpropInput(tran3.GetBackpropOutput());
    convBottleneck2.SetBackpropInput(tran3.GetBackpropOutput());
    tran3.SetBackpropInput(conv3s.GetBackpropOutput());
    conv3s.SetBackpropInput(tran4.GetBackpropOutput());
    tran4.SetBackpropInput(conv4s.GetBackpropOutput());
    conv4s.SetBackpropInput(loss.GetBackpropOutput());

    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = aInput.cols() / mInputDepth;
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, mInputDepth);
            MatrixXd Labels = aLabels.block(0, vIndex * mLabelDepth, mInputHeight * mInputWidth, mLabelDepth);
            conv1.SetData(Input);
            loss.SetData(Labels);

            // forward
            conv1.ForwardPass();
            conv1s.ForwardPass();
            maxp1.ForwardPass();
            conv2.ForwardPass();
            conv2s.ForwardPass();
            maxp2.ForwardPass();
            convBottleneck1.ForwardPass();
            convBottleneck2.ForwardPass();
            tran3.ForwardPass();
            conv3s.ForwardPass();
            tran4.ForwardPass();
            conv4s.ForwardPass();
            loss.ForwardPass();     

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + loss.GetLoss();

            //backward
            loss.BackwardPass();
            conv4s.BackwardPass();
            tran4.BackwardPass();
            conv3s.BackwardPass();
            tran3.BackwardPass();
            convBottleneck2.BackwardPass();
            convBottleneck1.BackwardPass();
            maxp2.BackwardPass();
            conv2s.BackwardPass();
            conv2.BackwardPass();
            maxp1.BackwardPass();
            conv1s.BackwardPass();
            conv1.BackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                      << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
        }
    }
}

const MatrixXd Test(MatrixXd aInput)
{
    std::cout << "testing:" << std::endl;
    conv1.SetData(aInput);
    // forward
    conv1.ForwardPass();
    conv1s.ForwardPass();
    maxp1.ForwardPass();
    conv2.ForwardPass();
    conv2s.ForwardPass();
    maxp2.ForwardPass();
    convBottleneck1.ForwardPass();
    convBottleneck2.ForwardPass();
    tran3.ForwardPass();
    conv3s.ForwardPass();
    tran4.ForwardPass();
    conv4s.ForwardPass();
    
    return *(conv4s.GetOutput());
}
};