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

template<class convType, class fcType>
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
    CNNClassificationAlignmentTest(const int aFilterSize1, const int aFilterSize2, const int aPadding1, const int aPadding2,const int aDepth1, const int aDepth2):mFilterSize1(aFilterSize1),mFilterSize2(aFilterSize2),mPadding1(aPadding1),mPadding2(aPadding2),mFilterDepth1(aDepth1),mFilterDepth2(aDepth2){};
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
template<class convType, class fcType>
std::vector<size_t> CNNClassificationAlignmentTest<convType,fcType>::runExample(const size_t aEpochNum)
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
                                             vInputSampleNumber,UpdateMethod::ADAM);
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
                                              vInputSampleNumber,UpdateMethod::ADAM);

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    fcType fcLayer(flattenLayer.GetOutputDims(), mNumCategories,UpdateMethod::ADAM);

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
            std::cout << "trainingLoss[" <<  vEpoch << "]" << "=" << currentLossAggregator/static_cast<double>(vTotalTrainSamples) << std::endl;
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

template<class convType, class fcType>
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
    CNNClassificationAltTest(const int aFilterSize1, const int aFilterSize2, const int aPadding1, const int aPadding2,const int aDepth1, const int aDepth2):mFilterSize1(aFilterSize1),mFilterSize2(aFilterSize2),mPadding1(aPadding1),mPadding2(aPadding2),mFilterDepth1(aDepth1),mFilterDepth2(aDepth2){};
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
template<class convType, class fcType>
std::vector<size_t> CNNClassificationAltTest<convType,fcType>::runExample(const size_t aEpochNum)
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
                                             vInputSampleNumber,UpdateMethod::ADAM);

    convType firstConvLayerAlt(vFilterHeight1,
                                             vFilterWidth1,
                                             vPaddingHeight1,
                                             vPaddingWidth1,
                                             vStride1,
                                             mInputDepth,
                                             mInputHeight,
                                             mInputWidth,
                                             vOutputDepth1,
                                             vInputSampleNumber,UpdateMethod::ADAM);
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
                                              vInputSampleNumber,UpdateMethod::ADAM);

    convType secondConvLayerAlt(vFilterHeight2,
                                              vFilterWidth2,
                                              vPaddingHeight2,
                                              vPaddingWidth2,
                                              vStride2,
                                              maxp1.GetOutputDims(),
                                              vOutputDepth2,
                                              vInputSampleNumber,UpdateMethod::ADAM);

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    FlattenLayer flattenLayerAlt(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    fcType fcLayer(flattenLayer.GetOutputDims(), mNumCategories,UpdateMethod::ADAM);

    fcType fcLayerAlt(flattenLayer.GetOutputDims(), mNumCategories,UpdateMethod::ADAM);

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
            std::cout << "trainingLoss[" <<  vEpoch << "]" << "=" << currentLossAggregator/static_cast<double>(vTotalTrainSamples) << std::endl;
            std::cout << "trainingLoss[" <<  vEpoch << "]" << "=" << currentLossAggregatorAlt/static_cast<double>(vTotalTrainSamples) << std::endl;
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