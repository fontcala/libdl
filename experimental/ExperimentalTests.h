#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/NetworkHelper.h>
#include <libdl/ConvLayer.h>
#include <libdl/ConvAMMLayer.h>
#include <libdl/ConvExperimentalLayer.h>
#include <libdl/ConvFA.h>
#include <libdl/ConvSFA.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/L2LossLayer.h>
#include <libdl/FullyConnectedLayer.h>
#include <libdl/FullyConnectedAMMLayer.h>
#include <libdl/FullyConnectedExperimentalLayer.h>
#include <libdl/FullyConnectedFA.h>
#include <libdl/FullyConnectedDFA.h>
#include <libdl/FullyConnectedSFA.h>

#include <chrono>

using Eigen::MatrixXd;


template <class convType, class fcType>
class CNNClassificationExperimentalTest
{
    // Network will have stride = 1 (conv layers) and batch size = 1
    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;

    const size_t mNumCategories;

    double mLearningRate;

    convType firstConvLayer;
    MaxPoolLayer<> maxp1;
    convType secondConvLayer;
    FlattenLayer<> flattenLayer;
    fcType fcLayer;
    SoftmaxLossLayer<> lossLayer;

    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;

    NetworkHelper<> net;

public:
    CNNClassificationExperimentalTest(
    const size_t aFilterSize1,
    const size_t aFilterSize2,
    const size_t aPadding1,
    const size_t aPadding2,
    const size_t aDepth1,
    const size_t aDepth2,
    const size_t aInputHeight,
    const size_t aInputWidth,
    const size_t aInputDepth,
    const size_t aLabelDepth,
    const size_t aNumCategories) :
    mInputHeight(aInputHeight),
    mInputWidth(aInputWidth),
    mInputDepth(aInputDepth),
    mLabelDepth(aLabelDepth),
    mNumCategories(aNumCategories),
    firstConvLayer{aFilterSize1, aFilterSize1, aPadding1, aPadding1, 1, aInputDepth, aInputHeight, aInputWidth, aDepth1, 1, UpdateMethod::ADAM},
    maxp1{firstConvLayer.GetOutputDims(), 2, 2, 1},
    secondConvLayer{aFilterSize2, aFilterSize2, aPadding2, aPadding2, 1, maxp1.GetOutputDims(), aDepth2, 1, UpdateMethod::ADAM},
    flattenLayer(secondConvLayer.GetOutputDims(), 1),
    fcLayer(flattenLayer.GetOutputDims(), mNumCategories, UpdateMethod::ADAM),
    lossLayer{},
    net{
    &firstConvLayer,
    &maxp1,
    &secondConvLayer,
    &flattenLayer,
    &fcLayer,
    &lossLayer
    }
    {
    };


    void setTrainInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
    }
    void setTrainLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
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

        firstConvLayer.SetLearningRate(mLearningRate);
        secondConvLayer.SetLearningRate(mLearningRate);
        fcLayer.SetLearningRate(mLearningRate);
    }

    void train(const size_t aEpochNum);

    std::vector<size_t> test();


};
template <class convType, class fcType>
void CNNClassificationExperimentalTest<convType, fcType>::train(const size_t aEpochNum)
{
    // TRAIN
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::cout << "start new epoch" << std::endl;
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);

            // forward
            net.FullForwardPass();

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + lossLayer.GetLoss();

            // backward
            net.FullBackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                        << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time Elapsed = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    

}

template <class convType, class fcType>
std::vector<size_t> CNNClassificationExperimentalTest<convType, fcType>::test()
{
    // TEST
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

        net.FullForwardPass();

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

template <class convType, class fcType, class finalLayer>
class CNNClassificationExperimentalDFATest
{
    // Network will have stride = 1 (conv layers) and batch size = 1
    const size_t mInputHeight;
    const size_t mInputWidth;
    const size_t mInputDepth;
    const size_t mLabelDepth;

    const size_t mNumCategories;

    double mLearningRate;

    convType firstConvLayer;
    MaxPoolLayer<> maxp1;
    convType secondConvLayer;
    FlattenLayer<> flattenLayer;
    fcType fcLayer;
    finalLayer fcLayerFinal;
    SoftmaxLossLayer<> lossLayer;

    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;

    NetworkHelper<> net;

public:
    CNNClassificationExperimentalDFATest(
    const size_t aFilterSize1,
    const size_t aFilterSize2,
    const size_t aPadding1,
    const size_t aPadding2,
    const size_t aDepth1,
    const size_t aDepth2,
    const size_t aInputHeight,
    const size_t aInputWidth,
    const size_t aInputDepth,
    const size_t aLabelDepth,
    const size_t aNumCategories) :
    mInputHeight(aInputHeight),
    mInputWidth(aInputWidth),
    mInputDepth(aInputDepth),
    mLabelDepth(aLabelDepth),
    mNumCategories(aNumCategories),
    firstConvLayer{aFilterSize1, aFilterSize1, aPadding1, aPadding1, 1, aInputDepth, aInputHeight, aInputWidth, aDepth1, 1, UpdateMethod::ADAM},
    maxp1{firstConvLayer.GetOutputDims(), 2, 2, 1},
    secondConvLayer{aFilterSize2, aFilterSize2, aPadding2, aPadding2, 1, maxp1.GetOutputDims(), aDepth2, 1, UpdateMethod::ADAM},
    flattenLayer(secondConvLayer.GetOutputDims(), 1),
    fcLayer(flattenLayer.GetOutputDims(), mNumCategories*2, mNumCategories, UpdateMethod::ADAM),
    fcLayerFinal(fcLayer.GetOutputDims(), mNumCategories, mNumCategories, UpdateMethod::ADAM),
    lossLayer{},
    net{
    &firstConvLayer,
    &maxp1,
    &secondConvLayer,
    &flattenLayer,
    &fcLayer,
    &fcLayerFinal,
    &lossLayer
    }
    {
    };


    void setTrainInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
    }
    void setTrainLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
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

        firstConvLayer.SetLearningRate(mLearningRate);
        secondConvLayer.SetLearningRate(mLearningRate);
        fcLayer.SetLearningRate(mLearningRate);
        fcLayerFinal.SetLearningRate(mLearningRate);
    }

    void train(const size_t aEpochNum);

    std::vector<size_t> test();


};
template <class convType, class fcType, class finalLayer>
void CNNClassificationExperimentalDFATest<convType, fcType, finalLayer>::train(const size_t aEpochNum)
{
    // TRAIN
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::cout << "start new epoch" << std::endl;
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);

            // forward
            net.FullForwardPass();

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + lossLayer.GetLoss();

            // backward
            net.FullBackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                        << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time Elapsed = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    

}

template <class convType, class fcType, class finalLayer>
std::vector<size_t> CNNClassificationExperimentalDFATest<convType, fcType, finalLayer>::test()
{
    // TEST
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

        net.FullForwardPass();

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

template <class fcType, class finalLayer>
class MLPTest
{
    // Network will have stride = 1 (conv layers) and batch size = 1
    const size_t mInputHeight;
    const size_t mInputWidth;

    const size_t mNumCategories;

    double mLearningRate;

    fcType fcLayer1;
    fcType fcLayer2;
    fcType fcLayer3;
    finalLayer fcLayerFinal;
    SoftmaxLossLayer<> lossLayer;

    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;

    NetworkHelper<> net;

public:
    MLPTest(
    const size_t aHiddenSize1,
    const size_t aHiddenSize2,
    const size_t aHiddenSize3,
    const size_t aInputHeight,
    const size_t aInputWidth,
    const size_t aNumCategories) :
    mInputHeight(aInputHeight),
    mInputWidth(aInputWidth),
    mNumCategories(aNumCategories),
    fcLayer1(mInputHeight * mInputWidth, aHiddenSize1, mNumCategories, UpdateMethod::ADAM),
    fcLayer2(aHiddenSize1, aHiddenSize2, mNumCategories, UpdateMethod::ADAM),
    fcLayer3(aHiddenSize2, aHiddenSize3, mNumCategories, UpdateMethod::ADAM),
    fcLayerFinal(aHiddenSize3, mNumCategories, mNumCategories, UpdateMethod::ADAM),
    lossLayer{},
    net{
    &fcLayer1,
    &fcLayer2,
    &fcLayer3,
    &fcLayerFinal,
    &lossLayer
    }
    {
        // Reconnect the layers for feedback alignment
        net.ConnectLayers(ConnectionType::DIRECT_FEEDBACK_ALIGNMENT);
    };


    void setTrainInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
    }
    void setTrainLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
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
        fcLayer1.SetLearningRate(mLearningRate);
        fcLayer2.SetLearningRate(mLearningRate);
        fcLayer3.SetLearningRate(mLearningRate);
        fcLayerFinal.SetLearningRate(mLearningRate);
    }

    void train(const size_t aEpochNum);

    std::vector<size_t> test();


};
template <class fcType, class finalLayer>
void MLPTest<fcType, finalLayer>::train(const size_t aEpochNum)
{
    // TRAIN
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::cout << "start new epoch" << std::endl;
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd transposedInput = Input.transpose();
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            fcLayer1.SetData(transposedInput);
            lossLayer.SetData(Label);

            // forward
            net.FullForwardPass();

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + lossLayer.GetLoss();

            // backward
            net.FullBackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                        << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time Elapsed = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    

}

template <class fcType, class finalLayer>
std::vector<size_t> MLPTest<fcType, finalLayer>::test()
{
    // TEST
    std::vector<size_t> vTestResults;
    const size_t vTotalTestSamples = mTestInput.cols();
    size_t vNumCorrectlyClassified = 0;
    std::vector<size_t> vIndexTestVector(vTotalTestSamples);
    std::iota(std::begin(vIndexTestVector), std::end(vIndexTestVector), 0); // Fill with 0, 1, ..., N.
    for (const auto &vIndex : vIndexTestVector)
    {
        MatrixXd Input = mTestInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
        MatrixXd Label = mTestLabels.block(vIndex, 0, 1, mNumCategories);
        fcLayer1.SetData(Input);
        lossLayer.SetData(Label);

        net.FullForwardPass();

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

template <class fcType, class finalLayer>
class MLPTestBP
{
    // Network will have stride = 1 (conv layers) and batch size = 1
    const size_t mInputHeight;
    const size_t mInputWidth;

    const size_t mNumCategories;

    double mLearningRate;

    fcType fcLayer1;
    fcType fcLayer2;
    fcType fcLayer3;
    finalLayer fcLayerFinal;
    SoftmaxLossLayer<> lossLayer;

    MatrixXd mTestInput;
    MatrixXd mTestLabels;
    MatrixXd mTrainInput;
    MatrixXd mTrainLabels;

    NetworkHelper<> net;

public:
    MLPTestBP(
    const size_t aHiddenSize1,
    const size_t aHiddenSize2,
    const size_t aHiddenSize3,
    const size_t aInputHeight,
    const size_t aInputWidth,
    const size_t aNumCategories) :
    mInputHeight(aInputHeight),
    mInputWidth(aInputWidth),
    mNumCategories(aNumCategories),
    fcLayer1(mInputHeight * mInputWidth, aHiddenSize1, UpdateMethod::ADAM),
    fcLayer2(aHiddenSize1, aHiddenSize2, UpdateMethod::ADAM),
    fcLayer3(aHiddenSize2, aHiddenSize3, UpdateMethod::ADAM),
    fcLayerFinal(aHiddenSize3, mNumCategories, UpdateMethod::ADAM),
    lossLayer{},
    net{
    &fcLayer1,
    &fcLayer2,
    &fcLayer3,
    &fcLayerFinal,
    &lossLayer
    }
    {
    };


    void setTrainInputs(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
    {
        mTrainInput = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
    }
    void setTrainLabels(const MatrixXd &aInput, const size_t aNumCategories)
    {
        mTrainLabels = aInput;
        // TODO SHOULD BE CHECKING SIZES AGAIN
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
        fcLayer1.SetLearningRate(mLearningRate);
        fcLayer2.SetLearningRate(mLearningRate);
        fcLayer3.SetLearningRate(mLearningRate);
        fcLayerFinal.SetLearningRate(mLearningRate);
    }

    void train(const size_t aEpochNum);

    std::vector<size_t> test();


};
template <class fcType, class finalLayer>
void MLPTestBP<fcType, finalLayer>::train(const size_t aEpochNum)
{
    // TRAIN
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::random_device rd;
    std::mt19937 g(rd());
    const size_t vTotalTrainSamples = mTrainInput.cols();
    std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
    std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0); // Fill with 0, 1, ..., N.
    double currentLossAggregator;
    for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
    {
        std::cout << "start new epoch" << std::endl;
        std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
        currentLossAggregator = 0;
        for (const auto &vIndex : vIndexTrainVector)
        {
            MatrixXd Input = mTrainInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
            MatrixXd transposedInput = Input.transpose();
            MatrixXd Label = mTrainLabels.block(vIndex, 0, 1, mNumCategories);
            fcLayer1.SetData(transposedInput);
            lossLayer.SetData(Label);

            // forward
            net.FullForwardPass();

            // Accumulate the loss
            currentLossAggregator = currentLossAggregator + lossLayer.GetLoss();

            // backward
            net.FullBackwardPass();
        }
        if (vEpoch % 1 == 0)
        {
            std::cout << "trainingLoss[" << vEpoch << "]"
                        << "=" << currentLossAggregator / static_cast<double>(vTotalTrainSamples) << std::endl;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time Elapsed = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    

}

template <class fcType, class finalLayer>
std::vector<size_t> MLPTestBP<fcType, finalLayer>::test()
{
    // TEST
    std::vector<size_t> vTestResults;
    const size_t vTotalTestSamples = mTestInput.cols();
    size_t vNumCorrectlyClassified = 0;
    std::vector<size_t> vIndexTestVector(vTotalTestSamples);
    std::iota(std::begin(vIndexTestVector), std::end(vIndexTestVector), 0); // Fill with 0, 1, ..., N.
    for (const auto &vIndex : vIndexTestVector)
    {
        MatrixXd Input = mTestInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
        MatrixXd transposedInput = Input.transpose();
        MatrixXd Label = mTestLabels.block(vIndex, 0, 1, mNumCategories);
        fcLayer1.SetData(transposedInput);
        lossLayer.SetData(Label);

        net.FullForwardPass();

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
