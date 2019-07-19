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
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);
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
        firstConvLayer.SetData(Input);
        lossLayer.SetData(Label);
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

class CNNClassificationExampleModel2
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
std::vector<size_t> CNNClassificationExampleModel2::runExample(const size_t aEpochNum)
{
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 5;
    const size_t vPaddingHeight1 = 0;
    const size_t vPaddingWidth1 = 0;
    const size_t vStride1 = 1;
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
    // MaxPool
    const size_t vPoolSize = 2;
    const size_t vStridePool = 2;
    MaxPoolLayer maxp1(firstConvLayer.GetOutputDims(),
                       vPoolSize,
                       vStridePool,
                       vInputSampleNumber);
    // Conv 2
    const size_t vFilterHeight2 = 3;
    const size_t vFilterWidth2 = 3;
    const size_t vPaddingHeight2 = 0;
    const size_t vPaddingWidth2 = 0;
    const size_t vStride2 = 1;
    const size_t vOutputDepth2 = 8;
    ConvLayer<ReLUActivation> secondConvLayer(vFilterHeight2,
                                              vFilterWidth2,
                                              vPaddingHeight2,
                                              vPaddingWidth2,
                                              vStride2,
                                              maxp1.GetOutputDims(),
                                              vOutputDepth2,
                                              vInputSampleNumber);

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    FullyConnectedLayer<LinearActivation> fcLayer(flattenLayer.GetOutputDims(), mNumCategories);

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
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);
            //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
            firstConvLayer.ForwardPass();
            //std::cout << "maxp1.ForwardPass()  ------" << std::endl;
            maxp1.ForwardPass();
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
            maxp1.BackwardPass();
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
        firstConvLayer.SetData(Input);
        lossLayer.SetData(Label);
        //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
        firstConvLayer.ForwardPass();
        //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
        maxp1.ForwardPass();
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

class CNNClassificationExampleModel3
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
std::vector<size_t> CNNClassificationExampleModel3::runExample(const size_t aEpochNum)
{
    // NETWORK DESIGN
    const size_t vInputSampleNumber = 1;

    // Conv 1
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 5;
    const size_t vPaddingHeight1 = 0;
    const size_t vPaddingWidth1 = 0;
    const size_t vStride1 = 1;
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
                                             vInputSampleNumber,UpdateMethod::ADAM);
    // MaxPool
    const size_t vPoolSize = 2;
    const size_t vStridePool = 2;
    MaxPoolLayer maxp1(firstConvLayer.GetOutputDims(),
                       vPoolSize,
                       vStridePool,
                       vInputSampleNumber);
    // Conv 2
    const size_t vFilterHeight2 = 3;
    const size_t vFilterWidth2 = 3;
    const size_t vPaddingHeight2 = 0;
    const size_t vPaddingWidth2 = 0;
    const size_t vStride2 = 1;
    const size_t vOutputDepth2 = 8;
    ConvLayer<ReLUActivation> secondConvLayer(vFilterHeight2,
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
    FullyConnectedLayer<LinearActivation> fcLayer(flattenLayer.GetOutputDims(), mNumCategories);

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
            firstConvLayer.SetData(Input);
            lossLayer.SetData(Label);
            //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
            firstConvLayer.ForwardPass();
            //std::cout << "maxp1.ForwardPass()  ------" << std::endl;
            maxp1.ForwardPass();
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
            maxp1.BackwardPass();
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
        firstConvLayer.SetData(Input);
        lossLayer.SetData(Label);
        //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
        firstConvLayer.ForwardPass();
        //std::cout << "firstConvLayer.ForwardPass()  ------" << std::endl;
        maxp1.ForwardPass();
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