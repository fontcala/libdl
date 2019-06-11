/** @file ConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
@class ConvLayer
@brief Conv Class for Network Layer elements.
 */
class ConvLayer : public ConnectedBaseLayer<ConvDataDims>
{
protected:
    // Layer-specific properties.
    const size_t mFilterHeight;
    const size_t mFilterWidth;
    const size_t mPaddingHeight;
    const size_t mPaddingWidth;
    const size_t mStride;

    const ConvDataDims mInputDims;
    const ConvDataDims mOutputDims;

    const size_t mInputSampleNumber; //TODO: could be deduced in the flatten Layer!
    const size_t mFilterSize;

public:
    // Constructors
    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const size_t aInputDepth,
              const size_t aInputHeight,
              const size_t aInputWidth,
              const size_t aOutputDepth,
              const size_t aInputSampleNumber);

    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const size_t aOutputDepth,
              const size_t aInputSampleNumber);

    // getters
    const ConvDataDims &GetOutputDims();

    // Layer-specific Forward-Backward passes.
    void ForwardPass();
    void BackwardPass();
};

ConvLayer::ConvLayer(const size_t aFilterHeight,
                     const size_t aFilterWidth,
                     const size_t aPaddingHeight,
                     const size_t aPaddingWidth,
                     const size_t aStride,
                     const size_t aInputDepth,
                     const size_t aInputHeight,
                     const size_t aInputWidth,
                     const size_t aOutputDepth,
                     const size_t aInputSampleNumber) : mFilterHeight(aFilterHeight),
                                                        mFilterWidth(aFilterWidth),
                                                        mPaddingHeight(aPaddingHeight),
                                                        mPaddingWidth(aPaddingWidth),
                                                        mStride(aStride),
                                                        mInputDims(aInputDepth, aInputHeight, aInputWidth),
                                                        mOutputDims(aOutputDepth, aInputHeight, aInputWidth, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride),
                                                        mInputSampleNumber(aInputSampleNumber),
                                                        mFilterSize(aFilterHeight * aFilterWidth * aInputDepth)
{
    InitParams(mFilterSize, mOutputDims.Depth);
};

ConvLayer::ConvLayer(const size_t aFilterHeight,
                     const size_t aFilterWidth,
                     const size_t aPaddingHeight,
                     const size_t aPaddingWidth,
                     const size_t aStride,
                     const ConvDataDims aInputDims,
                     const size_t aOutputDepth,
                     const size_t aInputSampleNumber) : mFilterHeight(aFilterHeight),
                                                        mFilterWidth(aFilterWidth),
                                                        mPaddingHeight(aPaddingHeight),
                                                        mPaddingWidth(aPaddingWidth),
                                                        mStride(aStride),
                                                        mInputDims(aInputDims),
                                                        mOutputDims(aOutputDepth, aInputDims.Height, aInputDims.Width, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride),
                                                        mInputSampleNumber(aInputSampleNumber),
                                                        mFilterSize(aFilterHeight * aFilterWidth * aInputDims.Depth)
{
    InitParams(mFilterSize, mOutputDims.Depth);
};

void ConvLayer::ForwardPass()
{
    MatrixXd vOutputConvolution(mOutputDims.Height * mOutputDims.Width, mOutputDims.Depth);
    dlfunctions::convolution(vOutputConvolution, mOutputDims.Height, mOutputDims.Width, mWeights, mFilterHeight, mFilterWidth, (*mInputPtr), mInputDims.Height, mInputDims.Width, mInputDims.Depth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);
    mOutput = vOutputConvolution + mBiases.replicate(mOutputDims.Height * mOutputDims.Width, 1);
    // std::cout << "vOutputConvolution" << std::endl;
    // std::cout << vOutputConvolution << std::endl;

    // std::cout << "mWeights" << std::endl;
    // std::cout << mWeights << std::endl;
    std::cout << "mWeights" << std::endl;
    std::cout << mWeights.rows() << " " << mWeights.cols() << std::endl;

    // std::cout  << "(*mInputPtr)" << std::endl;
    // std::cout  << (*mInputPtr) << std::endl;
}

void ConvLayer::BackwardPass()
{
    // Backprop input from previous layer.
    MatrixXd vBackpropInput = *mBackpropInputPtr;

    // derivative wrt to bias
    mGradientsBiases = vBackpropInput.colwise().sum();

    // derivative wrt filters (dOut/df = In conv Out)
    dlfunctions::convolution(mGradientsWeights, mFilterHeight, mFilterWidth, vBackpropInput, mOutputDims.Height, mOutputDims.Width, (*mInputPtr), mInputDims.Height, mInputDims.Width, mInputDims.Depth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);
    std::cout << "mOutputDims" << std::endl;
    std::cout << mOutputDims.Height << " " << mOutputDims.Width << std::endl;

    std::cout << "mGradientsWeights" << std::endl;
    std::cout << mGradientsWeights.rows()<< " " << mGradientsWeights.cols()<< std::endl;

    std::cout << "vBackpropInput" << std::endl;
    std::cout << vBackpropInput.rows()<<  " " << vBackpropInput.cols() << std::endl;

    std::cout << "Input rowscols" << std::endl;
    std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;

    std::cout << "mWeights" << std::endl;
    std::cout << mWeights.rows() << " " << mWeights.cols() << std::endl;
    // derivative wrt to input (full convolution means using padding Filter Height/Width - 1. Also stride = 1.)
    dlfunctions::fullconvolution(mBackpropOutput, mInputDims.Height, mInputDims.Width, dlfunctions::flip(mWeights, mInputDims.Depth), mFilterHeight, mFilterWidth, vBackpropInput, mOutputDims.Height, mOutputDims.Width, mOutputDims.Depth, 1, mInputSampleNumber);

    // Update.
    UpdateParams();

    // std::cout << "mWeights" << std::endl;
    // std::cout << mWeights.rows()<< " " << mWeights.cols()<< std::endl;

    // std::cout << "mFilterSize" << std::endl;
    // std::cout << mFilterSize << " " <<  "" << std::endl;

    // std::cout << "mInputPtr" << std::endl;
    // std::cout << mInputPtr->rows()<< " " << mInputPtr->cols()<< std::endl;

    // std::cout << "mGradientsWeights" << std::endl;
    // std::cout << mGradientsWeights.rows()<< " " << mGradientsWeights.cols()<< std::endl;

    // std::cout << "mGradientsBiases" << std::endl;
    // std::cout << mGradientsBiases.rows()<< " " << mGradientsBiases.cols()<< std::endl;

    // std::cout << "mBiases" << std::endl;
    // std::cout << mBiases.rows()<< " " << mBiases.cols()<< std::endl;

    // std::cout << "mBackpropOutput" << std::endl;
    // std::cout << mBackpropOutput.rows()<<  " " << mBackpropOutput.cols() << std::endl;

    // std::cout << "vBackpropInput" << std::endl;
    // std::cout << vBackpropInput.rows()<<  " " << vBackpropInput.cols() << std::endl;
}

const ConvDataDims &ConvLayer::GetOutputDims()
{

    return mOutputDims;
}

#endif