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
class ConvLayer : public ConnectedBaseLayer
{
protected:
    // Layer-specific properties.
    const size_t mFilterHeight;
    const size_t mFilterWidth;
    const size_t mPaddingHeight;
    const size_t mPaddingWidth;
    const size_t mStride;

    const size_t mInputDepth;
    const size_t mInputHeight;
    const size_t mInputWidth;

    const size_t mOutputDepth;
    const size_t mOutputHeight;
    const size_t mOutputWidth;

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
              const size_t aOutputHeight,
              const size_t aOutputWidth,
              const size_t aInputSampleNumber);

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
                     const size_t aOutputHeight,
                     const size_t aOutputWidth,
                     const size_t aInputSampleNumber) : mFilterHeight(aFilterHeight),
                                                        mFilterWidth(aFilterWidth),
                                                        mPaddingHeight(aPaddingHeight),
                                                        mPaddingWidth(aPaddingWidth),
                                                        mStride(aStride),
                                                        mInputDepth(aInputDepth),
                                                        mInputHeight(aInputHeight),
                                                        mInputWidth(aInputWidth),
                                                        mOutputDepth(aOutputDepth),
                                                        mOutputHeight(aOutputHeight),
                                                        mOutputWidth(aOutputWidth),
                                                        mInputSampleNumber(aInputSampleNumber),
                                                        mFilterSize(aFilterHeight * aFilterWidth * aInputDepth)
{
    InitParams(mFilterSize, mOutputDepth);
};

void ConvLayer::ForwardPass()
{
    MatrixXd vOutputConvolution(mOutputHeight * mOutputWidth, mOutputDepth);
    dlfunctions::convolution(vOutputConvolution, mOutputHeight, mOutputWidth, mWeights, mFilterHeight, mFilterWidth, (*mInputPtr), mInputHeight, mInputWidth, mInputDepth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);
    mOutput = vOutputConvolution + mBiases.replicate(mOutputHeight * mOutputWidth, 1);
    // std::cout << "vOutputConvolution" << std::endl;
    // std::cout << vOutputConvolution << std::endl;

    // std::cout << "mWeights" << std::endl;
    // std::cout << mWeights << std::endl;

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
    dlfunctions::convolution(mGradientsWeights, mFilterHeight, mFilterWidth, vBackpropInput, mOutputHeight, mOutputWidth, (*mInputPtr), mInputHeight, mInputWidth, mInputDepth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);

    // derivative wrt to input (full convolution means using padding Filter Height/Width - 1. Also stride = 1.)
    dlfunctions::fullconvolution(mBackpropOutput, mInputHeight, mInputWidth, dlfunctions::flip(mWeights, mInputDepth), mFilterHeight, mFilterWidth, vBackpropInput, mOutputHeight, mOutputWidth,mOutputDepth,1, mInputSampleNumber);

    // Update.
    UpdateParams();
}

#endif