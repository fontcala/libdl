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

    // Would not need these in the constructor but for now yes.
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
                                                        mPaddingHeight(mPaddingHeight),
                                                        mPaddingWidth(mPaddingWidth),
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
    dlfunctions::convolution(mFilterHeight, mFilterWidth, vOutputConvolution, mWeights, (*mInputPtr), mOutputHeight, mOutputWidth, mInputHeight, mInputWidth, mInputDepth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);
    mOutput = vOutputConvolution + mBiases.replicate(mOutputHeight * mOutputWidth, 1);
}

void ConvLayer::BackwardPass()
{

    // Flip the kernels
    MatrixXd vFlippedFilters = dlfunctions::flip(mWeights, mInputDepth);

    // Full convolution means using padding Filter Height/Width - 1.

    // Compute all of them.
}

#endif