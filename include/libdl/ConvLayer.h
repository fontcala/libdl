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
template <int FilterHeight, int FilterWidth, int PaddingHeight, int PaddingWidth, int Stride>
class ConvLayer : public ConnectedBaseLayer
{
protected:
    // Layer-specific properties.

    const size_t mInputDepth;
    const size_t mInputHeight;
    const size_t mInputWidth;

    const size_t mOutputDepth;
    const size_t mOutputHeight;
    const size_t mOutputWidth;

    const size_t mInputSampleNumber; //could be deduced in the flatten Layer!
    const size_t mFilterSize;

public:
    // Constructors
    ConvLayer(const size_t aInputDepth,
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

template <int FilterHeight, int FilterWidth, int PaddingHeight, int PaddingWidth, int Stride>
ConvLayer<FilterHeight, FilterWidth, PaddingHeight, PaddingWidth, Stride>::ConvLayer(const size_t aInputDepth,
                                                                                     const size_t aInputHeight,
                                                                                     const size_t aInputWidth,
                                                                                     const size_t aOutputDepth,
                                                                                     const size_t aOutputHeight,
                                                                                     const size_t aOutputWidth,
                                                                                     const size_t aInputSampleNumber) : mInputDepth(aInputDepth),
                                                                                                                        mInputHeight(aInputHeight),
                                                                                                                        mInputWidth(aInputWidth),
                                                                                                                        mOutputDepth(aOutputDepth),
                                                                                                                        mOutputHeight(aOutputHeight),
                                                                                                                        mOutputWidth(aOutputWidth),
                                                                                                                        mInputSampleNumber(aInputSampleNumber),
                                                                                                                        mFilterSize(FilterHeight * FilterWidth * aInputDepth)
{
    InitParams(mFilterSize, mOutputDepth);
};

template <int FilterHeight, int FilterWidth, int PaddingHeight, int PaddingWidth, int Stride>
void ConvLayer<FilterHeight, FilterWidth, PaddingHeight, PaddingWidth, Stride>::ForwardPass() {
    
    MatrixXd vOutputConvolution(mOutputHeight * mOutputWidth, mOutputDepth);
    dlfunctions::convolution<3, 3>(vOutputConvolution,mWeights,(*mInputPtr),mOutputHeight, mOutputWidth, mInputHeight, mInputWidth, mInputDepth, PaddingHeight, PaddingWidth, Stride,mInputSampleNumber);
    mOutput = vOutputConvolution + mBiases.replicate(mOutputHeight * mOutputWidth, 1);
}

template <int FilterHeight, int FilterWidth, int PaddingHeight, int PaddingWidth, int Stride>
void ConvLayer<FilterHeight, FilterWidth, PaddingHeight, PaddingWidth, Stride>::BackwardPass() {

    // Flip the kernels
    MatrixXd vFlippedFilters = dlfunctions::flip(mweights,mInputDepth);

    // Full convolution means using padding Filter Height/Width - 1.

    // Compute all of them.
}

#endif