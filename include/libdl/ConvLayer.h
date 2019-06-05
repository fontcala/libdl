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
template <int FilterHeight, int FilterWidth>
class ConvLayer : public ConnectedBaseLayer
{
protected:
    // Layer-specific properties.
    const size_t mInputFilterNumber;
    const size_t mOutputFilterNumber;
    const size_t mInputDataHeight;
    const size_t mInputDataWidth;
    const size_t mInputDataNumber;
    const size_t mFilterSize;


public:
    // Constructors
    ConvLayer(const size_t aInputFilterNumber, const size_t aOutputFilterNumber, const size_t aInputDataHeight, const size_t aInputDataWidth, const size_t aInputNumber);

    // Layer-specific Forward-Backward passes.
    void ForwardPass();
    void BackwardPass();
};

template <int FilterHeight, int FilterWidth>
ConvLayer<FilterHeight, FilterWidth>::ConvLayer(const size_t aInputFilterNumber,
                                                const size_t aOutputFilterNumber,
                                                const size_t aInputDataHeight,
                                                const size_t aInputDataWidth,
                                                const size_t aInputDataNumber) : mInputFilterNumber(aInputFilterNumber),
                                                                                 mOutputFilterNumber(aOutputFilterNumber),
                                                                                 mInputDataHeight(aInputDataHeight),
                                                                                 mInputDataWidth(aInputDataWidth),
                                                                                 mInputDataNumber(aInputDataNumber),
                                                                                 mFilterSize(FilterHeight * FilterWidth * aInputFilterNumber)
{
    InitParams(mFilterSize,mOutputFilterNumber);
};

template <int FilterHeight, int FilterWidth>
void ConvLayer<FilterHeight, FilterWidth>::ForwardPass() {}

template <int FilterHeight, int FilterWidth>
void ConvLayer<FilterHeight, FilterWidth>::BackwardPass() {}

#endif