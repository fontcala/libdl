/** @file FlattenLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "BaseLayer.h"
/**
@class FlattenLayer
@brief Flatten Layer, necessary interface between convolutional layers and fully connected layers.
 */
template <class DataType = double>
class FlattenLayer final : public BaseLayer<ConvDataDims, size_t, DataType>
{
private:
    const size_t mInputSampleNumber;

public:
    // Constructors
    FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber);
    FlattenLayer(const ConvDataDims aInputDims, const size_t aInputSampleNumber);
    void ForwardPass() override;
    void BackwardPass() override;
};

template <class DataType>
FlattenLayer<DataType>::FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber) : BaseLayer<ConvDataDims, size_t, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), (aInputDepth * aInputHeight * aInputWidth)),
                                                                                                                                                       mInputSampleNumber(aInputSampleNumber){};

template <class DataType>
FlattenLayer<DataType>::FlattenLayer(const ConvDataDims aInputDims, const size_t aInputSampleNumber) : BaseLayer<ConvDataDims, size_t, DataType>(aInputDims, (aInputDims.Depth * aInputDims.Height * aInputDims.Width)),
                                                                                                       mInputSampleNumber(aInputSampleNumber){};

template <class DataType>
void FlattenLayer<DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        this->mOutput = dlfunctions::flatten(*(this->mInputPtr), mInputSampleNumber);
    }
    else
    {
        throw(std::runtime_error("ForwardPass(): invalid input"));
    };
};
template <class DataType>
void FlattenLayer<DataType>::BackwardPass()
{
    if (this->mValidBackpropInputFlag)
    {
        this->mBackpropOutput = dlfunctions::unflatten(*(this->mBackpropInputPtr), this->mInputDims.Depth, this->mInputDims.Height, this->mInputDims.Width);
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
};
#endif