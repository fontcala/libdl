/** @file FlattenLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "BaseLayer.h"
/**
@class FlattenLayer
@brief Flatten Layer.
 */
template <class DataType>
class FlattenLayer : public BaseLayer<ConvDataDims, size_t, DataType>
{
private:
    const size_t mInputSampleNumber;

public:
    // Constructors
    FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber);
    FlattenLayer(const ConvDataDims aInputDims, const size_t aInputSampleNumber);
    void ForwardPass();
    void BackwardPass();
};

template <class DataType>
FlattenLayer<DataType>::FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber) : BaseLayer<ConvDataDims, size_t, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth),(aInputDepth * aInputHeight * aInputWidth)),
                                                                                                                                                        mInputSampleNumber(aInputSampleNumber){};

template <class DataType>
FlattenLayer<DataType>::FlattenLayer(const ConvDataDims aInputDims, const size_t aInputSampleNumber) : BaseLayer<ConvDataDims, size_t, DataType>(aInputDims, (aInputDims.Depth * aInputDims.Height * aInputDims.Width)),
                                                                                                       mInputSampleNumber(aInputSampleNumber){};

template <class DataType>
void FlattenLayer<DataType>::ForwardPass()
{
    this->mOutput = dlfunctions::flatten(*(this->mInputPtr), mInputSampleNumber);
    // std::cout << "(*mInputPtr)" << std::endl;
    // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    // std::cout << "mOutput" << std::endl;
    // std::cout << mOutput.rows() << " " << mOutput.cols() << std::endl;
};
template <class DataType>
void FlattenLayer<DataType>::BackwardPass()
{
    this->mBackpropOutput = dlfunctions::unflatten(*(this->mBackpropInputPtr), this->mInputDims.Depth, this->mInputDims.Height, this->mInputDims.Width);
    // std::cout << "(*mInputPtr)" << std::endl;
    // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    // std::cout << "mBackpropOutput" << std::endl;
    // std::cout << mBackpropOutput.rows() << " " << mBackpropOutput.cols() << std::endl;
    // std::cout << "(*mBackpropInputPtr)" << std::endl;
    // std::cout << (*mBackpropInputPtr).rows() << " " << (*mBackpropInputPtr).cols() << std::endl;
    // std::cout << "mOutput" << std::endl;
    // std::cout << mOutput.rows() << " " << mOutput.cols() << std::endl;
};
#endif