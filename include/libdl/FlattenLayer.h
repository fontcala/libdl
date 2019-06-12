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
class FlattenLayer : public BaseLayer<ConvDataDims, size_t, MatrixXd>
{
private:
    const ConvDataDims mInputDims;
    const size_t mOutputDims;
    const size_t mInputSampleNumber;

public:
    // Constructors
    FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber);
    FlattenLayer(const ConvDataDims aInputDims, const size_t aInputSampleNumber);
    void ForwardPass();
    void BackwardPass();
    size_t GetOutputDims();
};

FlattenLayer::FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber) : mInputDims(aInputDepth, aInputHeight, aInputWidth),
                                                                                                                                             mOutputDims(aInputDepth * aInputHeight * aInputWidth),
                                                                                                                                             mInputSampleNumber(aInputSampleNumber){};
FlattenLayer::FlattenLayer(const ConvDataDims aInputDims, const size_t aInputSampleNumber) : mInputDims(aInputDims),
                                                                                             mOutputDims(aInputDims.Depth * aInputDims.Height * aInputDims.Width),
                                                                                             mInputSampleNumber(aInputSampleNumber){};

void FlattenLayer::ForwardPass()
{
    mOutput = dlfunctions::flatten((*mInputPtr), mInputSampleNumber);
    std::cout << "(*mInputPtr)" << std::endl;
    std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    std::cout << "mOutput" << std::endl;
    std::cout << mOutput.rows() << " " << mOutput.cols() << std::endl;
};
void FlattenLayer::BackwardPass()
{
    mBackpropOutput = dlfunctions::unflatten((*mBackpropInputPtr), mInputDims.Depth, mInputDims.Height, mInputDims.Width);
    std::cout << "(*mInputPtr)" << std::endl;
    std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    std::cout << "mBackpropOutput" << std::endl;
    std::cout << mBackpropOutput.rows() << " " << mBackpropOutput.cols() << std::endl;
    std::cout << "(*mBackpropInputPtr)" << std::endl;
    std::cout << (*mBackpropInputPtr).rows() << " " << (*mBackpropInputPtr).cols() << std::endl;
    std::cout << "mOutput" << std::endl;
    std::cout << mOutput.rows() << " " << mOutput.cols() << std::endl;
};
size_t FlattenLayer::GetOutputDims()
{

    return mOutputDims;
}
#endif