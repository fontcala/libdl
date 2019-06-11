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
class FlattenLayer : public BaseLayer<ConvDataDims,size_t,MatrixXd>
{
public:
    const size_t mInputDepth;
    const size_t mInputHeight;
    const size_t mInputWidth;

    const size_t mInputSampleNumber;
    // Constructors
    FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber);

    void ForwardPass();
    void BackwardPass();
};

FlattenLayer::FlattenLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aInputSampleNumber) : mInputDepth(aInputDepth),
                                                                                                                                             mInputHeight(aInputHeight),
                                                                                                                                             mInputWidth(aInputWidth),
                                                                                                                                             mInputSampleNumber(aInputSampleNumber){};

void FlattenLayer::ForwardPass(){
    mOutput = dlfunctions::flatten((*mInputPtr),mInputSampleNumber);
};
void FlattenLayer::BackwardPass(){

};
#endif