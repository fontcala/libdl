#ifndef DLTYPES_H
#define DLTYPES_H

#include "dlfunctions.h"

struct ConvDataDims
{
    const size_t Depth;
    const size_t Height;
    const size_t Width;
    ConvDataDims();
    ConvDataDims(const size_t aDepth,
                 const size_t aHeight,
                 const size_t aWidth);
    ConvDataDims(const size_t aOutDepth,
                 const size_t aInHeight,
                 const size_t aInWidth,
                 const size_t aFilterHeight,
                 const size_t aFilterWidth,
                 const size_t aPaddingHeight,
                 const size_t aPaddingWidth,
                 const size_t aStride);
};
ConvDataDims::ConvDataDims() : Depth(0),
                               Height(0),
                               Width(0) {}
ConvDataDims::ConvDataDims(const size_t aDepth,
                           const size_t aHeight,
                           const size_t aWidth) : Depth(aDepth),
                                                  Height(aHeight),
                                                  Width(aWidth) {}
ConvDataDims::ConvDataDims(const size_t aOutDepth,
                           const size_t aInHeight,
                           const size_t aInWidth,
                           const size_t aFilterHeight,
                           const size_t aFilterWidth,
                           const size_t aPaddingHeight,
                           const size_t aPaddingWidth,
                           const size_t aStride) : Depth(aOutDepth),
                                                   Height((aInHeight - aFilterHeight + 2 * aPaddingHeight) / aStride + 1),
                                                   Width((aInWidth - aFilterWidth + 2 * aPaddingWidth) / aStride + 1)
{
}

// Make non copyable and non movable?
// template <class DimType, class DataType>
// class DataWrapper

// {
// public:
//     DataWrapper(DataType , DimType aDims)
//         : mPtr(aPtr), mDims(aDims) {}
//     DataWrapper()
//         : mPtr(), mDims() {}

//     const DimType &Dimensions() const { return mDims; }
    
// private:
//     DataType mData;
//     DimType mDims;
// };

// My classes templated with an activation function class, make all of this classes base of one given class?
template <class DataType>
class SigmoidActivation
{
private:
    DataType mSigmoidHelper;

public:
    void Activate(DataType& aInput);
    void Backpropagate(DataType& aBackpropInput);
};
template <class DataType>
void SigmoidActivation<DataType>::Activate(DataType& aInput){
    aInput = 1 / (1 + exp(-1 * aInput.array()));
    mSigmoidHelper = aInput; // helper for backprop
}
template <class DataType>
void SigmoidActivation<DataType>::Backpropagate(DataType& aBackpropInput){
    aBackpropInput = aBackpropInput.array() * (mSigmoidHelper.array() * (1 - mSigmoidHelper.array()));

}

template <class DataType>
class LinearActivation
{
public:
    void Activate(DataType& aInput);
    void Backpropagate(DataType& aBackpropInput);
};
template <class DataType>
void LinearActivation<DataType>::Activate(DataType& aInput){
}
template <class DataType>
void LinearActivation<DataType>::Backpropagate(DataType& aBackpropInput){
}

#endif