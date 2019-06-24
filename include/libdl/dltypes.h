#ifndef DLTYPES_H
#define DLTYPES_H

#include "dlfunctions.h"

/**
@class ConvDataDims.
@brief encapsulates dimensions of convolutional data.
 */
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
                 const size_t aPoolSize,
                 const size_t aStride);
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
ConvDataDims::ConvDataDims(const size_t aInDepth,
                           const size_t aInHeight,
                           const size_t aInWidth,
                           const size_t aPoolSize,
                           const size_t aStride) : Depth(aInDepth),
                                                   Height((aInHeight - aPoolSize) / aStride + 1),
                                                   Width((aInWidth - aPoolSize) / aStride + 1) {}
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

// TODO: Data Structure to pass pointer and dimensions too.
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
// Idea of this is to make the weight initialization for each layer dependant on which activation function class used.

/**
@class LinearActivation.
@brief linear activation, mainly for use in the layer before loss layer.
 */
template <class DataType>
class LinearActivation
{
public:
    void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
    void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput);
};
template <class DataType>
void LinearActivation<DataType>::ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
}
template <class DataType>
void LinearActivation<DataType>::BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput)
{
}

/**
@class SigmoidActivation.
@brief sigmoid activation.
 */
template <class DataType>
class SigmoidActivation
{
private:
    Eigen::Matrix<DataType, Dynamic, Dynamic> mSigmoidHelper;

public:
    void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
    void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput);
};
template <class DataType>
void SigmoidActivation<DataType>::ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    aInput = 1 / (1 + exp(-1 * aInput.array()));
    mSigmoidHelper = aInput; // helper for backprop
}
template <class DataType>
void SigmoidActivation<DataType>::BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput)
{
    aBackpropInput = aBackpropInput.array() * (mSigmoidHelper.array() * (1 - mSigmoidHelper.array()));
}

/**
@class ReLUActivation.
@brief ReLU activation.
 */
template <class DataType>
class ReLUActivation
{
private:
    Eigen::Matrix<DataType, Dynamic, Dynamic> mReLUHelper;

public:
    void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
    void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput);
};
template <class DataType>
void ReLUActivation<DataType>::ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    mReLUHelper = aInput;
    aInput = aInput.cwiseMax(0);
}
template <class DataType>
void ReLUActivation<DataType>::BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput)
{
    Eigen::Matrix<bool, Dynamic, Dynamic> vDerivative = (mReLUHelper.array() > 0);
    aBackpropInput = aBackpropInput.array() * vDerivative.cast<DataType>().array();
}

#endif