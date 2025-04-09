/** @file dltypes.h
 *  @author Adria Font Calvarons
 */

#ifndef DLTYPES_H
#define DLTYPES_H

#include "dlfunctions.h"
#include <math.h>

/**
@class ConvDataDims.
@brief encapsulates dimensions of convolutional data.
@note uses the Named Constructor Idiom.
@note member notation eased (eg: Depth instead of mDepth)
 */
struct ConvDataDims
{
    const size_t Depth;
    const size_t Height;
    const size_t Width;
    ConvDataDims();
    // Input Constructor
    ConvDataDims(const size_t aDepth,
                 const size_t aHeight,
                 const size_t aWidth);
    // MaxPool output Constructor
    ConvDataDims(const size_t aOutDepth,
                 const size_t aInHeight,
                 const size_t aInWidth,
                 const size_t aPoolSize,
                 const size_t aStride);
    // Normal Convolution output Constructor
    static ConvDataDims NormalConv(
        const size_t aOutDepth,
        const size_t aInHeight,
        const size_t aInWidth,
        const size_t aFilterHeight,
        const size_t aFilterWidth,
        const size_t aPaddingHeight,
        const size_t aPaddingWidth,
        const size_t aStride);
    // Transposed Convolution output Constructor
    static ConvDataDims TransposedConv(
        const size_t aOutDepth,
        const size_t aInHeight,
        const size_t aInWidth,
        const size_t aFilterHeight,
        const size_t aFilterWidth,
        const size_t aPaddingHeight,
        const size_t aPaddingWidth,
        const size_t aStride);
};
ConvDataDims ConvDataDims::NormalConv(
    const size_t aOutDepth,
    const size_t aInHeight,
    const size_t aInWidth,
    const size_t aFilterHeight,
    const size_t aFilterWidth,
    const size_t aPaddingHeight,
    const size_t aPaddingWidth,
    const size_t aStride)
{
    return ConvDataDims(aOutDepth,
                        (aInHeight - aFilterHeight + 2 * aPaddingHeight) / aStride + 1,
                        (aInWidth - aFilterWidth + 2 * aPaddingWidth) / aStride + 1);
}
ConvDataDims ConvDataDims::TransposedConv(
    const size_t aOutDepth,
    const size_t aInHeight,
    const size_t aInWidth,
    const size_t aFilterHeight,
    const size_t aFilterWidth,
    const size_t aPaddingHeight,
    const size_t aPaddingWidth,
    const size_t aStride)
{
    return ConvDataDims(aOutDepth,
                        (aInHeight - 1) * aStride + aFilterHeight - 2 * aPaddingHeight,
                        (aInWidth - 1) * aStride + aFilterWidth - 2 * aPaddingWidth);
}

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

/**
* operator== for type ConvDataDims
 */
bool operator==(const ConvDataDims &aRhs, const ConvDataDims &aLhs)
{
    bool aSame = false;
    if (aRhs.Depth == aLhs.Depth && aRhs.Height == aLhs.Height && aRhs.Width == aLhs.Width)
    {
        aSame = true;
    }
    return aSame;
}

/**
* operator!= for type ConvDataDims
 */
bool operator!=(const ConvDataDims &aRhs, const ConvDataDims &aLhs)
{
    return !(aRhs == aLhs);
}

// TODO: Data Structure to pass pointer and dimensions too?

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



/**
@class LinearActivation.
@brief linear activation, mainly for use in the layer before loss layer.
* 
* Use members \c ForwardFunction and \c BackwardFunction to apply the corresponding inplace transformation to the passed input
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
* @class SigmoidActivation.
* @brief sigmoid activation.
* 
* @copydetails LinearActivation
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
* @class ReLUActivation.
* @brief ReLU activation.
* 
* @copydetails LinearActivation
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

template <class DataType>
class RandomActivation
{
private:
    Eigen::Matrix<DataType, Dynamic, Dynamic> mRandomHelper;

public:
    void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
    void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput);
};
template <class DataType>
void RandomActivation<DataType>::ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{   
    // ugly way but for quick test, later put it in the constructor
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0.5,0.5);
    mRandomHelper = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(aInput.rows(),aInput.cols(), [&]() { return vRandDistr(vRandom); });
    aInput = aInput.array()*mRandomHelper.array();
}
template <class DataType>
void RandomActivation<DataType>::BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aBackpropInput)
{
    aBackpropInput = aBackpropInput.array() * mRandomHelper.array();
}

#endif