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

// My classes templated with an activation function class?
class SigmoidActivation
{
private:
    MatrixXd mSigmoidHelper;

public:
    void Activate(MatrixXd& aInput);
    void Backpropagate(MatrixXd& aBackpropInput);
};
void SigmoidActivation::Activate(MatrixXd& aInput){
    aInput = 1 / (1 + exp(-1 * aInput.array()));
    mSigmoidHelper = aInput; // helper for backprop
}
void SigmoidActivation::Backpropagate(MatrixXd& aBackpropInput){
    // std::cout << "- mSigmoidHelper" << std::endl;
    // std::cout << mSigmoidHelper.rows() << " " << mSigmoidHelper.cols()<< std::endl;
    aBackpropInput = aBackpropInput.array() * (mSigmoidHelper.array() * (1 - mSigmoidHelper.array()));
    // std::cout << "- aBackpropOutput" << std::endl;
    // std::cout << aBackpropInput.rows() << " " << aBackpropInput.cols()<< std::endl;
}
#endif