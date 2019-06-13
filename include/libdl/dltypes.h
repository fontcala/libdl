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

template <class DimType, class DataType>
class DataPtr
{
public:
    DataPtr(DataType *aPtr, DimType aDims)
        : mPtr(aPtr), mDims(aDims) {}
    DataPtr()
        : mPtr(NULL), mDims() {}

    DataType *operator->() const { return mPtr; }
    DataType &operator*() const { return *mPtr; }
    DimType IndexToArr() const { return mDims; }

private:
    DataType *mPtr;
    DimType mDims;
};

// My classes templated with an activation function class?
class SigmoidActivation
{
private:
    MatrixXd mSigmoidOutput;

public:
    void Activate();
    MatrixXd BackpropagationFactor();
};

#endif