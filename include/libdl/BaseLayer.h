/** @file BaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef BASELAYER_H
#define BASELAYER_H

#include <memory>
#include "dlfunctions.h"

/**
@class BaseLayer
@brief Base Class for Network Layer elements.
 */
template <class DataType>
class BaseLayer
{
protected:
    // Data
    DataType mOutput;
    DataType mBackpropOutput;

    // Readonly data from other layers
    const DataType *mInputPtr;
    const DataType *mBackpropInputPtr;

    // Checkers
    bool mInitializedFlag = false;
    bool mValidInput = false;

public:
    // Constructors
    BaseLayer();

    // Every Layer element must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;

    // Helpers to connect Layers
    void SetInput(const DataType &aInput);
    virtual void SetInput(const DataType *aInput);
    virtual void SetBackpropInput(const DataType *aOutput);
    virtual const DataType *GetOutput() const;
    virtual const DataType *GetBackpropOutput() const;
};

template <class DataType>
BaseLayer<DataType>::BaseLayer() : mInputPtr(NULL), mBackpropInputPtr(NULL){};

template <class DataType>
void BaseLayer<DataType>::SetInput(const DataType *aInput)
{
    // TODO check validity
    mInputPtr = aInput;
};
template <class DataType>
void BaseLayer<DataType>::SetInput(const DataType &aInput)
{
    // TODO check validity
    mInputPtr = &aInput;
};

template <class DataType>
void BaseLayer<DataType>::SetBackpropInput(const DataType *aBackpropInput)
{
    // TODO check validity
    mBackpropInputPtr = aBackpropInput;
};

template <class DataType>
const DataType *BaseLayer<DataType>::GetOutput() const
{
    return &mOutput;
};

template <class DataType>
const DataType *BaseLayer<DataType>::GetBackpropOutput() const
{
    return &mBackpropOutput;
};
#endif