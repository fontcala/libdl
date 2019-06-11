/** @file BaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef BASELAYER_H
#define BASELAYER_H

#include <memory>
#include "dlfunctions.h"
#include "dltypes.h"


/**
@class BaseLayer
@brief Base Class for Network Layer elements.
 */
template <class InputDimType, class BackpropInputDimType, class DataType>
class BaseLayer
{
protected:

    // Flags
    bool mInitializedFlag = false;
    bool mValidInput = false;
    // Data
    DataType mOutput;
    DataType mBackpropOutput;

    // Parent
    BaseLayer * mParent;

    // Readonly data from other layers
    const DataType *mInputPtr;
    const DataType *mBackpropInputPtr;

    // Data from other layers
    //const int whatev;
    // DataPtr<InputDimType,DataType> mInputDataPtr;
    // DataPtr<BackpropInputDimType,DataType> mBackpropInputDataPtr;


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

template <class InputDimType, class BackpropInputDimType, class DataType>
BaseLayer<InputDimType,BackpropInputDimType,DataType>::BaseLayer() : mInputPtr(NULL), mBackpropInputPtr(NULL){};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType,BackpropInputDimType,DataType>::SetInput(const DataType *aInput)
{
    // TODO check validity
    mInputPtr = aInput;
};
template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType,BackpropInputDimType,DataType>::SetInput(const DataType &aInput)
{
    // TODO check validity
    mInputPtr = &aInput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType,BackpropInputDimType,DataType>::SetBackpropInput(const DataType *aBackpropInput)
{
    // TODO check validity
    mBackpropInputPtr = aBackpropInput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
const DataType *BaseLayer<InputDimType,BackpropInputDimType,DataType>::GetOutput() const
{
    return &mOutput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
const DataType *BaseLayer<InputDimType,BackpropInputDimType,DataType>::GetBackpropOutput() const
{
    return &mBackpropOutput;
};
#endif