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
    Eigen::Matrix<DataType, Dynamic, Dynamic> mOutput;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mBackpropOutput;

    // Readonly data from other layers
    const Eigen::Matrix<DataType, Dynamic, Dynamic> *mInputPtr;
    const Eigen::Matrix<DataType, Dynamic, Dynamic> *mBackpropInputPtr;

    // Data dimensions
    const InputDimType mInputDims;
    const BackpropInputDimType mOutputDims;

    // Data from other layers
    // std::pair<InputDimType,DataType> mInputDataPtr;
    // DataPtr<BackpropInputDimType,DataType> mBackpropInputDataPtr;

public:
    // Constructors
    BaseLayer();
    BaseLayer(const InputDimType &aInputDims, const BackpropInputDimType &aOutputDims);

    // Every Layer element must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;

    // Helpers to connect Layers
    void SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
    virtual void SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aInput);
    virtual void SetBackpropInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aOutput);
    virtual const Eigen::Matrix<DataType, Dynamic, Dynamic> *GetOutput() const;
    virtual const Eigen::Matrix<DataType, Dynamic, Dynamic> *GetBackpropOutput() const;
    const BackpropInputDimType &GetOutputDims() const;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
BaseLayer<InputDimType, BackpropInputDimType, DataType>::BaseLayer() : mInputPtr(NULL),
                                                                       mBackpropInputPtr(NULL),
                                                                       mInputDims(),
                                                                       mOutputDims(){};

template <class InputDimType, class BackpropInputDimType, class DataType>
BaseLayer<InputDimType, BackpropInputDimType, DataType>::BaseLayer(const InputDimType &aInputDims, const BackpropInputDimType &aOutputDims) : mInputPtr(NULL),
                                                                                                                                              mBackpropInputPtr(NULL),
                                                                                                                                              mInputDims(aInputDims),
                                                                                                                                              mOutputDims(aOutputDims){};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aInput)
{
    // TODO check validity
    mInputPtr = aInput;
};

// TODO: using this one indicates it is the first layer, so no need for mBackpropagateOutput update (set a Flag)
template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    // TODO check validity
    mInputPtr = &aInput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetBackpropInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aBackpropInput)
{
    // TODO check validity
    mBackpropInputPtr = aBackpropInput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
const Eigen::Matrix<DataType, Dynamic, Dynamic> *BaseLayer<InputDimType, BackpropInputDimType, DataType>::GetOutput() const
{
    return &mOutput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
const Eigen::Matrix<DataType, Dynamic, Dynamic> *BaseLayer<InputDimType, BackpropInputDimType, DataType>::GetBackpropOutput() const
{
    return &mBackpropOutput;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
const BackpropInputDimType &BaseLayer<InputDimType, BackpropInputDimType, DataType>::GetOutputDims() const
{
    return mOutputDims;
};
#endif