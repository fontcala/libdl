/** @file BaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef BASELAYER_H
#define BASELAYER_H

#include <memory>
#include "NetworkElement.h"
#include "dlfunctions.h"
#include "dltypes.h"

/**
@class BaseLayer
@brief Base Class for Network Layer eelements.
 */
template <class InputDimType, class BackpropInputDimType, class DataType>
class BaseLayer : public NetworkElement<DataType>
{
protected:
    // Flags
    bool mInitializedFlag = false;
    bool mValidInputFlag = false;
    bool mValidBackpropInputFlag = false;

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
    // virtual void ForwardPass() = 0;
    // virtual void BackwardPass() = 0;

    // Setter/Getters to connect Layers
    void SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput) override;
    void SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aInput) override;
    void SetBackpropInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aOutput) override;
    const Eigen::Matrix<DataType, Dynamic, Dynamic> *GetOutput() const override;
    const Eigen::Matrix<DataType, Dynamic, Dynamic> *GetBackpropOutput() const override;
    
    // Getter helpers
    const BackpropInputDimType &GetOutputDims() const;
    const InputDimType &GetInputDims() const;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
BaseLayer<InputDimType, BackpropInputDimType, DataType>::BaseLayer() : mInputPtr(nullptr),
                                                                       mBackpropInputPtr(nullptr),
                                                                       mInputDims(),
                                                                       mOutputDims(){};

template <class InputDimType, class BackpropInputDimType, class DataType>
BaseLayer<InputDimType, BackpropInputDimType, DataType>::BaseLayer(const InputDimType &aInputDims, const BackpropInputDimType &aOutputDims) : mInputPtr(nullptr),
                                                                                                                                              mBackpropInputPtr(nullptr),
                                                                                                                                              mInputDims(aInputDims),
                                                                                                                                              mOutputDims(aOutputDims){};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aInput)
{
    if (aInput != nullptr)
    {
        mInputPtr = aInput;
        mValidInputFlag = true;
    }
    else
    {
        throw(std::runtime_error("SetInput(): layer parameters not initialized"));
    }
};

// TODO: using this one indicates it is the first layer, so no need for mBackpropagateOutput update (set a Flag)
template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    // TODO check validity
    mInputPtr = &aInput;
    mValidInputFlag = true;
};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetBackpropInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aBackpropInput)
{
    if (aBackpropInput != nullptr)
    {
        mBackpropInputPtr = aBackpropInput;
        mValidBackpropInputFlag = true;
    }
    else
    {
        throw(std::runtime_error("SetBackpropInput(): layer parameters not initialized"));
    }
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

template <class InputDimType, class BackpropInputDimType, class DataType>
const InputDimType &BaseLayer<InputDimType, BackpropInputDimType, DataType>::GetInputDims() const
{
    return mInputDims;
};
#endif