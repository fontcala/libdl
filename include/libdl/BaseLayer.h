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
@brief Base Class for Network Layer Elements.
*
* Provides layers with the notion of input, output, input dimensions, output dimensions which is common to all layers.
* Every Layer will store a pointer to the data it is going to use during its forward and backward passes as well as the output of the passes,
* as well as the methods to access these outputs.
 */
template <class InputDimType, class BackpropInputDimType, class DataType>
class BaseLayer : public NetworkElement<DataType>
{
protected:
    // Flags
    bool mIsFirstLayerFlag = false;
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

public:
    // Constructors
    BaseLayer();
    BaseLayer(const InputDimType &aInputDims, const BackpropInputDimType &aOutputDims);


    /**
    * BaseLayer::SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
    * overrides 
    * @copydoc NetworkElement::SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
    * 
    * The user probably prefers to set the training/testing data to the whole network like this instead of a pointer, like in \c SetInput.
    * 
    * This function expects an input with the col shape i.e an image is stretched into columns (one column per channel).
    * @remark Using this one indicates \c this is the input layer. In the input Layer there is no need for mBackpropagateOutput update
    * becase we are not going to backpropagate anymore. Knowing this would save an expensive operation at each full backwardpropagation.
    * And this is done setting a flag for whichever class uses this method.
    */
    virtual void SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput) override;
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
        throw(std::runtime_error("SetInput(): invalid input"));
    }
};

template <class InputDimType, class BackpropInputDimType, class DataType>
void BaseLayer<InputDimType, BackpropInputDimType, DataType>::SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    mInputPtr = &aInput;
    mValidInputFlag = true;
    mIsFirstLayerFlag = true;
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
        throw(std::runtime_error("SetBackpropInput(): invalid backprop input"));
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