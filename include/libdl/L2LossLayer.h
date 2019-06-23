/** @file L2LossLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef L2LOSSLAYER_H
#define L2LOSSLAYER_H

#include "LossBaseLayer.h"
/**
@class L2LossLayer
@brief L2 Loss Layer.
 */
template <class DataType = double>
class L2LossLayer : public LossBaseLayer<DataType>
{
protected:

    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientHelper;

public:
    // Constructors
    L2LossLayer();

    void ForwardPass();
    void BackwardPass();
};

template <class DataType>
L2LossLayer<DataType>::L2LossLayer(){};

template <class DataType>
void L2LossLayer<DataType>::ForwardPass()
{
    //check same number of training samples.
    const size_t vLabelNum = mLabels.rows();
    const size_t vOutputNum = (*mInputPtr).rows();
    if (vLabelNum == vOutputNum)
    {
        //L2
        mGradientHelper = (*mInputPtr) - mLabels;
        // Loss divided by number of examples;
        mLoss = (0.5/static_cast<double>(vOutputNum)) * mGradientHelper.rowwise().squaredNorm().sum();
    }
    else
    {
        throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
    }
};

template <class DataType>
void L2LossLayer<DataType>::BackwardPass()
{
    mBackpropOutput = mGradientHelper;
};
#endif