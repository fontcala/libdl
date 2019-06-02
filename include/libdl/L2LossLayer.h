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
class L2LossLayer : public LossBaseLayer
{
protected:

    MatrixXd mGradientHelper;

public:
    // Constructors
    L2LossLayer();

    void ForwardPass();
    void BackwardPass();
};

L2LossLayer::L2LossLayer(){};

void L2LossLayer::ForwardPass()
{
    const size_t vLabelNum = mLabels.rows();
    const size_t vOutputNum = (*mInputPtr).rows();
    if (vLabelNum == vOutputNum)
    {
        //L2
        mGradientHelper = (*mInputPtr) - mLabels;
        mLoss = 0.5 * mGradientHelper.squaredNorm();
    }
    else
    {
        throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
    }
};
void L2LossLayer::BackwardPass()
{
    mBackpropOutput = mGradientHelper;
};
#endif