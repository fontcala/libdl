/** @file L2LossLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef L2LOSSLAYER_H
#define L2LOSSLAYER_H

#include "LossBaseLayer.h"
/**
* @class L2LossLayer
* @brief L2 Loss Layer.
* 
* Implements the basic squared difference. Useful in task such as Autoencoder.
* 
* The output is set to be the input for interpretation of results.
* 
*  The loss value is normalized by \c this->mLossNormalizationFactor.
*/
template <class DataType = double>
class L2LossLayer final : public LossBaseLayer<DataType>
{
protected:
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientHelper;

public:
    // Constructors
    L2LossLayer(double aLossNormalizationFactor = 1.0);

    void ForwardPass() override;
    void BackwardPass() override;
};

template <class DataType>
L2LossLayer<DataType>::L2LossLayer(double aLossNormalizationFactor): LossBaseLayer<DataType>(aLossNormalizationFactor){};

template <class DataType>
void L2LossLayer<DataType>::ForwardPass()
{

    if (this->ValidData())
    {
        this->mOutput = *(this->mInputPtr);
        mGradientHelper = this->mOutput - this->mLabels;
        this->mLoss = (0.5 / this->mLossNormalizationFactor) * mGradientHelper.rowwise().squaredNorm().sum();
    }
    else
    {
        throw(std::runtime_error("L2LossLayer::ForwardPass(): dimension mismatch (may be caused by wrong label input or upsampling part of the network not matching the size of the downsampling part."));
    }
};

template <class DataType>
void L2LossLayer<DataType>::BackwardPass()
{
    this->mBackpropOutput = mGradientHelper;
};
#endif