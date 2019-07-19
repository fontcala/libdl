/** @file SoftmaxLossLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef SOFTMAXLOSSLAYER_H
#define SOFTMAXLOSSLAYER_H

#include "LossBaseLayer.h"
/**
@class SoftmaxLossLayer
@brief Softmax Loss Layer (cross entropy loss).
*
* The output is set to be the class probabilities.
*
* The loss value is normalized by \c this->mLossNormalizationFactor.
 */
template <class DataType = double>
class SoftmaxLossLayer final : public LossBaseLayer<DataType>
{
protected:
    // Data
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientHelper;

public:
    // Constructors
    SoftmaxLossLayer(double aLossNormalizationFactor = 1.0);

    void ForwardPass() override;
    void BackwardPass() override;
};

template <class DataType>
SoftmaxLossLayer<DataType>::SoftmaxLossLayer(double aLossNormalizationFactor): LossBaseLayer<DataType>(aLossNormalizationFactor){};

template <class DataType>
void SoftmaxLossLayer<DataType>::ForwardPass()
{
    if (this->ValidData())
    {
        //Softmax
        Eigen::Matrix<DataType, Dynamic, Dynamic> exp = (*(this->mInputPtr)).array().exp();
        this->mOutput = exp.array().colwise() / exp.rowwise().sum().array();
        Eigen::Matrix<DataType, Dynamic, Dynamic> logprobs = -this->mOutput.array().log();
        Eigen::Matrix<DataType, Dynamic, Dynamic> filtered = logprobs.cwiseProduct(this->mLabels);
        this->mLoss = filtered.array().sum() / (this->mLossNormalizationFactor);
    }
    else
    {
        throw(std::runtime_error("ForwardPass(): dimension mismatch"));
    };
};
template <class DataType>
void SoftmaxLossLayer<DataType>::BackwardPass()
{
    this->mBackpropOutput = this->mOutput - this->mLabels;
};
#endif