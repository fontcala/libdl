/** @file LossLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef SOFTMAXLOSSLAYER_H
#define SOFTMAXLOSSLAYER_H

#include "LossBaseLayer.h"
/**
@class SoftmaxLossLayer
@brief Softmax Loss Layer (cross entropy loss).

* The output is set to be the class probabilities.
 */
template <class DataType = double>
class SoftmaxLossLayer final : public LossBaseLayer<DataType>
{
protected:
    // Data
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientHelper;

public:
    // Constructors
    SoftmaxLossLayer();

    void ForwardPass() override;
    void BackwardPass() override;
};

template <class DataType>
SoftmaxLossLayer<DataType>::SoftmaxLossLayer(){};

template <class DataType>
void SoftmaxLossLayer<DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        //check same number of training samples.
        const size_t vLabelNum = this->mLabels.rows();
        const size_t vOutputNum = this->mInputPtr->rows();
        if (vLabelNum == vOutputNum)
        {
            //Softmax
            Eigen::Matrix<DataType, Dynamic, Dynamic> exp = (*(this->mInputPtr)).array().exp();
            this->mOutput = exp.array().colwise() / exp.rowwise().sum().array();
            Eigen::Matrix<DataType, Dynamic, Dynamic> logprobs = -this->mOutput.array().log();
            Eigen::Matrix<DataType, Dynamic, Dynamic> filtered = logprobs.cwiseProduct(this->mLabels);
            this->mLoss = filtered.array().sum() / static_cast<double>(vOutputNum);
        }
        else
        {
            throw(std::runtime_error("ForwardPass(): dimension mismatch"));
        }
    }
    else
    {
        throw(std::runtime_error("ForwardPass(): invalid input"));
    };
};
template <class DataType>
void SoftmaxLossLayer<DataType>::BackwardPass()
{
    this->mBackpropOutput = this->mOutput - this->mLabels;
};
#endif