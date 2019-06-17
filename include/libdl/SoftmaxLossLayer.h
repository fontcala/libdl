/** @file LossLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef SOFTMAXLOSSLAYER_H
#define SOFTMAXLOSSLAYER_H

#include "LossBaseLayer.h"
/**
@class SoftmaxLossLayer
@brief Softmax Loss Layer.
 */
template <class DataType>
class SoftmaxLossLayer : public LossBaseLayer<DataType>
{
protected:
    // Data
    MatrixXd mGradientHelper;

public:
    // Constructors
    SoftmaxLossLayer();

    void ForwardPass();
    void BackwardPass();
};

template <class DataType>
SoftmaxLossLayer<DataType>::SoftmaxLossLayer(){};

template <class DataType>
void SoftmaxLossLayer<DataType>::ForwardPass()
{
    //check same number of training samples.
    const size_t vLabelNum = this->mLabels.rows();
    const size_t vOutputNum = this->mInputPtr->rows();
    if (vLabelNum == vOutputNum)
    {
        //Softmax
        MatrixXd exp = (*(this->mInputPtr)).array().exp();
        this->mOutput = exp.array().colwise() / exp.rowwise().sum().array();
        MatrixXd logprobs = -this->mOutput.array().log();
        MatrixXd filtered = logprobs.cwiseProduct(this->mLabels);

        //mOutput = mGradientHelper;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*mInputPtr) << std::endl;
        // std::cout << "exp" << std::endl;
        // std::cout << exp << std::endl;
        // std::cout << "probs" << std::endl;
        // std::cout << mGradientHelper << std::endl;
        // std::cout << "filtered" << std::endl;
        // std::cout << filtered << std::endl;
        // Loss divided by number of examples
        this->mLoss = filtered.array().sum() / static_cast<double>(vOutputNum);
    }
    else
    {
        throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
    }
};
template <class DataType>
void SoftmaxLossLayer<DataType>::BackwardPass()
{
    this->mBackpropOutput = this->mOutput - this->mLabels;
    // std::cout << "(*mInputPtr)" << std::endl;
    // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    std::cout << "mBackpropOutput" << std::endl;
    std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols() << std::endl;
};
#endif