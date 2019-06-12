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
class SoftmaxLossLayer : public LossBaseLayer
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

SoftmaxLossLayer::SoftmaxLossLayer(){};

void SoftmaxLossLayer::ForwardPass()
{
    //check same number of training samples.
    const size_t vLabelNum = mLabels.rows();
    const size_t vOutputNum = (*mInputPtr).rows();
    if (vLabelNum == vOutputNum)
    {
        //Softmax
        MatrixXd exp = (*mInputPtr).array().exp();
        mGradientHelper = exp.array().colwise() / exp.rowwise().sum().array();
        MatrixXd logprobs = -mGradientHelper.array().log();

        MatrixXd filtered = logprobs.cwiseProduct(mLabels);

        std::cout << "(*mInputPtr)" << std::endl;
        std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*mInputPtr) << std::endl;
        // std::cout << "exp" << std::endl;
        // std::cout << exp << std::endl;
        // std::cout << "probs" << std::endl;
        // std::cout << mGradientHelper << std::endl;
        // std::cout << filtered << std::endl;
        // Loss divided by number of examples
        mLoss = filtered.array().sum() / static_cast<double>(vOutputNum);
    }
    else
    {
        throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
    }
};
void SoftmaxLossLayer::BackwardPass()
{
    mBackpropOutput = mGradientHelper - mLabels;
    std::cout << "(*mInputPtr)" << std::endl;
    std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    std::cout << "mBackpropOutput" << std::endl;
    std::cout << mBackpropOutput.rows() << " " << mBackpropOutput.cols() << std::endl;
};
#endif