/** @file ConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "BaseLayer.h"

/**
@class ConvLayer
@brief Conv Class for Network Layer elements.
 */
template <int FilterHeight, int FilterWidth>
class ConvLayer : public BaseLayer<MatrixXd>
{
protected:
    // Data
    const size_t mInputFilterNumber;
    const size_t mOutputFilterNumber;
    const size_t mInputDataHeight;
    const size_t mInputDataWidth;
    const size_t mInputDataNumber;
    // Initialize Params
    void InitParams();

public:
    // Constructors
    ConvLayer(const size_t aInputFilterNumber, const size_t aOutputFilterNumber, const size_t aInputDataHeight, const size_t aInputDataWidth, const size_t aInputNumber);

    // Every Layer element must implement these
    void InitParams();
    void ForwardPass();
    void BackwardPass();
};

template <int FilterHeight, int FilterWidth>
ConvLayer<FilterHeight, FilterWidth>::ConvLayer(const size_t aInputFilterNumber,
                                               const size_t aOutputFilterNumber,
                                               const size_t aInputDataHeight,
                                               const size_t aInputDataWidth,
                                               const size_t aInputDataNumber) : mInputFilterNumber(aInputFilterNumber),
                                                                            mOutputFilterNumber(aOutputFilterNumber),
                                                                            mInputDataHeight(aInputDataHeight),
                                                                            mIputDataWidth(aInputDataWidth),
                                                                            mInputDataNumber(aInputDataNumber){};

template <int FilterHeight, int FilterWidth>
void ConvLayer<FilterHeight, FilterWidth>::InitParams()
{
    if (InputHeight < FilterHeight || InputWidth < FilterWidth)
    {
        throw(std::runtime_error("ConvLayer::InitParams(): dimensions not right"));
    }
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, 1.0);
    double vParamScaleFactor = sqrt(1.0 / static_cast<double>(mInputDim));
    mWeights = vParamScaleFactor * MatrixXd::NullaryExpr(mInputDataHeight * mInputFilterNumber, mInputDataWidth *, [&]() { return vRandDistr(vRandom); });
    mBiases = vParamScaleFactor * MatrixXd::NullaryExpr(1, mOutputDim, [&]() { return vRandDistr(vRandom); });
    mMomentumUpdateWeights = MatrixXd::Zero(mInputDim, mOutputDim);
    mMomentumUpdateBiases = MatrixXd::Zero(1, mOutputDim);
    mMomentumUpdateParam = 0.9;
    mInitializedFlag = true;
}
#endif