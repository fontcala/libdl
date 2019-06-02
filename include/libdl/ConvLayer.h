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
template <int FilterSizeX, int FilterSizeY>
class ConvLayer : public BaseLayer<MatrixXd>
{
protected:
    // Data
    const size_t mInputFilterNumber;
    const size_t mOutputFilterNumber;
    const size_t mInputDataSizeX;
    const size_t mInputDataSizeY;
    const size_t mInputDataNumber;
    // Initialize Params
    void InitParams();

public:
    // Constructors
    ConvLayer(const size_t aInputFilterNumber, const size_t aOutputFilterNumber, const size_t aInputDataSizeX, const size_t aInputDataSizeY, const size_t aInputNumber);

    // Every Layer element must implement these
    void ForwardPass();
    void BackwardPass();
};

template <int FilterSizeX, int FilterSizeY>
ConvLayer<FilterSizeX, FilterSizeY>::ConvLayer(const size_t aInputFilterNumber,
                                               const size_t aOutputFilterNumber,
                                               const size_t aInputDataSizeX,
                                               const size_t aInputDataSizeY,
                                               const size_t aInputDataNumber) : mInputFilterNumber(aInputFilterNumber),
                                                                            mOutputFilterNumber(aOutputFilterNumber),
                                                                            mInputDataSizeX(aInputDataSizeX),
                                                                            mIputDataSizeY(aInputDataSizeY),
                                                                            mInputDataNumber(aInputDataNumber){};

template <int FilterSizeX, int FilterSizeY>
void ConvLayer<FilterSizeX, FilterSizeY>::InitParams()
{
    if (InputSizeX < FilterSizeX || InputSizeY < FilterSizeY)
    {
        throw(std::runtime_error("ConvLayer::InitParams(): dimensions not right"));
    }
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, 1.0);
    double vParamScaleFactor = sqrt(1.0 / static_cast<double>(mInputDim));
    mWeights = vParamScaleFactor * MatrixXd::NullaryExpr(mInputDataSizeX * mInputFilterNumber, mInputDataSizeY *, [&]() { return vRandDistr(vRandom); });
    mBiases = vParamScaleFactor * MatrixXd::NullaryExpr(1, mOutputDim, [&]() { return vRandDistr(vRandom); });
    mMomentumUpdateWeights = MatrixXd::Zero(mInputDim, mOutputDim);
    mMomentumUpdateBiases = MatrixXd::Zero(1, mOutputDim);
    mMomentumUpdateParam = 0.9;
    mInitializedFlag = true;
}
#endif