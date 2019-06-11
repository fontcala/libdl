/** @file ConnectedBaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONNECTEDBASELAYER_H
#define CONNECTEDBASELAYER_H

#include "BaseLayer.h"
/**
@class ConnectedBaseLayer
@brief Base Connected Layer introduces the parameter of cumputation layers
 */
template <class DimType>
class ConnectedBaseLayer : public BaseLayer<DimType,DimType,MatrixXd>
{
protected:
    MatrixXd mGradientsWeights;
    MatrixXd mGradientsBiases;

    // Weights to be modified often.
    MatrixXd mWeights;
    MatrixXd mBiases;
    MatrixXd mMomentumUpdateWeights;
    MatrixXd mMomentumUpdateBiases;

public:
    double mLearningRate;
    double mMomentumUpdateParam;

    // Processing
    /**
    @function InitParams
    @brief Initialization with <tt>std::mt19937</tt> so that every run is with a different set of weights and biases.
    */
    void InitParams(size_t aInputDim, size_t aOutputDim);
    void UpdateParams();
    // Constructor
    ConnectedBaseLayer();

    // TODO method that checks validity
    // TODO disharcode gradient update (maybe with lambda or sth)

    // Every Layer must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;
};

template <class DimType>
ConnectedBaseLayer<DimType>::ConnectedBaseLayer(){};

template <class DimType>
void ConnectedBaseLayer<DimType>::InitParams(size_t aInputDim, size_t aOutputDim)
{
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, 1.0); // TODO which distribution?
    mWeights = MatrixXd::NullaryExpr(aInputDim, aOutputDim, [&]() { return vRandDistr(vRandom); });
    mBiases = MatrixXd::NullaryExpr(1, aOutputDim, [&]() { return vRandDistr(vRandom); });
    mMomentumUpdateWeights = MatrixXd::Zero(aInputDim, aOutputDim);
    mMomentumUpdateBiases = MatrixXd::Zero(1, aOutputDim);
    mLearningRate = 0.05;
    mMomentumUpdateParam = 0.9;
    this->mInitializedFlag = true;
}
template <class DimType>
void ConnectedBaseLayer<DimType>::UpdateParams()
{
    // TODO User specified
    // Nesterov-Momentum
    MatrixXd vPreviousMomentumUpdateWeights = mMomentumUpdateWeights;
    mMomentumUpdateWeights = mMomentumUpdateParam * mMomentumUpdateWeights - mLearningRate * mGradientsWeights;
    mWeights = mWeights + (-mMomentumUpdateParam * vPreviousMomentumUpdateWeights) + (1 + mMomentumUpdateParam) * mMomentumUpdateWeights;
    MatrixXd vPreviousMomentumUpdateBiases = mMomentumUpdateBiases;
    mMomentumUpdateBiases = mMomentumUpdateParam * mMomentumUpdateBiases - mLearningRate * mGradientsBiases;
    mBiases = mBiases + (-mMomentumUpdateParam * vPreviousMomentumUpdateBiases) + (1 + mMomentumUpdateParam) * mMomentumUpdateBiases;

    // Vanilla Descent
    // mWeights = mWeights - mLearningRate * mGradientsWeights;
    // mBiases = mBiases - mLearningRate * mGradientsBiases;
}

#endif