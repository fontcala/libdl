/** @file ConnectedBaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONNECTEDBASELAYER_H
#define CONNECTEDBASELAYER_H

#include "BaseLayer.h"
/**
@class ConnectedBaseLayer
@brief Base Connected Layer.
 */
class ConnectedBaseLayer : public BaseLayer<MatrixXd>
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

    // Every connected leyer must initialize its params
    virtual void InitParams() = 0;
    void UpdateParams();
    // Constructor
    ConnectedBaseLayer();

    // TODO method that checks validity
    // TODO disharcode gradient update (maybe with lambda or sth)

    // Every Layer must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;
};

ConnectedBaseLayer::ConnectedBaseLayer(){};

void ConnectedBaseLayer::UpdateParams()
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