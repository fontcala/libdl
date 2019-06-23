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
template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
class ConnectedBaseLayer : public BaseLayer<DimType, DimType, DataType>
{
protected:

    // todo make this dependent on DataType instead.
    ActivationFunctionType<DataType> ActivationFunction;
    
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientsWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientsBiases;

    // Weights to be modified often.
    Eigen::Matrix<DataType, Dynamic, Dynamic> mWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mBiases;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mMomentumUpdateWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mMomentumUpdateBiases;

public:
    DataType mLearningRate;
    DataType mMomentumUpdateParam;

    // Processing
    /**
    @function InitParams
    @brief Initialization with <tt>std::mt19937</tt> so that every run is with a different set of weights and biases.
    */
    void InitParams(size_t aInputDim, size_t aOutputDim, double aInitVariance);
    void UpdateParams();
    // Constructor
    ConnectedBaseLayer();
    ConnectedBaseLayer(const DimType& aInputDims, const DimType& aOutputDims);

    // TODO method that checks validity
    // TODO disharcode gradient update (maybe with lambda or sth)

    // Every Layer must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;
};

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
ConnectedBaseLayer<DimType, ActivationFunctionType,DataType>::ConnectedBaseLayer(){};

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
ConnectedBaseLayer<DimType, ActivationFunctionType,DataType>::ConnectedBaseLayer(const DimType& aInputDims, const DimType& aOutputDims):BaseLayer<DimType, DimType, DataType>(aInputDims,aOutputDims){};

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType,DataType>::InitParams(size_t aInputDim, size_t aOutputDim, double aInitVariance)
{
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, 1.0); // TODO which distribution? (maybe try Ho / Xavier initialization)
    mWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(aInputDim, aOutputDim, [&]() { return vRandDistr(vRandom); });
    mBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(1, aOutputDim, [&]() { return vRandDistr(vRandom); });
    mMomentumUpdateWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(aInputDim, aOutputDim);
    mMomentumUpdateBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDim);
    mLearningRate = 0.05;
    mMomentumUpdateParam = 0.9;
    this->mInitializedFlag = true;
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType,DataType>::UpdateParams()
{
    // TODO User specified
    // Nesterov-Momentum
    Eigen::Matrix<DataType, Dynamic, Dynamic> vPreviousMomentumUpdateWeights = mMomentumUpdateWeights;
    mMomentumUpdateWeights = mMomentumUpdateParam * mMomentumUpdateWeights - mLearningRate * mGradientsWeights;
    mWeights = mWeights + (-mMomentumUpdateParam * vPreviousMomentumUpdateWeights) + (1 + mMomentumUpdateParam) * mMomentumUpdateWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vPreviousMomentumUpdateBiases = mMomentumUpdateBiases;
    mMomentumUpdateBiases = mMomentumUpdateParam * mMomentumUpdateBiases - mLearningRate * mGradientsBiases;
    mBiases = mBiases + (-mMomentumUpdateParam * vPreviousMomentumUpdateBiases) + (1 + mMomentumUpdateParam) * mMomentumUpdateBiases;

    // Vanilla Descent
    // mWeights = mWeights - mLearningRate * mGradientsWeights;
    // mBiases = mBiases - mLearningRate * mGradientsBiases;
}

#endif