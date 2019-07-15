/** @file ConnectedBaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONNECTEDBASELAYER_H
#define CONNECTEDBASELAYER_H

#include "BaseLayer.h"
#include <math.h>

enum class UpdateMethod
{
    NESTEROV,
    VANILLA,
    ADAM
};

/**
@class ConnectedBaseLayer
@brief Base Layer for classes with parameters, introduces the parameters, their update and initialization.

@note Weight Initialization is He Initialization.
@note Gradient Update is SGD with nesterov momentum.
 */
template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
class ConnectedBaseLayer : public BaseLayer<DimType, DimType, DataType>
{
protected:
    // todo make this dependent on DataType instead.
    ActivationFunctionType<DataType> ActivationFunction;

    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientsWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientsBiases;

    // Update Method:
    UpdateMethod mUpdateMethod;
    // Weights and params:
    Eigen::Matrix<DataType, Dynamic, Dynamic> mWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mBiases;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mMomentumUpdateWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mMomentumUpdateBiases;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mSecondMomentumUpdateWeights;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mSecondMomentumUpdateBiases;

public:
    DataType mLearningRate;
    DataType mMomentumUpdateParam;
    DataType mSecondMomentumUpdateParam;
    DataType mStep;

    // Processing
    /**
    @function InitParams
    @brief Initialization with <tt>std::mt19937</tt> so that every run is with a different set of weights and biases.
    @note He Initialization dependent on layer type with the parameter  \c aInitVariance
    */
    void InitParams(size_t aInputDim, size_t aOutputDimWeights, size_t aOutputDimBiases, double aInitVariance);
    void UpdateParams();
    // Constructor
    ConnectedBaseLayer();
    ConnectedBaseLayer(const DimType &aInputDims, const DimType &aOutputDims, const UpdateMethod aUpdateMethod);

    // Every final Layer must implement these
    // virtual void ForwardPass() = 0;
    // virtual void BackwardPass() = 0;
};

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::ConnectedBaseLayer(){};

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::ConnectedBaseLayer(const DimType &aInputDims, const DimType &aOutputDims, const UpdateMethod aUpdateMethod) : BaseLayer<DimType, DimType, DataType>(aInputDims, aOutputDims), mUpdateMethod(aUpdateMethod){};

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::InitParams(size_t aInputDim, size_t aOutputDimWeights, size_t aOutputDimBiases, double aInitVariance)
{
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, sqrt(2 / aInitVariance));
    mWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(aInputDim, aOutputDimWeights, [&]() { return vRandDistr(vRandom); });
    mBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDimBiases); //, aOutputDimBiases, [&]() { return vRandDistr(vRandom); }); // Biases should be initialized to 0 apparently.
    mLearningRate = 0.05;
    if (mUpdateMethod != UpdateMethod::VANILLA)
    {
        mMomentumUpdateWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(aInputDim, aOutputDimWeights);
        mMomentumUpdateBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDimBiases);
        mMomentumUpdateParam = 0.9;
        if (mUpdateMethod == UpdateMethod::ADAM)
        {
            mStep = 0;
            mSecondMomentumUpdateWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(aInputDim, aOutputDimWeights);
            mSecondMomentumUpdateBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDimBiases);
            mSecondMomentumUpdateParam = 0.999;
        }
    }
    this->mInitializedFlag = true;
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::UpdateParams()
{

    if (mUpdateMethod == UpdateMethod::ADAM)
    {
        // Adam
        mStep++;

        mMomentumUpdateWeights = mMomentumUpdateParam * mMomentumUpdateWeights + (1 - mMomentumUpdateParam) * mGradientsWeights;
        Eigen::Matrix<double, Dynamic, Dynamic> vCorrectedMomentumUpdateWeights = mMomentumUpdateWeights.array() / (1 - pow(mMomentumUpdateParam, mStep));
        mSecondMomentumUpdateWeights = mSecondMomentumUpdateParam * mSecondMomentumUpdateWeights.array() + (1 - mSecondMomentumUpdateParam) * mGradientsWeights.array() * mGradientsWeights.array();
        Eigen::Matrix<double, Dynamic, Dynamic> vCorrectedSecondMomentumUpdateWeights = mSecondMomentumUpdateWeights.array() / (1 - pow(mSecondMomentumUpdateParam, mStep));
        mWeights = mWeights - (mLearningRate * vCorrectedMomentumUpdateWeights.array() / (mSecondMomentumUpdateWeights.cwiseSqrt().array() + 0.00000001)).matrix();

        mMomentumUpdateBiases = mMomentumUpdateParam * mMomentumUpdateBiases + (1 - mMomentumUpdateParam) * mGradientsBiases;
        Eigen::Matrix<double, Dynamic, Dynamic> vCorrectedMomentumUpdateBiases = mMomentumUpdateBiases.array() / (1 - pow(mMomentumUpdateParam, mStep));
        mSecondMomentumUpdateBiases = mSecondMomentumUpdateParam * mSecondMomentumUpdateBiases.array() + (1 - mSecondMomentumUpdateParam) * mGradientsBiases.array() * mGradientsBiases.array();
        Eigen::Matrix<double, Dynamic, Dynamic> vCorrectedSecondMomentumUpdateBiases = mSecondMomentumUpdateBiases.array() / (1 - pow(mSecondMomentumUpdateParam, mStep));
        mBiases = mBiases - (mLearningRate * vCorrectedMomentumUpdateBiases.array() / (mSecondMomentumUpdateBiases.cwiseSqrt().array() + 0.00000001)).matrix();
    
    }
    else if (mUpdateMethod == UpdateMethod::VANILLA)
    {
        // Vanilla Descent
        mWeights = mWeights - mLearningRate * mGradientsWeights;
        mBiases = mBiases - mLearningRate * mGradientsBiases;
    }
    else
    {
        // Nesterov-Momentum
        Eigen::Matrix<DataType, Dynamic, Dynamic> vPreviousMomentumUpdateWeights = mMomentumUpdateWeights;
        mMomentumUpdateWeights = mMomentumUpdateParam * mMomentumUpdateWeights - mLearningRate * mGradientsWeights;
        mWeights = mWeights + (-mMomentumUpdateParam * vPreviousMomentumUpdateWeights) + (1 + mMomentumUpdateParam) * mMomentumUpdateWeights;
        Eigen::Matrix<DataType, Dynamic, Dynamic> vPreviousMomentumUpdateBiases = mMomentumUpdateBiases;
        mMomentumUpdateBiases = mMomentumUpdateParam * mMomentumUpdateBiases - mLearningRate * mGradientsBiases;
        mBiases = mBiases + (-mMomentumUpdateParam * vPreviousMomentumUpdateBiases) + (1 + mMomentumUpdateParam) * mMomentumUpdateBiases;
    }

    // Adam
}

#endif