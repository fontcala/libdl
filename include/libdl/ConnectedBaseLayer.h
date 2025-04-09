/** @file ConnectedBaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONNECTEDBASELAYER_H
#define CONNECTEDBASELAYER_H

#include "BaseLayer.h"
#include <math.h>

/**
* UpdateMethod
* @brief Encapsulates Optimization methods.
* 
* currently available are:
* 
* - NESTEROV (sgd with nesterov momentum)  
* - VANILLA (basic gradient descent)  
* - ADAM (adam method annealing learning rate)  
*/
enum class UpdateMethod
{
    NESTEROV,
    VANILLA,
    ADAM
};

/**
* @class ConnectedBaseLayer
* @brief Base Layer for classes with parameters, introduces the parameters, their update and initialization.
* 
* 
* 
* Weight initialization is He/Xavier style. Choice left to the child classes.
* 
* In the case of convolutional-related images, Weights of sizes (x,y,z) each are stored in 2D matrices of size (x * y * z, N). This minimizes reshapes needed.
* 
* Parameter Update method (choice left to child classes) offers options from enum \c:
* @copydetails UpdateMethod
*/
template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
class ConnectedBaseLayer : public BaseLayer<DimType, DimType, DataType>
{
protected:
    bool mInitializedFlag = false;

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
    DataType mStep;
    DataType mLearningRate;
    DataType mMomentumUpdateParam;
    DataType mSecondMomentumUpdateParam;

    void InitUpdateParams(size_t aInputDim, size_t aOutputDimWeights, size_t aOutputDimBiases, DataType aLearningRate, DataType aMomentumUpdateParam, DataType aSecondMomentumUpdateParam);

public:
    // Processing
    /**
    * @brief Parameter Initialization and variance control.
    * 
    * Initialization with <tt>std::mt19937</tt> so that every run is with a different set of weights (using eigen's random is not random enough). According to cs231n biases are better initialized to 0
    *
    * He/Xavier Initialization dependent on layer type with the parameter \c aInitVariance to control the variance of weights.
    * 
    * Sets \c mInitializedFlag to \c true, which conditions the parameter update
    */
    void InitParams(size_t aInputDim, size_t aOutputDimWeights, size_t aOutputDimBiases, double aInitVariance, DataType aLearningRate = 0.005, DataType aMomentumUpdateParam = 0.9, DataType aSecondMomentumUpdateParam = 0.999);
    void UpdateParams();

    /**
    * @brief Initialization of parameters with a user defined value (eg: Pretrained weights).
    * 
    * Sets \c mInitializedFlag to \c true, which conditions the parameter update
    * @warning The user is responsible to ensure the matrix dimensions in the input match the ones expected given the layer parameters.
    */
    void SetCustomParams(Eigen::Matrix<DataType, Dynamic, Dynamic> aInWeights, Eigen::Matrix<DataType, Dynamic, Dynamic> aInBiases, DataType aLearningRate = 0.005, DataType aMomentumUpdateParam = 0.9, DataType aSecondMomentumUpdateParam = 0.999);
    void SetLearningParams(DataType aLearningRate = 0.005, DataType aMomentumUpdateParam = 0.9, DataType aSecondMomentumUpdateParam = 0.999);
    void SetLearningRate(DataType aLearningRate);
    // Constructors
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
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::InitParams(size_t aInputDim, size_t aOutputDimWeights, size_t aOutputDimBiases, double aInitVariance, DataType aLearningRate, DataType aMomentumUpdateParam, DataType aSecondMomentumUpdateParam)
{
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, sqrt(static_cast<double>(2) / static_cast<double>(aInitVariance)));
    mWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(aInputDim, aOutputDimWeights, [&]() { return vRandDistr(vRandom); });
    mBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDimBiases);
    InitUpdateParams(aInputDim, aOutputDimWeights, aOutputDimBiases, aLearningRate, aMomentumUpdateParam, aSecondMomentumUpdateParam);
    mInitializedFlag = true;
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::InitUpdateParams(size_t aInputDim, size_t aOutputDimWeights, size_t aOutputDimBiases, DataType aLearningRate, DataType aMomentumUpdateParam, DataType aSecondMomentumUpdateParam)
{
    mLearningRate = aLearningRate;
    if (mUpdateMethod != UpdateMethod::VANILLA)
    {
        mMomentumUpdateWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(aInputDim, aOutputDimWeights);
        mMomentumUpdateBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDimBiases);
        mMomentumUpdateParam = aMomentumUpdateParam;
        if (mUpdateMethod == UpdateMethod::ADAM)
        {
            mStep = 0;
            mSecondMomentumUpdateWeights = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(aInputDim, aOutputDimWeights);
            mSecondMomentumUpdateBiases = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(1, aOutputDimBiases);
            mSecondMomentumUpdateParam = aSecondMomentumUpdateParam;
        }
    }
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::UpdateParams()
{
    if (mInitializedFlag)
    {
        if (mUpdateMethod == UpdateMethod::ADAM)
        {
            // Adam
            mStep++;

            auto vPreWeights = mWeights; 
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
    }
    else
    {
        throw(std::runtime_error("UpdateParams(): parameters not initialized"));
    }
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::SetCustomParams(Eigen::Matrix<DataType, Dynamic, Dynamic> aInWeights, Eigen::Matrix<DataType, Dynamic, Dynamic> aInBiases, DataType aLearningRate, DataType aMomentumUpdateParam, DataType aSecondMomentumUpdateParam)
{
    mWeights = aInWeights;
    mBiases = aInBiases;
    const size_t vInputDim = mWeights.rows();
    const size_t vOutputDimWeights = mWeights.cols();
    const size_t vOutputDimBiases = aInBiases.cols();
    InitUpdateParams(vInputDim, vOutputDimWeights, vOutputDimBiases, aLearningRate, 0.9, 0.999);
    mInitializedFlag = true;
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::SetLearningParams(DataType aLearningRate, DataType aMomentumUpdateParam, DataType aSecondMomentumUpdateParam)
{
    SetLearningRate(aLearningRate);
    mMomentumUpdateParam = aMomentumUpdateParam;
    mSecondMomentumUpdateParam = aSecondMomentumUpdateParam;
}

template <typename DimType, template <typename> class ActivationFunctionType, typename DataType>
void ConnectedBaseLayer<DimType, ActivationFunctionType, DataType>::SetLearningRate(DataType aLearningRate)
{
    mLearningRate = aLearningRate;
}

#endif