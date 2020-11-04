/** @file FullyConnectedExperimentalLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FullyConnectedExperimentalLayer_H
#define FullyConnectedExperimentalLayer_H

#include <memory>
#include "dlfunctions.h"
#include "ConnectedBaseLayer.h"

/**
* @class FullyConnectedExperimentalLayer
* @brief Fully connected layer for dense layer elements.
*
* This class encapsulates the forward backward computations needed for this kind of layer, implemented in matrix form. For further information see \c ForwardPass and \c BackwardPass. 
*/
template <template <typename> class ActivationFunctionType, typename DataType = double>
class FullyConnectedExperimentalLayer final : public ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>
{
public:

  Eigen::Matrix<DataType, Dynamic, Dynamic> mMaxAbsMomentum;
  Eigen::Matrix<DataType, Dynamic, Dynamic> mMomentumMultipliers;
  // Constructors
  FullyConnectedExperimentalLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

  // Layer-specific Forward-Backward passes.

  /**
    * FullyConnectedExperimentalLayer::ForwardPass
    * overrides 
    * @copydoc NetworkElement::ForwardPass
    * 
    * This additionally makes the output go through the activation function specified by the \c ActivationFunctionType template parameter.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    * @remark Could be done as a simple matrix multiplication by appending ones to the input matrix and putting the bias and weights together. 
    * However, it seems more elegant to keep weights and biases separated, since for other types of layers I cannot join them together.
    * In the meantime,  \c replicate() is deemed well enough to include the biases, alternatively it would be possible to use the colwise/rowwise operator.
    */
  void ForwardPass() override;

  /**
    * FullyConnectedExperimentalLayer::BackwardPass
    * overrides 
    * @copydoc NetworkElement::BackwardPass
    * 
    * This additionally updates the parameters of the layer.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    */
  void BackwardPass() override;
};

template <template <typename> class ActivationFunctionType, typename DataType>
FullyConnectedExperimentalLayer<ActivationFunctionType, DataType>::FullyConnectedExperimentalLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>(aInputDim, aOutputDim, aUpdateMethod)
{
  this->InitParams(aInputDim, aOutputDim, aOutputDim, aInputDim);

  std::random_device rd;
  std::mt19937 vRandom(rd());
  std::normal_distribution<float> vRandDistr(0, sqrt(2 / aInputDim));
  mMaxAbsMomentum = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(aInputDim,aOutputDim, [&]() { return vRandDistr(vRandom); });
  std::cout << mMaxAbsMomentum << std::endl;
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedExperimentalLayer<ActivationFunctionType, DataType>::ForwardPass()
{
  if (this->mValidInputFlag)
  {
    // Multiplication and addition
    (this->mOutput) = *(this->mInputPtr) * this->mWeights + this->mBiases.replicate((this->mInputPtr)->rows(), 1);
    // Activation Function
    this->ActivationFunction.ForwardFunction(this->mOutput);  
  }
  else
  {
    throw(std::runtime_error("ForwardPass(): invalid input"));
  };
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedExperimentalLayer<ActivationFunctionType, DataType>::BackwardPass()
{
  //std::cout << "momentum" << this->mMomentumUpdateWeights << std::endl;
  //std::cout << "weights"  << this->mWeights << std::endl;
  if (this->mValidBackpropInputFlag)
  {
    Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);
    this->ActivationFunction.BackwardFunction(vBackpropInput);
    this->mGradientsWeights = this->mInputPtr->transpose() * vBackpropInput;
    this->mGradientsBiases = vBackpropInput.colwise().sum();

    if (!this->mIsFirstLayerFlag)
    {
      this->mBackpropOutput = vBackpropInput * this->mWeights.transpose();
    }

    // Update Parameters
    this->UpdateParams();
  }
  else
  {
    throw(std::runtime_error("BackwardPass(): invalid input"));
  };
};
#endif