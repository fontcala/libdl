/** @file FullyConnectedSignedAlignmentLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FULLYCONNECTEDSIGNEDALIGNMENTLAYER_H
#define FULLYCONNECTEDSIGNEDALIGNMENTLAYER_H

#include <memory>
#include "dlfunctions.h"
#include "ConnectedBaseLayer.h"

/**
* @class FullyConnectedSignedAlignmentLayer
* @brief Fully connected layer for dense layer elements.
*
* This class encapsulates the forward backward computations needed for this kind of layer, implemented in matrix form. For further information see \c ForwardPass and \c BackwardPass. 
*/
template <template <typename> class ActivationFunctionType, typename DataType = double>
class FullyConnectedSignedAlignmentLayer final : public ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>
{
public:
  
  Eigen::Matrix<DataType, Dynamic, Dynamic> mRandomWeightMatrix;
  // Constructors
  FullyConnectedSignedAlignmentLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

  // Layer-specific Forward-Backward passes.

  /**
    * FullyConnectedSignedAlignmentLayer::ForwardPass
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
    * FullyConnectedSignedAlignmentLayer::BackwardPass
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
FullyConnectedSignedAlignmentLayer<ActivationFunctionType, DataType>::FullyConnectedSignedAlignmentLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>(aInputDim, aOutputDim, aUpdateMethod)
{
  this->InitParams(aInputDim, aOutputDim, aOutputDim, aInputDim);
  // Generate Random Weight Matrix
  //mRandomWeightMatrix  = this->mWeights;
  std::random_device rd;
  std::mt19937 vRandom(rd());
  std::normal_distribution<float> vRandDistr(0, sqrt(static_cast<double>(2) / static_cast<double>(aInputDim)));
  mRandomWeightMatrix = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(aInputDim,aOutputDim, [&]() { return vRandDistr(vRandom); });
  mRandomWeightMatrix = mRandomWeightMatrix.cwiseAbs();
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedSignedAlignmentLayer<ActivationFunctionType, DataType>::ForwardPass()
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
void FullyConnectedSignedAlignmentLayer<ActivationFunctionType, DataType>::BackwardPass()
{

  if (this->mValidBackpropInputFlag)
  {
    Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);
    this->ActivationFunction.BackwardFunction(vBackpropInput);
    this->mGradientsWeights = this->mInputPtr->transpose() * vBackpropInput;
    this->mGradientsBiases = vBackpropInput.colwise().sum();

    if (!this->mIsFirstLayerFlag)
    {
      Eigen::Matrix<DataType, Dynamic, Dynamic> vWeightSign = this->mWeights.array().sign(); // Recover the sign of the weight matrix
      mRandomWeightMatrix.cwiseProduct(vWeightSign);
      this->mBackpropOutput = vBackpropInput * mRandomWeightMatrix.transpose();
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