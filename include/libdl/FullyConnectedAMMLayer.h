/** @file FullyConnectedAMMLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FULLYCONNECTEDAMMLAYER_H
#define FULLYCONNECTEDAMMLAYER_H

#include <memory>
#include "dlfunctions.h"
#include "ConnectedBaseLayer.h"

/**
* @class FullyConnectedAMMLayer
* @brief Fully connected layer for dense layer elements with approximate matrix multiplication
*
* This class encapsulates the forward backward computations needed for this kind of layer, implemented in matrix form. For further information see \c ForwardPass and \c BackwardPass. 
*/
template <template <typename> class ActivationFunctionType, typename DataType = double>
class FullyConnectedAMMLayer final : public ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>
{
public:
  // AMM Parameters.
  double mAMMFractionForward;
  double mAMMFractionBackward;
  // Constructors
  FullyConnectedAMMLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

  // Layer-specific Forward-Backward passes.

  /**
    * FullyConnectedAMMLayer::ForwardPass
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
    * FullyConnectedAMMLayer::BackwardPass
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
FullyConnectedAMMLayer<ActivationFunctionType, DataType>::FullyConnectedAMMLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>(aInputDim, aOutputDim, aUpdateMethod)
{
  mAMMFractionForward = 0.7;
  mAMMFractionBackward = 0.3;
  this->InitParams(aInputDim, aOutputDim, aOutputDim, aInputDim);
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedAMMLayer<ActivationFunctionType, DataType>::ForwardPass()
{
  if (this->mValidInputFlag)
  {
    // Multiplication and addition
    Eigen::Matrix<DataType, Dynamic, Dynamic> intermediateAMM = dlfunctions::topkAMM(*(this->mInputPtr),this->mWeights,mAMMFractionForward);
    //Eigen::Matrix<DataType, Dynamic, Dynamic> intermediateAMM = *(this->mInputPtr) * this->mWeights;
    this->mOutput = intermediateAMM + this->mBiases.replicate((this->mInputPtr)->rows(), 1);
    // Activation Function
    this->ActivationFunction.ForwardFunction(this->mOutput);
  }
  else
  {
    throw(std::runtime_error("ForwardPass(): invalid input"));
  };
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedAMMLayer<ActivationFunctionType, DataType>::BackwardPass()
{

  if (this->mValidBackpropInputFlag)
  {
    Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);
    this->ActivationFunction.BackwardFunction(vBackpropInput);
    this->mGradientsWeights = dlfunctions::topkAMM(static_cast<Eigen::Matrix<DataType, Dynamic, Dynamic>>(this->mInputPtr->transpose()),vBackpropInput,mAMMFractionBackward);
    //this->mGradientsWeights = this->mInputPtr->transpose()* vBackpropInput;
    this->mGradientsBiases = vBackpropInput.colwise().sum();

    if (!this->mIsFirstLayerFlag)
    {
      this->mBackpropOutput = dlfunctions::topkAMM(static_cast<Eigen::Matrix<DataType, Dynamic, Dynamic>>(vBackpropInput),static_cast<Eigen::Matrix<DataType, Dynamic, Dynamic>>(this->mWeights.transpose()),mAMMFractionBackward);
      //this->mBackpropOutput = vBackpropInput * this->mWeights.transpose();
    }

    //Update Parameters
    this->UpdateParams();
  }
  else
  {
    throw(std::runtime_error("BackwardPass(): invalid input"));
  };
};
#endif