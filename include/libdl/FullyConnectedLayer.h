/** @file FullyConnectedLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <memory>
#include "dlfunctions.h"
#include "ConnectedBaseLayer.h"

/**
@class FullyConnectedLayer
@brief Base Class for Network Layer elements.

@note Weight Initialization random but factor and distribution still hardcoded.
@note Activation Function still hardcoded.
@note Loss Function still hardcoded.
@note Gradient Update still hardcoded.
 */
template <template <typename> class ActivationFunctionType, typename DataType = double>
class FullyConnectedLayer : public ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>
{
public:
  // Constructors
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim);

  // Layer-specific Forward-Backward passes.
  void ForwardPass();
  void BackwardPass();
};

template <template <typename> class ActivationFunctionType, typename DataType>
FullyConnectedLayer<ActivationFunctionType, DataType>::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim) : ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>(aInputDim, aOutputDim)
{
  this->InitParams(aInputDim, aOutputDim, aInputDim);
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedLayer<ActivationFunctionType, DataType>::ForwardPass()
{
  if (this->mInitializedFlag)
  {
    // TODO This can be rewritten as single matrix product. TODO look into coefficient-wise sum (solve with array() method)
    (this->mOutput) = *(this->mInputPtr) * this->mWeights + this->mBiases.replicate((this->mInputPtr)->rows(), 1);
    this->ActivationFunction.Activate(this->mOutput);
    //TODO Erase all this
    // std::cout << "- (*mInputPtr)" << std::endl;
    // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols()<< std::endl;
    // std::cout << "(this->mOutput)" << std::endl;
    // std::cout << (this->mOutput).rows() << " " << (this->mOutput).cols() << std::endl;
    // std::cout << "- mWeights:" << std::endl;
    // std::cout << mWeights << std::endl;
    // std::cout << "- mBiases:" << std::endl;
    // std::cout << mBiases << std::endl;
    // std::cout << "- (this->mOutput):" << std::endl;
    // std::cout << (this->mOutput) << std::endl;

    // Activation Function
    //ActivationFunction((this->mOutput));

    // std::cout << "- (this->mOutput) after activation:" << std::endl;
    // std::cout << (this->mOutput) << std::endl;
  }
  else
  {
    throw(std::runtime_error("ForwardPass(): weights not initialized (FullyConnectedLayer)"));
  };
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedLayer<ActivationFunctionType, DataType>::BackwardPass()
{
  Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);
  this->ActivationFunction.Backpropagate(vBackpropInput);
  // std::cout << "- vBackpropOutput" << std::endl;
  // std::cout << vBackpropInput.rows() << " " << vBackpropInput.cols()<< std::endl;
  // times Sigmoid Derivative
  //DataType vDerivatedBackPropInput = vBackpropInput.array() * ((this->mOutput).array() * (1 - (this->mOutput).array()));

  // times ReLu Derivative
  //DataType vDerivatedBackPropInput = vBackpropInput.cwiseMax(0);
  this->mGradientsWeights = this->mInputPtr->transpose() * vBackpropInput;
  this->mGradientsBiases = vBackpropInput.colwise().sum();
  this->mBackpropOutput = vBackpropInput * this->mWeights.transpose();

  // std::cout << "- mBackpropOutput" << std::endl;
  // std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols()<< std::endl;
  // std::cout << "(*mInputPtr)" << std::endl;
  // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols()<< std::endl;
  // std::cout << "- *vBackPropInput" << std::endl;
  // std::cout << vBackpropInput << std::endl;
  // std::cout << "- *vDerivatedBackPropInput" << std::endl;
  // std::cout << vDerivatedBackPropInput << std::endl;
  // std::cout << "- *mGradientsWeights" << std::endl;
  // std::cout << mGradientsWeights << std::endl;

  this->UpdateParams();
};
#endif