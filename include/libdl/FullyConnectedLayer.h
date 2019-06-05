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
class FullyConnectedLayer : public ConnectedBaseLayer
{
protected:
  // Layer-specific Properties
  const size_t mInputDim;
  const size_t mOutputDim;

public:
  // Constructors
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim);

  // Layer-specific Forward-Backward passes.
  void ForwardPass();
  void BackwardPass();
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim) : mInputDim(aInputDim), mOutputDim(aOutputDim)
{
  InitParams(aInputDim,aOutputDim);
};

void FullyConnectedLayer::ForwardPass()
{
  if (mInitializedFlag)
  {
    // TODO This can be rewritten as single matrix product. TODO look into coefficient-wise sum (solve with array() method)
    mOutput = (*mInputPtr) * mWeights + mBiases.replicate(mInputPtr->rows(), 1);

    //TODO Erase all this
    // std::cout << "- *mInputPtr" << std::endl;
    // std::cout << (*mInputPtr) << std::endl;
    // std::cout << "- mWeights:" << std::endl;
    // std::cout << mWeights << std::endl;
    // std::cout << "- mBiases:" << std::endl;
    // std::cout << mBiases << std::endl;
    // std::cout << "- mOutput:" << std::endl;
    // std::cout << mOutput << std::endl;

    // Activation Function
    //ActivationFunction(mOutput);

    // std::cout << "- mOutput after activation:" << std::endl;
    // std::cout << mOutput << std::endl;
  }
  else
  {
    throw(std::runtime_error("ForwardPass(): weights not initialized"));
  };
};

void FullyConnectedLayer::BackwardPass()
{
  // TODO this as a function of the input and then no need specialization.
  MatrixXd vBackpropInput = *mBackpropInputPtr;

  // times Sigmoid Derivative
  //MatrixXd vDerivatedBackPropInput = vBackpropInput.array() * (mOutput.array() * (1 - mOutput.array()));

  // times ReLu Derivative
  //MatrixXd vDerivatedBackPropInput = vBackpropInput.cwiseMax(0);

  mGradientsWeights = (*mInputPtr).transpose() * vBackpropInput;
  mGradientsBiases = vBackpropInput.colwise().sum();
  mBackpropOutput = vBackpropInput * mWeights.transpose();

  // std::cout << "- *vBackPropInput" << std::endl;
  // std::cout << vBackpropInput << std::endl;
  // std::cout << "- *vDerivatedBackPropInput" << std::endl;
  // std::cout << vDerivatedBackPropInput << std::endl;
  // std::cout << "- *mGradientsWeights" << std::endl;
  // std::cout << mGradientsWeights << std::endl;

  UpdateParams();
};
#endif