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
@brief Fully connected layer for dense layer elements.
 */
template <template <typename> class ActivationFunctionType, typename DataType = double>
class FullyConnectedLayer final : public ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>
{
public:
  // Constructors
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

  // Layer-specific Forward-Backward passes.
  void ForwardPass() override;
  void BackwardPass() override;
};

template <template <typename> class ActivationFunctionType, typename DataType>
FullyConnectedLayer<ActivationFunctionType, DataType>::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<size_t, ActivationFunctionType, DataType>(aInputDim, aOutputDim, aUpdateMethod)
{
  this->InitParams(aInputDim, aOutputDim, aOutputDim, aInputDim);
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedLayer<ActivationFunctionType, DataType>::ForwardPass()
{
  if (this->mValidInputFlag)
  {
    // TODO This can be rewritten as single matrix product. TODO look into coefficient-wise sum (solve with array() method)
    (this->mOutput) = *(this->mInputPtr) * this->mWeights + this->mBiases.replicate((this->mInputPtr)->rows(), 1);
    this->ActivationFunction.ForwardFunction(this->mOutput);

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
  }
  else
  {
    throw(std::runtime_error("ForwardPass(): invalid input"));
  };
};

template <template <typename> class ActivationFunctionType, typename DataType>
void FullyConnectedLayer<ActivationFunctionType, DataType>::BackwardPass()
{

  if (this->mValidBackpropInputFlag)
  {
    Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);
    this->ActivationFunction.BackwardFunction(vBackpropInput);
    this->mGradientsWeights = this->mInputPtr->transpose() * vBackpropInput;
    this->mGradientsBiases = vBackpropInput.colwise().sum();
    this->mBackpropOutput = vBackpropInput * this->mWeights.transpose();

    //TODO Erase all this
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
  }
  else
  {
    throw(std::runtime_error("BackwardPass(): invalid input"));
  };
};
#endif