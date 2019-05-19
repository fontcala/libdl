/** @file hello.h
 *  @author Adria Font Calvarons
 */
#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <memory>
#include "dlfunctions.h"

/** hello function
    @param name
    @return 0
*/

class FullyConnectedLayer
{
protected:
  MatrixXd mOutput;
  MatrixXd mGradientsWeights;
  MatrixXd mGradientsBiases;
  MatrixXd mGradientsInputs;
  const MatrixXd *mInputPtr;
  const FullyConnectedLayer *mNextLayer;
  bool mInitializedFlag = false;
  bool mValidInput = false;

public:
  MatrixXd mWeights;
  MatrixXd mBiases;
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim);
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, const MatrixXd *aInput);
  void SetNext(const FullyConnectedLayer *aNext);
  void InitParams(const size_t aInputDim, const size_t aOutputDim);
  void UpdateParams();
  void SetInput(const MatrixXd &aInput);
  void ForwardPass();
  void BackwardPass();
  void ActivationFunction(MatrixXd &mOutput);
  const MatrixXd *GetOutput() const;
  const MatrixXd *GetBackpropOutput() const;
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim)
{
  InitParams(aInputDim, aOutputDim);
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, const MatrixXd *aInput) : mInputPtr(aInput), mValidInput(true)
{
  InitParams(aInputDim, aOutputDim);
}

void FullyConnectedLayer::SetNext(const FullyConnectedLayer *aNext)
{
  //TODO validity checks
  mNextLayer = aNext;
}

void FullyConnectedLayer::InitParams(const size_t aInputDim, const size_t aOutputDim)
{
  std::random_device rd;
  std::mt19937 vRandom(rd());
  std::normal_distribution<float> vRandDistr(0, 1.0); // TODO which distribution?
  double vParamScaleFactor = sqrt(1.0 / static_cast<double>(aInputDim));
  mWeights = vParamScaleFactor * MatrixXd::NullaryExpr(aInputDim, aOutputDim, [&]() { return vRandDistr(vRandom); });
  mBiases = vParamScaleFactor * MatrixXd::NullaryExpr(1, aOutputDim, [&]() { return vRandDistr(vRandom); });
  mInitializedFlag = true;
};

void FullyConnectedLayer::UpdateParams()
{
  // TODO no hardcode lrate
  const double vLearningRate = 0.3;
  mWeights = mWeights - vLearningRate * mGradientsWeights;
  mBiases = mBiases - vLearningRate * mGradientsBiases;
}

// Method for first
void FullyConnectedLayer::SetInput(const MatrixXd &aInput)
{
  const size_t vInputDim = aInput.cols();
  if (vInputDim == mWeights.rows())
  {
    mValidInput = true;
    mInputPtr = &aInput;
  }
  else
  {
    throw(std::runtime_error("SetInput(): dimension mismatch"));
  }
};

// TODO if the functional way not useful, use mOutput here directly.
void FullyConnectedLayer::ActivationFunction(MatrixXd &aOutput)
{
  //ReLu
  //aOutput = aOutput.cwiseMax(0);
  //Sigmoid
  aOutput = 1 / (1 + exp(-1 * aOutput.array()));
};

void FullyConnectedLayer::ForwardPass()
{
  if (mInitializedFlag && mValidInput)
  {
    // TODO This can be rewritten as single matrix product. TODO look into coefficient-wise sum.
    mOutput = (*mInputPtr) * mWeights + mBiases.replicate(mInputPtr->rows(), 1);

    // TODO Erase all this
    // std::cout << "- *mInputPtr" << std::endl;
    // std::cout << (*mInputPtr) << std::endl;
    // std::cout << "- mWeights:" << std::endl;
    // std::cout << mWeights << std::endl;
    // std::cout << "- mBiases:" << std::endl;
    // std::cout << mBiases << std::endl;
    // std::cout << "- mOutput:" << std::endl;
    // std::cout << mOutput << std::endl;

    // Activation Function
    ActivationFunction(mOutput);

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
  //TODO this as a function of the input and then no need specialization.
  MatrixXd vBackpropInput = (*(mNextLayer->GetBackpropOutput()));
  // std::cout << "- *vBackPropInput" << std::endl;
  // std::cout << vBackpropInput << std::endl;
  //SigmoidSpecific
  MatrixXd vDerivatedBackPropInput = vBackpropInput.array()*(mOutput.array() * (1 - mOutput.array()));
  // std::cout << "- *vDerivatedBackPropInput" << std::endl;
  // std::cout << vDerivatedBackPropInput << std::endl;
  mGradientsWeights = (*mInputPtr).transpose() * vDerivatedBackPropInput;
  // std::cout << "- *mGradientsWeights" << std::endl;
  // std::cout << mGradientsWeights << std::endl;
  mGradientsBiases = vBackpropInput.colwise().sum();
  mGradientsInputs = vBackpropInput * mWeights.transpose();
  UpdateParams();
};

const MatrixXd *FullyConnectedLayer::GetOutput() const
{
  return &mOutput;
};

const MatrixXd *FullyConnectedLayer::GetBackpropOutput() const
{
  return &mGradientsInputs;
};

#endif