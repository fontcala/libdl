/** @file FullyConnectedLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <memory>
#include "dlfunctions.h"

/**
@class FullyConnectedLayer
@brief Base Class for Network Layer elements.

@note Weight Initialization random but factor and distribution still hardcoded.
@note Activation Function still hardcoded.
@note Loss Function still hardcoded.
@note Gradient Update still hardcoded.
 */
class FullyConnectedLayer
{
protected:
  // Properties
  const size_t mInputDim;
  const size_t mOutputDim;
  // Data
  MatrixXd mOutput;
  MatrixXd mGradientsWeights;
  MatrixXd mGradientsBiases;
  MatrixXd mGradientsInputs;
  void ActivationFunction(MatrixXd &mOutput);

  // Readonly data from other layers
  const MatrixXd *mInputPtr;
  const MatrixXd *mBackpropInputPtr;

  // Checkers
  bool mInitializedFlag = false;
  bool mValidInput = false;

public:
  // Weights to be modified often.
  MatrixXd mWeights;
  MatrixXd mBiases;
  MatrixXd mMomentumUpdateWeights;
  MatrixXd mMomentumUpdateBiases;
  // Learning Rate to be modified often
  double mLearningRate;
  double mMomentumUpdateParam;
  // Constructors
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim);
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, const MatrixXd *aInput);

  // Processing
  /**
  @function InitParams
  @brief Initialization with <tt>std::mt19937</tt> so that every run is with a different set of weights and biases.
 */
  void InitParams();
  void UpdateParams();
  void ForwardPass();
  virtual void BackwardPass();

  // Virtual Dummy Methods
  virtual void ComputeLoss(const MatrixXd &aLabels);
  virtual double GetLoss() const;

  // Helpers to connect Layers
  void SetNext(const FullyConnectedLayer *aNext);
  void SetInput(const MatrixXd &aInput);
  void SetInput(const MatrixXd *aInput);
  void SetBackpropInput(const MatrixXd *aOutput);
  const MatrixXd *GetOutput() const;
  const MatrixXd *GetBackpropOutput() const;
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim) : mInputDim(aInputDim), mOutputDim(aOutputDim), mLearningRate(0.3), mInputPtr(NULL), mBackpropInputPtr(NULL)
{
  InitParams();
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, const MatrixXd *aInput) : mInputDim(aInputDim), mOutputDim(aOutputDim), mLearningRate(0.3), mInputPtr(aInput), mBackpropInputPtr(NULL)
{
  InitParams();
}

void FullyConnectedLayer::InitParams()
{
  std::random_device rd;
  std::mt19937 vRandom(rd());
  std::normal_distribution<float> vRandDistr(0, 1.0); // TODO which distribution?
  double vParamScaleFactor = sqrt(1.0 / static_cast<double>(mInputDim));
  mWeights = vParamScaleFactor * MatrixXd::NullaryExpr(mInputDim, mOutputDim, [&]() { return vRandDistr(vRandom); });
  mBiases = vParamScaleFactor * MatrixXd::NullaryExpr(1, mOutputDim, [&]() { return vRandDistr(vRandom); });
  mMomentumUpdateWeights = MatrixXd::Zero(mInputDim, mOutputDim);
  mMomentumUpdateBiases = MatrixXd::Zero(1, mOutputDim);
  mMomentumUpdateParam = 0.9;
  mInitializedFlag = true;
};

void FullyConnectedLayer::UpdateParams()
{
  // TODO User specified
  // Nesterov-Momentum
  MatrixXd vPreviousMomentumUpdateWeights = mMomentumUpdateWeights;
  mMomentumUpdateWeights =  mMomentumUpdateParam * mMomentumUpdateWeights - mLearningRate * mGradientsWeights;
  mWeights = mWeights + (-mMomentumUpdateParam * vPreviousMomentumUpdateWeights) + (1 + mMomentumUpdateParam)* mMomentumUpdateWeights;

  MatrixXd vPreviousMomentumUpdateBiases = mMomentumUpdateBiases;
  mMomentumUpdateBiases =  mMomentumUpdateParam * mMomentumUpdateBiases - mLearningRate * mGradientsBiases;
  mBiases = mBiases + (-mMomentumUpdateParam * vPreviousMomentumUpdateBiases) + (1 + mMomentumUpdateParam)* mMomentumUpdateBiases;

  // normal Gradient Descent
  // mWeights = mWeights - mLearningRate * mGradientsWeights;
  // mBiases = mBiases - mLearningRate * mGradientsBiases;
}

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
  // TODO this as a function of the input and then no need specialization.
  MatrixXd vBackpropInput = *mBackpropInputPtr;
  //Sigmoid Derivative
  MatrixXd vDerivatedBackPropInput = vBackpropInput.array() * (mOutput.array() * (1 - mOutput.array()));

  //MatrixXd vDerivatedBackPropInput = MatrixXd::Ones(vBackpropInput.size());
  //MatrixXd vDerivatedBackPropInputHelper = vBackpropInput.cwiseMax(0);

  mGradientsWeights = (*mInputPtr).transpose() * vDerivatedBackPropInput;
  mGradientsBiases = vDerivatedBackPropInput.colwise().sum();
  mGradientsInputs = vDerivatedBackPropInput * mWeights.transpose();

  // std::cout << "- *mGradientsWeights" << std::endl;
  // std::cout << mGradientsWeights << std::endl;
  // std::cout << "- *vDerivatedBackPropInput" << std::endl;
  // std::cout << vDerivatedBackPropInput << std::endl;
  // std::cout << "- *vBackPropInput" << std::endl;
  // std::cout << vBackpropInput << std::endl;

  UpdateParams();
};

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

void FullyConnectedLayer::SetInput(const MatrixXd *aInput)
{
  mInputPtr = aInput;
};

void FullyConnectedLayer::SetBackpropInput(const MatrixXd *aBackpropInput)
{
  mBackpropInputPtr = aBackpropInput;
};

const MatrixXd *FullyConnectedLayer::GetOutput() const
{
  return &mOutput;
};

const MatrixXd *FullyConnectedLayer::GetBackpropOutput() const
{
  return &mGradientsInputs;
};

void FullyConnectedLayer::ComputeLoss(const MatrixXd &aLabels)
{
}
double FullyConnectedLayer::GetLoss() const
{
}
#endif