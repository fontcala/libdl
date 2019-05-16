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
  MatrixXd mWeights;
  MatrixXd mBiases;
  MatrixXd mOutput;
  MatrixXd mGradients;
  std::shared_ptr<MatrixXd> mInputPtr;
  std::shared_ptr<MatrixXd> mOutputPtr;
  MatrixXd *mInPtr;
  bool mInitializedFlag = false;
  bool mValidInput = false;

public:
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim);
  FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, std::shared_ptr<MatrixXd> aInput);
  void InitParams(const size_t aInputDim, const size_t aOutputDim);
  void SetInput(const MatrixXd &aInput);
  void ForwardPass();
  void ActivationFunction(MatrixXd &mOutput);
  std::shared_ptr<MatrixXd> GetOutput() const;
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim)
{
  InitParams(aInputDim, aOutputDim);
};

FullyConnectedLayer::FullyConnectedLayer(const size_t aInputDim, const size_t aOutputDim, std::shared_ptr<MatrixXd> aInput) : mInputPtr(aInput),
                                                                                                                              mValidInput(true)
{
  InitParams(aInputDim, aOutputDim);
};

void FullyConnectedLayer::InitParams(const size_t aInputDim, const size_t aOutputDim)
{
  std::random_device rd;
  std::mt19937 vRandom(rd());
  std::normal_distribution<float> vRandDistr(0, 1.0); // TODO which distribution?
  double vParamScaleFactor = sqrt(2.0 / static_cast<double>(aInputDim));
  mWeights = vParamScaleFactor * MatrixXd::NullaryExpr(aInputDim, aOutputDim, [&]() { return vRandDistr(vRandom); });
  mBiases = vParamScaleFactor * MatrixXd::NullaryExpr(1, aOutputDim, [&]() { return vRandDistr(vRandom); });
  mInitializedFlag = true;
};

// Method for first
void FullyConnectedLayer::SetInput(const MatrixXd &aInput)
{
  const size_t vInputDim = aInput.cols();
  if (vInputDim == mWeights.rows())
  {
    mValidInput = true;
    mInputPtr = std::make_shared<MatrixXd>(aInput);
  }
  else
  {
    throw(std::runtime_error("SetInput(): dimension mismatch"));
  }
};

void FullyConnectedLayer::SetInput(const MatrixXd &aInput)
{
  const size_t vInputDim = aInput.cols();
  if (vInputDim == mWeights.rows())
  {
    mValidInput = true;
    mInputPtr = std::make_shared<MatrixXd>(aInput);
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
  aOutput = aOutput.cwiseMax(0);
};

void FullyConnectedLayer::ForwardPass()
{
  if (mInitializedFlag && mValidInput)
  {
    // TODO This can be rewritten as single matrix product.
    mOutput = (*mInputPtr) * mWeights + mBiases.replicate(mInputPtr->rows(), 1);
    std::cout << "- *mInputPtr" << std::endl;
    std::cout << (*mInputPtr) << std::endl;
    std::cout << "- *mInPtr" << std::endl;
    std::cout << (*mInPtr) << std::endl;
    std::cout << "- mWeights:" << std::endl;
    std::cout << mWeights << std::endl;
    std::cout << "- mBiases:" << std::endl;
    std::cout << mBiases << std::endl;

    // Activation Function
    ActivationFunction(mOutput);
  }
  else
  {
    throw(std::runtime_error("ForwardPass(): weights not initialized"));
  };
};

std::shared_ptr<MatrixXd> FullyConnectedLayer::GetOutput() const
{

  return mOutputPtr(&mOutput);
};

#endif