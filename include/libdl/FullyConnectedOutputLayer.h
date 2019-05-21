/** @file FullyConnectedOutputLayer.h
 *  @author Adria Font Calvarons
 */

#ifndef FULLYCONNECTEDOUTPUTLAYER_H
#define FULLYCONNECTEDOUTPUTLAYER_H

#include "FullyConnectedLayer.h"

/**
 * @class FullyConnectedOutputLayer
 * @brief Class with additional notion of Loss
 *
 *
 */
class FullyConnectedOutputLayer : public FullyConnectedLayer
{
private:
  double mLoss;
  MatrixXd mGradientLoss;

public:
  using FullyConnectedLayer::FullyConnectedLayer;

  //void ActivationFunction(MatrixXd &mOutput);
  void ComputeLoss(const MatrixXd &aLabels);
  void BackwardPass();
  double GetLoss() const;
};

// void FullyConnectedOutputLayer::ActivationFunction(MatrixXd &aOutput)
// {
//   // Constant
//   aOutput = aOutput;
// };
void FullyConnectedOutputLayer::ComputeLoss(const MatrixXd &aLabels)
{
  const size_t vLabelNum = aLabels.rows();
  const size_t vOutputNum = mOutput.rows();
  if (vLabelNum == vOutputNum)
  {
    //L2
    // sigmoid from
    mGradientLoss = mOutput - aLabels;
    mLoss = 0.5 * mGradientLoss.squaredNorm();
  }
  else
  {
    throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
  }
};

void FullyConnectedOutputLayer::BackwardPass()
{
  //sigmoid specific (dNorm/dsigmoid)*(dsigmoid/dmGradientLoss)
  MatrixXd vGradientSigmoid = mGradientLoss;
  ActivationFunction(vGradientSigmoid);
  MatrixXd vGradientBackprop = mGradientLoss.array() * vGradientSigmoid.array() * (1 - vGradientSigmoid.array());

  mGradientsWeights = (*mInputPtr).transpose() * vGradientBackprop;

  mGradientsBiases = vGradientBackprop.colwise().sum();

  mGradientsInputs = vGradientBackprop * mWeights.transpose();

  // std::cout << "- vGradientBackprop" << std::endl;
  // std::cout << vGradientBackprop << std::endl;
  // std::cout << "- (*mInputPtr).transpose()" << std::endl;
  // std::cout << (*mInputPtr).transpose() << std::endl;
  // std::cout << "- mGradientsWeights" << std::endl;
  // std::cout << mGradientsWeights << std::endl;
  // std::cout << "- mGradientsBiases" << std::endl;
  // std::cout << mGradientsBiases << std::endl;
  // std::cout << "- mGradientInputs" << std::endl;
  // std::cout << mGradientsInputs << std::endl;

  UpdateParams();
};

double FullyConnectedOutputLayer::GetLoss() const
{
  return mLoss;
};

#endif