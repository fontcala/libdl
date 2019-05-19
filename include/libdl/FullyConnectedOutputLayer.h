/** @file hello.h
 *  @author Adria Font Calvarons
 */

#ifndef FULLYCONNECTEDOUTPUTLAYER_H
#define FULLYCONNECTEDOUTPUTLAYER_H

#include "FullyConnectedLayer.h"

class FullyConnectedOutputLayer : public FullyConnectedLayer
{
private:
  double mLoss;
  MatrixXd mGradientVar;

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
    mGradientVar = mOutput - aLabels;
    mLoss = 0.5 * mGradientVar.squaredNorm();
  }
  else
  {
    throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
  }
};

void FullyConnectedOutputLayer::BackwardPass()
{
  //sigmoid specific (dNorm/dsigmoid)*(dsigmoid/dmGradientVar)
  MatrixXd vGradientSigmoid =  mGradientVar;
  ActivationFunction(vGradientSigmoid);
  MatrixXd vGradientBackprop = mGradientVar.array() * vGradientSigmoid.array() * (1 - vGradientSigmoid.array());
  // std::cout << "- vGradientBackprop" << std::endl;
  // std::cout << vGradientBackprop << std::endl;
  // std::cout << "- (*mInputPtr).transpose()" << std::endl;
  // std::cout << (*mInputPtr).transpose() << std::endl;
  mGradientsWeights = (*mInputPtr).transpose() * vGradientBackprop;
  // std::cout << "- mGradientsWeights" << std::endl;
  // std::cout << mGradientsWeights << std::endl;
  mGradientsBiases = vGradientBackprop.colwise().sum();
  // std::cout << "- mGradientsBiases" << std::endl;
  // std::cout << mGradientsBiases << std::endl;
  mGradientsInputs = vGradientBackprop * mWeights.transpose();
  // std::cout << "- mGradientInputs" << std::endl;
  // std::cout << mGradientsInputs << std::endl;
  UpdateParams();
};

double FullyConnectedOutputLayer::GetLoss() const
{
  return mLoss;
};

#endif