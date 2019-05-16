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

  void ActivationFunction(MatrixXd &mOutput);
  void ComputeLoss(const MatrixXd &aLabels);
  double GetLoss() const;
};

void FullyConnectedOutputLayer::ActivationFunction(MatrixXd &aOutput)
{
  // Constant
  aOutput = aOutput;
};
void FullyConnectedOutputLayer::ComputeLoss(const MatrixXd &aLabels)
{
  const size_t vLabelNum = aLabels.rows();
  const size_t vOutputNum = mOutput.rows();
  if (vLabelNum == vOutputNum)
  {
    //L2
    mGradientVar = mOutput - aLabels;
    mLoss = 0.5 * mGradientVar.squaredNorm();
  }
  else{
    throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
  }
};
double FullyConnectedOutputLayer::GetLoss() const
{
  return mLoss;
};

#endif