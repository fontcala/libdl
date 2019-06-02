/** @file SigmoidActivationLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef SIGMOIDACTIVATIONLAYER_H
#define SIGMOIDACTIVATIONLAYER_H

#include "BaseLayer.h"
/**
@class SigmoidActivationLayer
@brief Sigmoid Activation Layer.
 */
class SigmoidActivationLayer: public BaseLayer<MatrixXd>
{
public:
    // Constructors
    SigmoidActivationLayer();

    void ForwardPass();
    void BackwardPass();
};

SigmoidActivationLayer::SigmoidActivationLayer(){};

void SigmoidActivationLayer::ForwardPass()
{
  mOutput =  1 / (1 + exp(-1 * (*mInputPtr).array()));
};
void SigmoidActivationLayer::BackwardPass()
{
  mBackpropOutput = (*mBackpropInputPtr).array() * (mOutput.array() * (1 - mOutput.array()));
};
#endif