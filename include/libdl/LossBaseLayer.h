/** @file LossBaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef LOSSBASELAYER_H
#define LOSSBASELAYER_H

#include "BaseLayer.h"
/**
@class LossBaseLayer
@brief L2 Loss Layer.
 */
class LossBaseLayer : public BaseLayer<MatrixXd>
{ 
protected:
    double mLoss;
    MatrixXd mLabels;

public:
    // Constructor
    LossBaseLayer();
    // Assume one-hot encoding Labels
    void SetLabels(const MatrixXd &aLabels);
    double GetLoss() const;
    // TODO method that checks validity
    // TODO Do something about GetOutput
    // Every Layer must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;
};

LossBaseLayer::LossBaseLayer(){};

void LossBaseLayer::SetLabels(const MatrixXd &aLabels)
{
    mLabels = aLabels;
}

double LossBaseLayer::GetLoss() const
{
  return mLoss;
};

#endif