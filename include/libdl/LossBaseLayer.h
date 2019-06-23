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
template <class DataType>
class LossBaseLayer : public BaseLayer<size_t,size_t, DataType>
{ 
protected:
    DataType mLoss;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mLabels;

public:
    // Constructor
    LossBaseLayer();
    // Assume one-hot encoding Labels
    void SetLabels(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aLabels);
    double GetLoss() const;
    // TODO method that checks validity
    // TODO Do something about GetOutput
    // Every Layer must implement these
    virtual void ForwardPass() = 0;
    virtual void BackwardPass() = 0;
};

template <class DataType>
LossBaseLayer<DataType>::LossBaseLayer(){};

template <class DataType>
void LossBaseLayer<DataType>::SetLabels(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aLabels)
{
    mLabels = aLabels;
}

template <class DataType>
double LossBaseLayer<DataType>::GetLoss() const
{
  return mLoss;
};

#endif