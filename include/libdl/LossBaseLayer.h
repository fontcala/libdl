/** @file LossBaseLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef LOSSBASELAYER_H
#define LOSSBASELAYER_H

#include "BaseLayer.h"
/**
* @class LossBaseLayer
* @brief Base Layer for Loss function classes, introduces label inputs and getters.
* 
* Each Loss Layer deriving from this class should find an appropriate way to return an output
*  apart from the loss that is meaningful to the learning task. (Eg: class probabilites).
*
* Additionally, in this layer (last layer) there is a check of the dimensions between the labels and the resulting output
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
    // Every Layer must implement these
    // virtual void ForwardPass() = 0;
    // virtual void BackwardPass() = 0;
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