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
class LossBaseLayer : public BaseLayer<size_t, size_t, DataType>
{
protected:

    /**
    * mLossNormalizationFactor defines how to scale the loss value and is a user choice (default value = 1)
    */
    const double mLossNormalizationFactor;
    DataType mLoss;
    Eigen::Matrix<DataType, Dynamic, Dynamic> mLabels;

    bool ValidData() const;

public:
    // Constructor
    LossBaseLayer(double aLossNormalizationFactor = 1.0);
    
    /**
    * LossLayer::SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
    * overrides
    * @copydoc BaseLayer::SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
    * 
    * In this case, labels are the expected Input.
    */
    void SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aLabels) override;

    /**
    * Loss normalized by mLossNormalizationFactor.
    */
    double GetLoss() const;
};

template <class DataType>
LossBaseLayer<DataType>::LossBaseLayer(double aLossNormalizationFactor):mLossNormalizationFactor(aLossNormalizationFactor){};

template <class DataType>
void LossBaseLayer<DataType>::SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aLabels)
{
    mLabels = aLabels;
}

template <class DataType>
double LossBaseLayer<DataType>::GetLoss() const
{
    return mLoss;
};

template <class DataType>
bool LossBaseLayer<DataType>::ValidData() const
{
    bool vValidData = false;
    if (this->mValidInputFlag)
    {

        if (this->mLabels.rows() == this->mInputPtr->rows() && this->mLabels.cols() == this->mInputPtr->cols())
        {
            vValidData = true;
        }
    }
    else
    {
        throw(std::runtime_error("ValidData(): invalid input (flag)"));
    };
    return vValidData;
};

#endif