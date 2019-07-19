/** @file NetworkHelper.h
 *  @author Adria Font Calvarons
 */
#ifndef NETWORKHELPER_H
#define NETWORKHELPER_H

#include "BaseLayer.h"
#include "NetworkElement.h"

/**
* @class NetworkHelper
* @brief Class facilitates use of layers.
*
* Example use (given previously constructed layers):
* @code
NetworkHelper vNetworkExample({&firstLayer,
                                    &hiddenLayer1,
                                    &hiddenLayer2,
                                    &outputLayer});
* @endcode
*/
template <typename DataType = double>
class NetworkHelper
{
private:
    std::vector<NetworkElement<DataType> *> mNetwork;
    bool mValidNetwork;
    size_t mNumberLayers;

public:
    double mLearningRate;
    void FullForwardPass();
    void FullBackwardPass();
    Eigen::Matrix<DataType, Dynamic, Dynamic> FullForwardTestPass();
    /**
    * NetworkHelper
    * @note input @code const std::initializer_list<NetworkElement<DataType> *> &
    * @endcode 
    * must contain lvalue <tt>NetworkElement<DataType> *</tt> objects.
    */
    NetworkHelper(const std::initializer_list<NetworkElement<DataType> *> &aLayers);
    void ConnectLayers();
    void SetInputData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
    void SetLabelData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aLabels);
};
template <class DataType>
NetworkHelper<DataType>::NetworkHelper(const std::initializer_list<NetworkElement<DataType> *> &aLayers) : mNetwork(aLayers), mValidNetwork(false), mNumberLayers(aLayers.size())
{
    ConnectLayers();
}

template <class DataType>
void NetworkHelper<DataType>::ConnectLayers()
{
    for (size_t i = 1; i < mNumberLayers; i++)
    {
        mNetwork[i]->SetInput(mNetwork[i - 1]->GetOutput());
        mNetwork[i - 1]->SetBackpropInput(mNetwork[i]->GetBackpropOutput());
    }
    mValidNetwork = true;
}
template <class DataType>
void NetworkHelper<DataType>::FullBackwardPass()
{
    for (size_t vProcess = mNumberLayers; vProcess > 0; vProcess--)
    {
        mNetwork[vProcess - 1]->BackwardPass();
    }
}
template <class DataType>
void NetworkHelper<DataType>::FullForwardPass()
{
    for (size_t vProcess = 0; vProcess < mNumberLayers; vProcess++)
    {
        mNetwork[vProcess]->ForwardPass();
    }
}
template <class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> NetworkHelper<DataType>::FullForwardTestPass()
{
    for (size_t vProcess = 0; vProcess < mNumberLayers - 1; vProcess++)
    {
        mNetwork[vProcess]->ForwardPass();
    }
    return *(mNetwork[mNumberLayers - 2]->GetOutput());
}

template <class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> NetworkHelper<DataType>::SetInputData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    mNetwork.front()->SetData(aInput)
}
template <class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> NetworkHelper<DataType>::SetLabelData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aLabels)
{
    mNetwork.back()->SetData(aLabels)
}

#endif