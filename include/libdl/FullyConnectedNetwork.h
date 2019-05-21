/** @file FullyConnectedNetwork.h
 *  @author Adria Font Calvarons
 */
#ifndef FULLYCONNECTEDNETWORK_H
#define FULLYCONNECTEDNETWORK_H

#include "FullyConnectedLayer.h"
#include "FullyConnectedOutputLayer.h"

/**
@class FullyConnectedNetwork
@brief Class facilitates use of layers.
@code
FullyConnectedNetwork vNetworkExample({&firstLayer,
                                    &hiddenLayer1,
                                    &hiddenLayer2,
                                    &outputLayer});
vNetworkExample.ConnectLayers();
vNetworkExample.Train(inputData, inputLabels, 0.3, 4, 24001);
@endcode
@note Weight Initialization random but factor and distribution still hardcoded.
@note Activation Function still hardcoded.
@note Loss Function still hardcoded.
@note Gradient Update still hardcoded.
Example use (given previously constructed layers):
*/
class FullyConnectedNetwork
{
private:
    std::vector<FullyConnectedLayer *> mNetwork;
    bool mValidNetwork;
    size_t mNumberLayers;
    void FullForwardPass();
    void FullBackwardPass(const MatrixXd &aInputSampleLabel);

public:
    double mLearningRate;
    /**
    @function FullyConnectedNetwork
    @note input @code const std::initializer_list<FullyConnectedLayer *> &
    @endcode 
    must contain lvalue <tt>FullyConnectedLayer</tt> objects.
    */
    FullyConnectedNetwork(const std::initializer_list<FullyConnectedLayer *> &aLayers);
    void ConnectLayers();
    void Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const double aBatchsize, const double aIters);
    void Predict(const MatrixXd &aInput, const MatrixXd &aLabels);
};
FullyConnectedNetwork::FullyConnectedNetwork(const std::initializer_list<FullyConnectedLayer *> &aLayers) : mNetwork(aLayers), mValidNetwork(false), mNumberLayers(aLayers.size())
{
}
// TODO check right dimensions
// TODO check network more than one layer
// TODO check last layer output layer
// TODO Then set mValidNetwork true
void FullyConnectedNetwork::ConnectLayers()
{
    for (size_t i = 1; i < mNumberLayers; i++)
    {
        mNetwork[i]->SetInput(mNetwork[i - 1]->GetOutput());
        mNetwork[i - 1]->SetBackpropInput(mNetwork[i]->GetBackpropOutput());
    }
    mValidNetwork = true;
}
void FullyConnectedNetwork::FullBackwardPass(const MatrixXd &aInputSampleLabel)
{
    mNetwork.back()->ComputeLoss(aInputSampleLabel);
    for (int vProcess = mNumberLayers; vProcess > 0; vProcess--)
    {
        mNetwork[vProcess - 1]->BackwardPass();
    }
}
void FullyConnectedNetwork::FullForwardPass()
{
    for (int vProcess = 0; vProcess < mNumberLayers; vProcess++)
    {
        mNetwork[vProcess]->ForwardPass();
    }
}

void FullyConnectedNetwork::Train(const MatrixXd &aInput, const MatrixXd &aLabels, const double aLearningRate, const double aBatchSize, const double aIters)
{
    if (mValidNetwork)
    {
        for (auto vNetworkLayerPtr : mNetwork)
        {
            vNetworkLayerPtr->mLearningRate = aLearningRate;
        }
        // Generate Random Subset
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, aInput.rows() - 1);
        std::vector<int> vSampleIndices(aBatchSize);

        for (size_t vIter = 0; vIter < aIters; vIter++)
        {
            // Set Random subset as input
            for (size_t k = 0; k < aBatchSize; k++)
            {
                vSampleIndices[k] = dis(gen);
            }

            MatrixXd inputSample = aInput(vSampleIndices, Eigen::all);
            MatrixXd inputSampleLabel = aLabels(vSampleIndices, Eigen::all);
            mNetwork[0]->SetInput(aInput);
            FullForwardPass();
            FullBackwardPass(aLabels);
            if (vIter % 100 == 0)
            {
                std::cout << "----- Iter: " << vIter << std::endl;
                Predict(aInput, aLabels);
            }
        }
    }
    else
    {
        throw(std::runtime_error("Train(): invalid Network"));
    }
}
void FullyConnectedNetwork::Predict(const MatrixXd &aInput, const MatrixXd &aLabels)
{
    mNetwork[0]->SetInput(aInput);
    FullForwardPass();
    mNetwork.back()->ComputeLoss(aLabels);
    std::cout << "----- Predict Loss: ----" << std::endl;
    std::cout << mNetwork.back()->GetLoss() << std::endl;
    std::cout << "----- Predict Out: -----" << std::endl;
    std::cout << *(mNetwork.back()->GetOutput()) << std::endl;
    std::cout << "----- Predict Labels: --" << std::endl;
    std::cout << aLabels << std::endl;
}

#endif