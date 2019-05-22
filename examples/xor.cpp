#include <iostream>
#include "libdl/FullyConnectedNetwork.h"

int main()
{
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    MatrixXd inputLabels(4, 1);
    inputLabels << 1.0, 1.0, 0.0, 0.0;
    // in:   labels:
    // 0 1    1
    // 1 0    1
    // 0 0    0
    // 1 1    0

    // Construct the Layers
    FullyConnectedLayer firstLayer(2, 2);
    FullyConnectedOutputLayer secondLayer(2, 1);

    // OPTIONAL Weights and biases made Public in case you want to set your own weights.
    bool vManualSetWeights = false;
    if (vManualSetWeights)
    {
        MatrixXd vTrueWeightsFirst(2, 2);
        MatrixXd vTrueWeightsSecond(2, 1);
        MatrixXd vTrueBiasesFirst(1, 2);
        MatrixXd vTrueBiasesSecond(1, 1);
        vTrueWeightsFirst << -2.0, 1.0, -2.0, 1.0;
        vTrueWeightsSecond << 1.08, 1.1;
        vTrueBiasesFirst << 3.0, -0.5;
        vTrueBiasesSecond << -1.5;
        firstLayer.mWeights = vTrueWeightsFirst;
        firstLayer.mBiases = vTrueBiasesFirst;
        secondLayer.mWeights = vTrueWeightsSecond;
        secondLayer.mBiases = vTrueBiasesSecond;
    }

    // Construct network and train with parameters that seem to work.
    // Since we are overfitting a dataset, no need for predicting a test set.
    // Train method prints performance on the whole dataset (inputData,inputLabels) to see the evolution.
    FullyConnectedNetwork vNetwork221({&firstLayer,
                                       &secondLayer});
    vNetwork221.ConnectLayers();
    vNetwork221.Train(inputData, inputLabels, 0.05, 12001, false);

    std::cout << "Rerun a few times (random weight initialization)" << std::endl;
}
