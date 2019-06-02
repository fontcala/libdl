#include <iostream>
#include "libdl/FullyConnectedNetwork.h"

int main()
{
    // MatrixXd inputData(4, 2);
    // inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    // MatrixXd inputLabels(4, 1);
    // inputLabels << 1.0, 1.0, 0.0, 0.0;
    // // in:   labels:
    // // 0 1    1
    // // 1 0    1
    // // 0 0    0
    // // 1 1    0

    // // Construct the Layers
    // FullyConnectedLayer firstLayer(2, 2);
    // FullyConnectedLayer firstHiddenLayer(2, 5);
    // FullyConnectedLayer secondHiddenLayer(5, 5);
    // FullyConnectedLayer thirdHiddenLayer(5, 2);
    // FullyConnectedOutputLayer finalLayer(2, 1);

    // // Construct network and train with parameters that seem to work.
    // // Since we are overfitting a dataset, no need for predicting a test set.
    // // Train method prints performance on the whole dataset (inputData,inputLabels) to see the evolution.
    // FullyConnectedNetwork vNetwork221({&firstLayer,
    //                                    &firstHiddenLayer,
    //                                    &secondHiddenLayer,
    //                                    &thirdHiddenLayer,
    //                                    &finalLayer});
    // vNetwork221.ConnectLayers();
    // vNetwork221.Train(inputData, inputLabels, 0.05, 12001, false);

    // std::cout << "Rerun a few times (random weight initialization)" << std::endl;
}
