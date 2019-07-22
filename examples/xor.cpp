#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/FullyConnectedLayer.h>
#include <libdl/L2LossLayer.h>

using Eigen::MatrixXd;
int main()
{
    // inputData:   inputLabels:
    // 0 1          1
    // 1 0          1
    // 0 0          0
    // 1 1          0
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    MatrixXd inputLabels(4, 1);
    inputLabels << 1.0, 1.0, 0.0, 0.0;

    // Construct the Layers, making the simplest network able to solve the problem (note only one final node, so Softmax is not suitable.)
    FullyConnectedLayer<SigmoidActivation> firstLayer(2, 2);
    FullyConnectedLayer<SigmoidActivation> secondLayer(2, 1);
    L2LossLayer L2Layer{};

    // Connect the layers.
    secondLayer.SetInput(firstLayer.GetOutput());
    L2Layer.SetInput(secondLayer.GetOutput());
    firstLayer.SetBackpropInput(secondLayer.GetBackpropOutput());
    secondLayer.SetBackpropInput(L2Layer.GetBackpropOutput());

    firstLayer.SetLearningRate(0.05);
    secondLayer.SetLearningRate(0.05);

    // set distribution and Batch Size.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, inputData.rows() - 1);
    const int vBatchSize = 2;
    std::vector<int> vSampleIndices(vBatchSize);
    for (size_t i = 0; i < 12000; i++)
    {
        // Pick N samples form the dataset randomly
        for (size_t k = 0; k < vBatchSize; k++)
        {
            vSampleIndices[k] = dis(gen);
        }
        // Set picked Input samples and corresponding Labels
        MatrixXd inputSample = inputData(vSampleIndices, Eigen::all);
        MatrixXd inputSampleLabel = inputLabels(vSampleIndices, Eigen::all);
        firstLayer.SetData(inputSample);
        L2Layer.SetData(inputSampleLabel);

        // Forward and Backward Passes
        firstLayer.ForwardPass();
        secondLayer.ForwardPass();
        L2Layer.ForwardPass();
        L2Layer.BackwardPass();
        secondLayer.BackwardPass();
        firstLayer.BackwardPass();

        // Test with the whole data
        if (i % 100 == 0)
        {
            firstLayer.SetData(inputData);
            L2Layer.SetData(inputLabels);
            firstLayer.ForwardPass();
            secondLayer.ForwardPass();
            L2Layer.ForwardPass();
            std::cout << "----- Total Unnormalized Loss: ----" << std::endl;
            std::cout << L2Layer.GetLoss() << std::endl;
            std::cout << "----- Computed Output: ----" << std::endl;
            std::cout << *(secondLayer.GetOutput()) << std::endl;
            std::cout << "----- Total Labels: --" << std::endl;
            std::cout << inputLabels << std::endl;
        }
    }

    std::cout << "Rerun a few times (random weight initialization)" << std::endl;
}
