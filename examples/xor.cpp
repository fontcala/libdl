#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/FullyConnectedLayer.h>
#include <libdl/L2LossLayer.h>

using Eigen::MatrixXd;
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
    FullyConnectedLayer<SigmoidActivation> firstLayer(2, 2);
    FullyConnectedLayer<SigmoidActivation> secondLayer(2, 1);
    L2LossLayer L2Layer{};

    // // OPTIONAL Weights and biases made Public in case you want to set your own weights.
    // bool vManualSetWeights = false;
    // if (vManualSetWeights)
    // {
    //     MatrixXd vTrueWeightsFirst(2, 2);
    //     MatrixXd vTrueWeightsSecond(2, 1);
    //     MatrixXd vTrueBiasesFirst(1, 2);
    //     MatrixXd vTrueBiasesSecond(1, 1);
    //     vTrueWeightsFirst << -2.0, 1.0, -2.0, 1.0;
    //     vTrueWeightsSecond << 1.08, 1.1;
    //     vTrueBiasesFirst << 3.0, -0.5;
    //     vTrueBiasesSecond << -1.5;
    //     firstLayer.mWeights = vTrueWeightsFirst;
    //     firstLayer.mBiases = vTrueBiasesFirst;
    //     secondLayer.mWeights = vTrueWeightsSecond;
    //     secondLayer.mBiases = vTrueBiasesSecond;
    // }

    // Construct network and train with parameters that seem to work.
    // Since we are overfitting a dataset, no need for predicting a test set.
    // Train method prints performance on the whole dataset (inputData,inputLabels) to see the evolution.
    bool vManualCompute = true;

    if (!vManualCompute)
    {
        // FullyConnectedNetwork vNetwork221({&firstLayer,
        //                                    &secondLayer});
        // vNetwork221.ConnectLayers();
        // vNetwork221.Train(inputData, inputLabels, 0.05, 12001,false);
    }
    else
    {
        

        secondLayer.SetInput(firstLayer.GetOutput());
        L2Layer.SetInput(secondLayer.GetOutput());

        firstLayer.SetBackpropInput(secondLayer.GetBackpropOutput());
        secondLayer.SetBackpropInput(L2Layer.GetBackpropOutput());

        firstLayer.mLearningRate = 0.05;
        secondLayer.mLearningRate = 0.05;
        // set input subset
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, inputData.rows() - 1);
        const int vBatchSize = 2;
        std::vector<int> vSampleIndices(vBatchSize);
        for (size_t i = 0; i < 24001; i++)
        {
            for (size_t k = 0; k < vBatchSize; k++)
            {
                vSampleIndices[k] = dis(gen);
            }

            MatrixXd inputSample = inputData(vSampleIndices, Eigen::all);
            MatrixXd inputSampleLabel = inputLabels(vSampleIndices, Eigen::all);
            //std::cout << "---- firstLayer.SetInput ----" << std::endl;
            firstLayer.SetInput(inputData);
            L2Layer.SetLabels(inputLabels);
            //std::cout << "---- firstLayer.ForwardPass() ----" << std::endl;
            firstLayer.ForwardPass();
            //std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
            secondLayer.ForwardPass();
            //std::cout << "---- secondLayer.ComputeLoss() ----" << std::endl;
            L2Layer.ForwardPass();
            //std::cout << "---- L2Layer.ComputeLoss() ----" << std::endl;
            L2Layer.BackwardPass();
            //std::cout << "---- secondLayer.BackwardPass() ----" << std::endl;
            secondLayer.BackwardPass();
            //std::cout << "---- firstLayer.BackwardPass() ----" << std::endl;
            firstLayer.BackwardPass();
            if (i % 100 == 0)
            {
                firstLayer.SetInput(inputData);
                L2Layer.SetLabels(inputLabels);
                firstLayer.ForwardPass();
                secondLayer.ForwardPass();
                L2Layer.ForwardPass();
                std::cout << "----- Total Loss: ----" << std::endl;
                std::cout << L2Layer.GetLoss() << std::endl;
                std::cout << "----- Total Labels: --" << std::endl;
                std::cout << inputLabels << std::endl;
            }
        }
    }


    std::cout << "Rerun a few times (random weight initialization)" << std::endl;
}
