#include <iostream>
#include "libdl/FullyConnectedLayer.h"
#include "libdl/SigmoidActivationLayer.h"
#include "libdl/L2LossLayer.h"

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
    FullyConnectedLayer secondLayer(2, 1);
    SigmoidActivationLayer firstSigmoidLayer;
    SigmoidActivationLayer secondSigmoidLayer;
    L2LossLayer L2Layer;

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
        
        firstSigmoidLayer.SetInput(firstLayer.GetOutput());
        secondLayer.SetInput(firstSigmoidLayer.GetOutput());
        secondSigmoidLayer.SetInput(secondLayer.GetOutput());
        L2Layer.SetInput(secondSigmoidLayer.GetOutput());

        firstLayer.SetBackpropInput(firstSigmoidLayer.GetBackpropOutput());
        firstSigmoidLayer.SetBackpropInput(secondLayer.GetBackpropOutput());
        secondLayer.SetBackpropInput(secondSigmoidLayer.GetBackpropOutput());
        secondSigmoidLayer.SetBackpropInput(L2Layer.GetBackpropOutput());

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
            //std::cout << "---- firstSigmoidLayer.ForwardPass() ----" << std::endl;
            firstSigmoidLayer.ForwardPass();
            //std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
            secondLayer.ForwardPass();
            //std::cout << "---- secondSigmoidLayer.ForwardPass() ----" << std::endl;
            secondSigmoidLayer.ForwardPass();
            //std::cout << "---- secondLayer.ComputeLoss() ----" << std::endl;
            L2Layer.ForwardPass();
            //std::cout << "---- secondLayer.BackwardPass() ----" << std::endl;
            L2Layer.BackwardPass();
            //
            secondSigmoidLayer.BackwardPass();
            //
            secondLayer.BackwardPass();
            //std::cout << "---- firstSigmoidLayer.BackwardPass() ----" << std::endl;
            firstSigmoidLayer.BackwardPass();
            //std::cout << "---- firstLayer.BackwardPass() ----" << std::endl;
            firstLayer.BackwardPass();
            if (i % 100 == 0)
            {
                firstLayer.SetInput(inputData);
                L2Layer.SetLabels(inputLabels);
                firstLayer.ForwardPass();
                firstSigmoidLayer.ForwardPass();
                secondLayer.ForwardPass();
                secondSigmoidLayer.ForwardPass();
                L2Layer.ForwardPass();
                std::cout << "----- Total Loss: ----" << std::endl;
                std::cout << L2Layer.GetLoss() << std::endl;
                std::cout << "----- Total Out: -----" << std::endl;
                std::cout << *(secondSigmoidLayer.GetOutput()) << std::endl;
                std::cout << "----- Total Labels: --" << std::endl;
                std::cout << inputLabels << std::endl;
            }
        }
    }


    std::cout << "Rerun a few times (random weight initialization)" << std::endl;
}
