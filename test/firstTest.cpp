//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include "catch2/catch.hpp"
//#include "libdl/Layer.h"

//TEST_CASE( "first  tests", "[includes]" ) {
//REQUIRE( 0 == 0 );
//}
#include <iostream>
#include "libdl/FullyConnectedNetwork.h"

int main()
{
    bool vManualSetWeights = false;
    bool vManualCompute = false;
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    MatrixXd inputLabels(4, 1);
    inputLabels << 1.0, 1.0, 0.0, 0.0;

    FullyConnectedLayer firstLayer(2, 2);
    FullyConnectedLayer hiddenLayer1(5,5);
    FullyConnectedLayer hiddenLayer2(5,2);
    FullyConnectedOutputLayer secondLayer(2, 1);

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

    if (!vManualCompute)
    {
        FullyConnectedNetwork vNetwork221({&firstLayer,
                                           &secondLayer});
        vNetwork221.ConnectLayers();
        vNetwork221.Train(inputData, inputLabels, 0.05, 12001,false);
    }
    else
    {
        firstLayer.SetBackpropInput(secondLayer.GetBackpropOutput());
        secondLayer.SetInput(firstLayer.GetOutput());

        firstLayer.mLearningRate = 0.003;
        secondLayer.mLearningRate = 0.003;
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
            //std::cout << "---- firstLayer.ForwardPass() ----" << std::endl;
            firstLayer.ForwardPass();
            //std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
            secondLayer.ForwardPass();
            //std::cout << "---- secondLayer.ComputeLoss() ----" << std::endl;
            secondLayer.ComputeLoss(inputLabels);
            //std::cout << "---- secondLayer.BackwardPass() ----" << std::endl;
            secondLayer.BackwardPass();
            //std::cout << "---- firstLayer.BackwardPass() ----" << std::endl;
            firstLayer.BackwardPass();
            if (i % 100 == 0)
            {
                firstLayer.SetInput(inputData);
                firstLayer.ForwardPass();
                secondLayer.ForwardPass();
                secondLayer.ComputeLoss(inputLabels);
                std::cout << "----- Total Loss: ----" << std::endl;
                std::cout << secondLayer.GetLoss() << std::endl;
                std::cout << "----- Total Out: -----" << std::endl;
                std::cout << *(secondLayer.GetOutput()) << std::endl;
                std::cout << "----- Total Labels: --" << std::endl;
                std::cout << inputLabels << std::endl;
            }
        }
    }
}
