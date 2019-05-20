//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include "catch2/catch.hpp"
//#include "libdl/Layer.h"

//TEST_CASE( "first  tests", "[includes]" ) {
//REQUIRE( 0 == 0 );
//}
#include <iostream>
#include "libdl/FullyConnectedLayer.h"
#include "libdl/FullyConnectedOutputLayer.h"

int main()
{
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    MatrixXd inputLabels(4, 1);
    inputLabels << 1.0, 1.0, 0.0, 0.0;

    FullyConnectedLayer firstLayer(2, 2);
    FullyConnectedOutputLayer secondLayer(2, 1);
    firstLayer.SetBackpropInput(secondLayer.GetBackpropOutput());
    secondLayer.SetInput(firstLayer.GetOutput());

    firstLayer.mLearningRate = 0.2;
    secondLayer.mLearningRate = 0.2;

    if (true)
    {
        // set input subset
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, inputData.rows() - 1);
        const int vBatchSize = 2;
        std::vector<int> vSampleIndices(vBatchSize);
        for (size_t i = 0; i < 23001; i++)
        {
            for (size_t k = 0; k < vBatchSize; k++)
            {
                vSampleIndices[k] = dis(gen);
            }

            MatrixXd inputSample = inputData(vSampleIndices, Eigen::all);
            MatrixXd inputSampleLabel = inputLabels(vSampleIndices, Eigen::all);
            //std::cout << inputSample << std::endl;
            //std::cout << inputSampleLabel << std::endl;

            //std::cout << "---- firstLayer.SetInput ----" << std::endl;
            firstLayer.SetInput(inputSample);
            //std::cout << "---- firstLayer.ForwardPass() ----" << std::endl;
            firstLayer.ForwardPass();
            //std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
            secondLayer.ForwardPass();
            secondLayer.ComputeLoss(inputSampleLabel);
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
