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

    FullyConnectedLayer firstLayer(2, 2, &inputData);
    FullyConnectedOutputLayer secondLayer(2, 1, firstLayer.GetOutput());
    firstLayer.SetNext(&secondLayer);
    firstLayer.SetInput(inputData);

    // MatrixXd vTrueWeightsFirst(2, 2);
    // MatrixXd vTrueWeightsSecond(2, 1);
    // MatrixXd vTrueBiasesFirst(1, 2);
    // MatrixXd vTrueBiasesSecond(1, 1);
    // vTrueWeightsFirst << -2.0, 1.0, -2.0, 1.0;
    // vTrueWeightsSecond << 1.08, 1.1;
    // vTrueBiasesFirst << 3.0, -0.5;
    // vTrueBiasesSecond << -1.5;

    // firstLayer.mWeights = vTrueWeightsFirst;
    // firstLayer.mBiases = vTrueBiasesFirst;
    // secondLayer.mWeights = vTrueWeightsSecond;
    // secondLayer.mBiases = vTrueBiasesSecond;

    if (true)
    {
        for (size_t i = 0; i < 16000; i++)
        {
            //std::cout << "---- firstLayer.ForwardPass() ----" << std::endl;
            firstLayer.ForwardPass();
            //std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
            secondLayer.ForwardPass();

            secondLayer.ComputeLoss(inputLabels);
            if (i % 100 == 0)
            {
                std::cout << "----- Loss: ----" << std::endl;
                std::cout << secondLayer.GetLoss() << std::endl;
                std::cout << "----- Out: -----" << std::endl;
                std::cout << *(secondLayer.GetOutput()) << std::endl;
            }

            //std::cout << "---- secondLayer.BackwardPass() ----" << std::endl;
            secondLayer.BackwardPass();
            //std::cout << "---- firstLayer.BackwardPass() ----" << std::endl;
            firstLayer.BackwardPass();
        }
        std::cout << "----- Labels: --" << std::endl;
        std::cout << inputLabels << std::endl;
    }
}
