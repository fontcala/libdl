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
    inputData << 0.1, 1, 0.6, 0.5, 0.5, 1, 0.1, 0.5;
    MatrixXd inputLabels(4, 1);
    inputLabels << 1, 1, 0, 0;

    FullyConnectedLayer firstLayer(2, 2);

    FullyConnectedOutputLayer secondLayer(2, 1, firstLayer.GetOutput());

    // firstLayer.SetInput(inputData);
    // std::cout << "---- firstLayer.ForwardPass() ----" << std::endl;
    // firstLayer.ForwardPass();
    // std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
    // secondLayer.ForwardPass();

    // std::cout << "---- Loss: ----" << std::endl;
    // secondLayer.ComputeLoss(inputLabels);
    // std::cout << secondLayer.GetLoss() << std::endl;

    std::shared_ptr<MatrixXd> mInputPtr = std::make_shared<MatrixXd>(inputData);
    firstLayer.SetInput(mInputPtr);
    std::cout << "---- firstLayer.ForwardPass() ----" << std::endl;
    firstLayer.ForwardPass();
    std::cout << "---- secondLayer.ForwardPass() ----" << std::endl;
    secondLayer.ForwardPass();

    std::cout << "---- Loss: ----" << std::endl;
    secondLayer.ComputeLoss(inputLabels);
    std::cout << secondLayer.GetLoss() << std::endl;
}
