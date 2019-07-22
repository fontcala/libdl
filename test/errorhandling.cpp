#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/ConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/L2LossLayer.h>
#include <libdl/NetworkHelper.h>
#include <libdl/FullyConnectedLayer.h>
#include <exception>

using Eigen::MatrixXd;

TEST_CASE("Forgetting some forward layer connection should throw a recognizable error", "layers")
{
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;

    // Construct the Layers
    FullyConnectedLayer<SigmoidActivation> firstLayer(2, 2);
    FullyConnectedLayer<SigmoidActivation> secondLayer(2, 1);

    firstLayer.SetData(inputData);
    firstLayer.ForwardPass();
    const std::string cExpectedErrorMessage = "ForwardPass(): invalid input";
    try{
        secondLayer.ForwardPass();
    }
    catch (std::exception& e)
    {
        std::string vErrorMessage(e.what());
        REQUIRE(cExpectedErrorMessage == vErrorMessage);
    }
}


TEST_CASE("Forgetting some backward layer connection should throw a recognizable error", "layers")
{
    MatrixXd vBackpropInputData(4, 1);
    vBackpropInputData << 0.0, 1.0, 1.0, 0.0;

    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;

    // Construct the Layers
    FullyConnectedLayer<SigmoidActivation> firstLayer(2, 2);
    FullyConnectedLayer<SigmoidActivation> secondLayer(2, 1);

    firstLayer.SetData(inputData);
    secondLayer.SetInput(&inputData);
    firstLayer.ForwardPass();
    secondLayer.ForwardPass();
    secondLayer.SetBackpropInput(&vBackpropInputData);
    std::cout << "here" << std::endl;
    secondLayer.BackwardPass();
    const std::string cExpectedErrorMessage = "BackwardPass(): invalid input";
    std::cout << "here2" << std::endl;
    try{
        firstLayer.BackwardPass();
    }
    catch (std::exception& e)
    {
        std::string vErrorMessage(e.what());
        REQUIRE(cExpectedErrorMessage == vErrorMessage);
    }
}

TEST_CASE("mismatch Input/Labels should throw a recognizable error", "layers")
{
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    MatrixXd inputLabels(3, 1);
    inputLabels << 1.0, 1.0, 0.0;

    // Construct the Layers, making the simplest network able to solve the problem (note only one final node, so Softmax is not suitable.)
    FullyConnectedLayer<SigmoidActivation> firstLayer(2, 1);
    L2LossLayer L2Layer{};

    // Connect the layers.
    L2Layer.SetInput(firstLayer.GetOutput());
    firstLayer.SetBackpropInput(L2Layer.GetBackpropOutput());

    firstLayer.SetData(inputData);
    L2Layer.SetData(inputLabels);
    const std::string cExpectedErrorMessage = "L2LossLayer::ForwardPass(): dimension mismatch (may be caused by wrong label input or upsampling part of the network not matching the size of the downsampling part.)";

    firstLayer.ForwardPass();
    try{
        L2Layer.ForwardPass();
    }
    catch (std::exception& e)
    {
        std::string vErrorMessage(e.what());
        REQUIRE(cExpectedErrorMessage == vErrorMessage);
    }
}