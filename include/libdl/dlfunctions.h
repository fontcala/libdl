#ifndef DLFUNCTIONS_H
#define DLFUNCTIONS_H

#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <Eigen/Core>
//#include <spdlog/spdlog.h>

// Question, is it a good idea to have for instance a header with activation functions and Loss functions and then eiter:
// - pass these functions to the layer constructor.
// - use these functions as template parameters of templated class layer.

using Eigen::MatrixXd;

namespace dlfunctions
{
void ReLUActivationFunction(MatrixXd &aOutput)
{
    aOutput = aOutput.cwiseMax(0);
}
void ConstantActivationFunction(MatrixXd &aOutput)
{
    aOutput = aOutput;
}
} // namespace dlfunctions

#endif