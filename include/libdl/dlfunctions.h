#ifndef DLFUNCTIONS_H
#define DLFUNCTIONS_H

#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <Eigen/Core>
//#include <spdlog/spdlog.h>

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