#include <iostream>
#include <libdl/NetworkHelper.h>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/FlattenLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/SoftmaxLossLayer.h>
#include <libdl/L2LossLayer.h>
#include <libdl/FullyConnectedLayer.h>

using Eigen::MatrixXd;
int main()
{
    MatrixXd A = MatrixXd::Random(10,3);
    MatrixXd B = MatrixXd::Random(5,3);

    MatrixXd C = (A * B.transpose());

    MatrixXd D = (B.transpose() * A);

    std::cout << C << std::endl;
    std::cout << "---------" << std::endl;
    std::cout << D << std::endl;

    
}
