#include <iostream>
#include <type_traits>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <math.h>

using Eigen::MatrixXd;
int main()
{
  ConvDataDims A(3,4,5);
  ConvDataDims B(3,4,7);
  std::cout << (A == B) << std::endl;
}
