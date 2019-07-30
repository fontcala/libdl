#include <iostream>
#include <type_traits>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <math.h>

using Eigen::MatrixXd;
int main()
{

    MatrixXd InputVol= MatrixXd::Random(16, 3);

    MatrixXd flattened = dlfunctions::flatten(InputVol, 2);
}
