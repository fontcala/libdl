#include <iostream>
#include <type_traits>
#include <libdl/dlfunctions.h>
#include <libdl/MaxPoolLayer.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
int main()
{
    // This works
    // MatrixXd InputVol1(4, 4);
    // MatrixXd InputVol2(4, 4);
    // MatrixXd InputVol3(4, 4);
    // MatrixXd InputVol(4, 4 * 3);
    // InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 330, 311, 312, 313, 314, 315, 316;
    // InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
    // InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 211, 212, 213, 214, 215, 216;
    // InputVol << InputVol1, InputVol2, InputVol3;
    // std::cout << "InputVol" << std::endl;
    // std::cout << InputVol << std::endl;
    // std::cout << "InputVol" << std::endl;
    // std::cout << InputVol1 << std::endl;
    // // Forward Pass
    // VectorXd maxVal = InputVol1.rowwise().maxCoeff();
    // std::cout << "Maxval" << std::endl;
    // std::cout << maxVal << std::endl;
    // // Backward Pass
    // Eigen::Matrix<double, Dynamic, Dynamic> vDerivative = (InputVol1.colwise() - maxVal);
    // std::cout << "vDerivative" << std::endl;
    // std::cout << vDerivative<< std::endl;
    // Eigen::Matrix<bool, Dynamic, Dynamic> vDerivativeBool = vDerivative.cwiseAbs().array() <= std::numeric_limits<double>::epsilon();
    // std::cout << "vDerivativeBool" << std::endl;
    // std::cout << vDerivativeBool<< std::endl;

    // Input
    MatrixXd InputVol1(16, 1);
    MatrixXd InputVol2(16, 1);
    MatrixXd InputVol3(16, 1);
    InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316;
    InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
    InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216;
    MatrixXd InputVol(16, 3);
    InputVol << InputVol1, InputVol2, InputVol3;
    std::cout << "InputVol" << std::endl;
    std::cout << InputVol << std::endl;

    // Parameters
    const size_t tFILTERSIZE = 2;
    size_t vNumChannelsVol = 3;
    size_t vImageHeightVol = 4;
    size_t vImageWidthVol = 4;
    size_t vStrideVol = 2;
    size_t vPaddingVol = 0;
    size_t vOutHeightVol = (vImageHeightVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
    size_t vOutWidthVol = (vImageWidthVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
    size_t vOutFieldsVol = tFILTERSIZE * tFILTERSIZE;
    size_t vNumSamples = 1;

    MaxPoolLayer maxP(vNumChannelsVol,vImageHeightVol,vImageWidthVol,tFILTERSIZE,vStrideVol,vNumSamples);
    maxP.SetInput(InputVol);
    maxP.ForwardPass();
    std::cout << "*(maxP.GetOutput())"<< std::endl;
    std::cout << *(maxP.GetOutput())<< std::endl;
    maxP.SetBackpropInput(maxP.GetOutput());
    maxP.BackwardPass();
    std::cout << "*(maxP.GetBackpropOutput())"<< std::endl;
    std::cout << *(maxP.GetBackpropOutput())<< std::endl;

 
}
