#include <iostream>
#include <type_traits>
#include <libdl/dlfunctions.h>

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
    const size_t tFILTERSIZE = 2; // Both sides
    size_t vNumChannelsVol = 3;
    size_t vImageHeightVol = 4;
    size_t vImageWidthVol = 4;
    size_t vStrideVol = 2;
    size_t vPaddingVol = 0;
    size_t vOutHeightVol = (vImageHeightVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
    size_t vOutWidthVol = (vImageWidthVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
    size_t vOutFieldsVol = tFILTERSIZE * tFILTERSIZE;
    size_t vNumSamples = 1;
    MatrixXd OutputVol(vOutHeightVol * vOutWidthVol, vNumChannelsVol);
    for (size_t vChannelIndex = 0; vChannelIndex < vNumChannelsVol; vChannelIndex++)
    {
        OutputVol.col(vChannelIndex) = dlfunctions::im2pool(tFILTERSIZE, InputVol.col(vChannelIndex).data(), vOutHeightVol, vOutWidthVol, vImageHeightVol, vImageWidthVol, vStrideVol, vNumSamples);
    }
    std::cout << "OutputVol" << std::endl;
    std::cout << OutputVol << std::endl;

    // Backward Pass
    MatrixXd BackpropOutputSingleVol(vOutHeightVol * vOutWidthVol, vOutFieldsVol);
    MatrixXd BackpropOutputAll(vOutHeightVol * vOutWidthVol, vOutFieldsVol * vNumChannelsVol);
    for (size_t vChannelIndex = 0; vChannelIndex < vNumChannelsVol; vChannelIndex++)
    {
        dlfunctions::im2colpool(tFILTERSIZE, InputVol.col(vChannelIndex).data(), BackpropOutputSingleVol.data(), vOutHeightVol, vOutWidthVol, vImageHeightVol, vImageWidthVol, vStrideVol, vNumSamples);
        Eigen::Matrix<double, Dynamic, Dynamic> someMat = BackpropOutputSingleVol.colwise() - BackpropOutputSingleVol.rowwise().maxCoeff();
        Eigen::Matrix<bool, Dynamic, Dynamic> someMatBool = someMat.cwiseAbs().array() < std::numeric_limits<double>::epsilon();
        Eigen::Matrix<double, Dynamic, Dynamic> backRel = (OutputVol.col(vChannelIndex).replicate(1, vOutFieldsVol));
        BackpropOutputAll.block(0, vOutFieldsVol * vChannelIndex, vOutHeightVol * vOutWidthVol, vOutFieldsVol) = backRel.array() * someMatBool.cast<double>().array();
    }
    std::cout << "BackpropOutputAll" << std::endl;
    std::cout << BackpropOutputAll << std::endl;
    MatrixXd BackpropVol(16, 3);
    size_t vBackFieldsVol = tFILTERSIZE * tFILTERSIZE * vNumChannelsVol;
    dlfunctions::colpool2im(tFILTERSIZE, BackpropOutputAll.data(), BackpropVol.data(), vOutHeightVol, vOutWidthVol, vBackFieldsVol, vImageHeightVol, vImageWidthVol, vStrideVol, vNumSamples);
    std::cout << "BackpropVol" << std::endl;
    std::cout << BackpropVol << std::endl;
    // Eigen::Matrix<double, Dynamic, Dynamic> vDerivative = (InputVol1.colwise() - maxVal);
    // std::cout << "vDerivative" << std::endl;
    // std::cout << vDerivative<< std::endl;
    // Eigen::Matrix<bool, Dynamic, Dynamic> vDerivativeBool = vDerivative.cwiseAbs().array() <= std::numeric_limits<double>::epsilon();
    // std::cout << "vDerivativeBool" << std::endl;
    // std::cout << vDerivativeBool<< std::endl;
}
