#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <type_traits>
#include "libdl/dlfunctions.h"

// TODO make scenarios for different stride, sizes etc...
using Eigen::MatrixXd;

TEST_CASE("im2col with 2D Data and hardcoded example", "image im2col")
{

    // Template parameter vFilterSize = 2
    MatrixXd Input(4, 4);
    Input << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    std::cout << "Input" << std::endl;
    std::cout << Input << std::endl;

    size_t vNumChannels = 1;
    size_t vImageHeight = Input.rows();
    size_t vImageWidth = Input.cols() / vNumChannels;
    size_t vStride = 2;
    size_t vOutHeight = (vImageHeight - 2) / vStride + 1;
    size_t vOutWidth = (vImageWidth - 2) / vStride + 1;
    MatrixXd Output(vOutHeight * vOutWidth, 2 * 2 * vNumChannels);
    dlfunctions::im2col<2, 2>(&Output, &Input, vStride, vNumChannels, vImageHeight, vImageWidth);

    MatrixXd vExpectedOutput(vOutHeight * vOutWidth, 2 * 2 * vNumChannels);
    vExpectedOutput << 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16;

    REQUIRE(Output == vExpectedOutput);
}

TEST_CASE("im2com with 3D Data and hardcoded example", "image im2col")
{

    // Template parameter vFilterSize = 2
    MatrixXd InputVol1(4, 4);
    MatrixXd InputVol2(4, 4);
    MatrixXd InputVol3(4, 4);
    MatrixXd InputVol(4, 4 * 3);
    InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 330, 311, 312, 313, 314, 315, 316;
    InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
    InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 211, 212, 213, 214, 215, 216;
    InputVol << InputVol1, InputVol2, InputVol3;
    std::cout << "InputVol" << std::endl;
    std::cout << InputVol << std::endl;

    size_t vNumChannelsVol = 3;
    size_t vImageHeightVol = InputVol.rows();
    size_t vImageWidthVol = InputVol.cols() / vNumChannelsVol;
    size_t vStrideVol = 2;
    size_t vOutHeightVol = (vImageHeightVol - 2) / vStrideVol + 1;
    size_t vOutWidthVol = (vImageWidthVol - 2) / vStrideVol + 1;
    MatrixXd OutputVol(vOutHeightVol * vOutWidthVol, 2 * 2 * vNumChannelsVol);
    dlfunctions::im2col<2, 2>(&OutputVol, &InputVol, vStrideVol, vNumChannelsVol, vImageHeightVol, vImageWidthVol);

    MatrixXd vExpectedOutputVol(vOutHeightVol * vOutWidthVol, 2 * 2 * vNumChannelsVol);
    vExpectedOutputVol << 101, 102, 105, 106, 201, 202, 205, 206, 301, 302, 305, 306, 103, 104, 107, 108, 203, 204, 207, 208, 303, 304, 307, 308, 109, 110, 113, 114, 209, 220, 213, 214, 309, 330, 313, 314, 111, 112, 115, 116, 211, 212, 215, 216, 311, 312, 315, 316;

    REQUIRE(OutputVol == vExpectedOutputVol);
}

TEST_CASE("flip", "flip")
{
    MatrixXd InputVol1(4, 1);
    MatrixXd InputVol2(4, 1);
    MatrixXd InputVol3(4, 1);
    InputVol3 << 301, 302, 303, 304;
    InputVol1 << 101, 102, 103, 104;
    InputVol2 << 201, 202, 203, 204;
    MatrixXd InputVol(4, 3);
    InputVol << InputVol1, InputVol2, InputVol3;

    SECTION("simple Flip")
    {
        size_t vInputDepth = 2;
        MatrixXd vFlipped = dlfunctions::flip(InputVol, vInputDepth);
        MatrixXd ExpectedVol1(4, 1);
        MatrixXd ExpectedVol2(4, 1);
        MatrixXd ExpectedVol3(4, 1);
        ExpectedVol3 << 302, 301, 304, 303;
        ExpectedVol1 << 102, 101, 104, 103;
        ExpectedVol2 << 202, 201, 204, 203;
        MatrixXd ExpectedVol(4, 3);
        ExpectedVol << ExpectedVol1, ExpectedVol2, ExpectedVol3;

        REQUIRE(vFlipped == ExpectedVol);
    }
}

TEST_CASE("flatten and unflatten one after the other should return the input", "flattenLayer")
{
    // Input
    MatrixXd InputVol1(16, 1);
    MatrixXd InputVol2(16, 1);
    MatrixXd InputVol3(16, 1);
    InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 330, 311, 312, 313, 314, 315, 316;
    InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
    InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 211, 212, 213, 214, 215, 216;

    MatrixXd InputVol(16, 3);
    InputVol << InputVol1, InputVol2, InputVol3;

    MatrixXd flattened = dlfunctions::flatten(InputVol, 2);
    std::cout << "- flattened" << std::endl;
    std::cout << flattened.rows() << " " << flattened.cols() << std::endl;
    MatrixXd unflattened = dlfunctions::unflatten(InputVol, 3, 4, 4);
    std::cout << "- unflattened" << std::endl;
    std::cout << unflattened.rows() << " " << unflattened.cols() << std::endl;
    REQUIRE(unflattened == InputVol);
}

TEST_CASE("forward and backward maxpool elements", "maxpool")
{

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
    MatrixXd BackpropVol = MatrixXd::Zero(16, 3);
    size_t vBackFieldsVol = tFILTERSIZE * tFILTERSIZE * vNumChannelsVol;
    dlfunctions::colpool2im(tFILTERSIZE, BackpropOutputAll.data(), BackpropVol.data(), vOutHeightVol, vOutWidthVol, vBackFieldsVol, vImageHeightVol, vImageWidthVol, vStrideVol, vNumSamples);
    std::cout << "BackpropVol" << std::endl;
    std::cout << BackpropVol << std::endl;

    MatrixXd ShouldVol1(16, 1);
    MatrixXd ShouldVol2(16, 1);
    MatrixXd ShouldVol3(16, 1);
    ShouldVol3 << 0, 0, 0, 0, 0, 306, 0, 308, 0, 0, 0, 0, 0, 314, 0, 316;
    ShouldVol1 << 0, 0, 0, 0, 0, 106, 0, 108, 0, 0, 0, 0, 0, 114, 0, 116;
    ShouldVol2 << 0, 0, 0, 0, 0, 206, 0, 208, 0, 0, 0, 0, 0, 214, 0, 216;
    MatrixXd ShouldVol(16, 3);
    ShouldVol << ShouldVol1, ShouldVol2, ShouldVol3;

    REQUIRE(BackpropVol == ShouldVol);
}