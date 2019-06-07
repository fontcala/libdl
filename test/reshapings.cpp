#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include "libdl/dlfunctions.h"

// TODO make scenarios for different stride, sizes etc...
TEST_CASE("2D Data", "2D Data")
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
    dlfunctions::im2col<2,2>(&Output, &Input, vStride, vNumChannels, vImageHeight, vImageWidth);

    MatrixXd vExpectedOutput(vOutHeight * vOutWidth, 2 * 2 * vNumChannels);
    vExpectedOutput << 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16;

    REQUIRE(Output == vExpectedOutput);
}

TEST_CASE("3D Data", "3D Data")
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
    dlfunctions::im2col<2,2>(&OutputVol, &InputVol, vStrideVol, vNumChannelsVol,vImageHeightVol,vImageWidthVol);


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

TEST_CASE("flatten and unflatten", "flattenLayer")
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


  MatrixXd flattened = dlfunctions::flatten(InputVol,2);
  MatrixXd unflattened = dlfunctions::unflatten(InputVol,3,4,4);
  REQUIRE(unflattened == InputVol);

}