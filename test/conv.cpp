#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include "libdl/dlfunctions.h"

using Eigen::MatrixXd;
TEST_CASE("convolution with stacked Identity image kernels should be the same as the input", "convolution")
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

    // Parameters
    const size_t tFILTERSIZE = 3; // Both sides
    size_t vNumChannelsVol = 3;
    size_t vImageHeightVol = 4;
    size_t vImageWidthVol = 4;
    size_t vStrideVol = 1;
    size_t vPaddingVol = 1;
    size_t vOutHeightVol = (vImageHeightVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
    size_t vOutWidthVol = (vImageWidthVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
    size_t vOutFieldsVol = tFILTERSIZE * tFILTERSIZE * vNumChannelsVol;
    size_t vNumSamples = 1;

    // Filters (Identity)
    MatrixXd FilterVol1(vOutFieldsVol, 1);
    MatrixXd FilterVol2(vOutFieldsVol, 1);
    MatrixXd FilterVol3(vOutFieldsVol, 1);
    FilterVol3 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
    FilterVol1 << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    FilterVol2 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    MatrixXd FilterVol(vOutFieldsVol, 3);
    FilterVol << FilterVol1, FilterVol2, FilterVol3;

    SECTION("applying im2col manually")
    {
        MatrixXd OutputVol(vOutHeightVol * vOutWidthVol, vOutFieldsVol);
        dlfunctions::im2col(tFILTERSIZE,tFILTERSIZE,InputVol.data(), OutputVol.data(), vOutHeightVol, vOutWidthVol, vOutFieldsVol, vImageHeightVol, vImageWidthVol, vPaddingVol, vPaddingVol, vStrideVol);
        MatrixXd OutputConv = OutputVol * FilterVol;

        REQUIRE(OutputConv == InputVol);
    }
    SECTION("using the convolution function")
    {
        // Now using the conv function
        size_t vFiltersNumber = 3;
        MatrixXd OutputVolCnv(vOutHeightVol * vOutWidthVol, vFiltersNumber);
        dlfunctions::convolution(OutputVolCnv,vOutHeightVol, vOutWidthVol,FilterVol,tFILTERSIZE,tFILTERSIZE, InputVol, vImageHeightVol, vImageWidthVol, vNumChannelsVol, vPaddingVol, vPaddingVol, vStrideVol, vNumSamples);

        REQUIRE(OutputVolCnv == InputVol);
    }
}