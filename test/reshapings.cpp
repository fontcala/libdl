#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include "libdl/dlfunctions.h"

// TODO make scenarios for different stride, sizes etc...
TEST_CASE("shapes", "3D horizontal stacked im2col")
{

    const size_t vInputSampleNumber = 1;
    const size_t vInputDepth1 = 3;
    const size_t vInputHeight1 = 7;
    const size_t vInputWidth1 = 5;
    MatrixXd Input(vInputHeight1 * vInputWidth1, vInputDepth1);
    Input << 0.680375, 0.0485744, 0.0632129, -0.211234, -0.012834, -0.921439, 0.566198, 0.94555, -0.124725, 0.59688, -0.414966, 0.86367, 0.823295, 0.542715, 0.86162, -0.604897, 0.05349, 0.441905, -0.329554, 0.539828, -0.431413, 0.536459, -0.199543, 0.477069, -0.444451, 0.783059, 0.279958, 0.10794, -0.433371, -0.291903 - 0.0452059, -0.295083, 0.375723, 0.257742, 0.615449, -0.668052, -0.270431, 0.838053, -0.119791, 0.0268018, -0.860489, 0.76015, 0.904459, 0.898654, 0.658402, 0.83239, 0.0519907, -0.339326, 0.271423, -0.827888, -0.542064, 0.434594, -0.615572, 0.786745, -0.716795, 0.326454, -0.29928, 0.213938, 0.780465, 0.37334, -0.967399, -0.302214, 0.912937, -0.514226, -0.871657, 0.17728, -0.725537, -0.959954, 0.314608, 0.608354, -0.0845965, 0.717353, -0.686642, -0.873808, -0.12088, -0.198111, -0.52344, 0.84794, -0.740419, 0.941268, -0.203127, -0.782382, 0.804416, 0.629534, 0.997849, 0.70184, 0.368437, -0.563486, -0.466669, 0.821944, 0.0258648, 0.0795207, -0.0350187, 0.678224, -0.249586, -0.56835, 0.22528, 0.520497, 0.900505, -0.407937, 0.0250707, 0.840257, 0.275105, 0.335448, -0.70468;
    //std::cout << "Input" << std::endl;
    //std::cout << Input << std::endl;

    //Params
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 2;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 2;

    const size_t vOutputDepth1 = 6;
    const size_t vOutputHeight1 = (vInputHeight1 - vFilterHeight1 + 2 * vPaddingHeight1) / vStride1 + 1;
    const size_t vOutputWidth1 = (vInputWidth1 - vFilterWidth1 + 2 * vPaddingWidth1) / vStride1 + 1;

    size_t vOutFields = vFilterHeight1 * vFilterWidth1 * vInputDepth1;
    MatrixXd im2ColImage(vOutputHeight1 * vOutputWidth1, vOutFields);
    dlfunctions::im2col(vFilterHeight1, vFilterWidth1, Input.data(), im2ColImage.data(), vOutputHeight1, vOutputWidth1, vOutFields, vInputHeight1, vInputWidth1, vInputDepth1,
                        vPaddingHeight1, vPaddingWidth1, vStride1, 1);
    
    MatrixXd im2ColVerifyRow(1, vOutFields);
    im2ColVerifyRow << 0,0,-0.211234,0.566198,-0.329554,0.536459,0.615449,0.838053,-0.827888,-0.615572,0,0,-0.012834,0.94555,0.539828,-0.199543,-0.668052,-0.119791,-0.542064,0.786745,0,0,-0.921439,-0.124725,-0.431413,0.477069,-0.270431,0.0268018,0.434594,-0.716795;

    REQUIRE(im2ColImage.row(1) == im2ColVerifyRow);

    // How to test the col2im part?
    // MatrixXd col2ImImage = MatrixXd::Zero(vInputHeight1 * vInputWidth1, vInputDepth1);
    // dlfunctions::col2im(vFilterHeight1, vFilterWidth1, im2ColImage.data(), col2ImImage.data(), vOutputHeight1, vOutputWidth1, vOutFields, vInputHeight1, vInputWidth1, vInputDepth1,
    //         vPaddingHeight1, vPaddingWidth1, vStride1, 1);
    // std::cout << "OutputVol 3" << std::endl;
    // std::cout << col2ImImage << std::endl;
}

TEST_CASE("2D Data", "image im2col")
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

TEST_CASE("3D Data", "image im2col")
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

    MatrixXd flattened = dlfunctions::flatten(InputVol, 2);
    MatrixXd unflattened = dlfunctions::unflatten(InputVol, 3, 4, 4);
    REQUIRE(unflattened == InputVol);
}