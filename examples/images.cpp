#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/ConvLayer.h>
#include <libdl/SigmoidActivationLayer.h>

int main()
{
  // // Input
  // MatrixXd InputVol1(16, 1);
  // MatrixXd InputVol2(16, 1);
  // MatrixXd InputVol3(16, 1);
  // InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 330, 311, 312, 313, 314, 315, 316;
  // InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
  // InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 211, 212, 213, 214, 215, 216;
  // MatrixXd InputVol(16, 3);
  // InputVol << InputVol1, InputVol2, InputVol3;

  // // Parameters
  // const size_t tFILTERSIZE = 3; // Both sides
  // size_t vNumChannelsVol = 3;
  // size_t vImageHeightVol = 4;
  // size_t vImageWidthVol = 4;
  // size_t vStrideVol = 1;
  // size_t vPaddingVol = 1;
  // size_t vOutHeightVol = (vImageHeightVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
  // size_t vOutWidthVol = (vImageWidthVol - tFILTERSIZE + 2 * vPaddingVol) / vStrideVol + 1;
  // size_t vOutFieldsVol = tFILTERSIZE * tFILTERSIZE * vNumChannelsVol;
  // size_t vNumSamples = 1;

  // // Filters (Identity)
  // // MatrixXd FilterVol1(vOutFieldsVol, 1);
  // // MatrixXd FilterVol2(vOutFieldsVol, 1);
  // // MatrixXd FilterVol3(vOutFieldsVol, 1);
  // // FilterVol3 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
  // // FilterVol1 << 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // // FilterVol2 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // // MatrixXd FilterVol(vOutFieldsVol, 3);
  // // FilterVol << FilterVol1, FilterVol2, FilterVol3;
  // size_t vFiltersNumber = 3;
  // MatrixXd FilterVol = MatrixXd::Random(vOutFieldsVol, vFiltersNumber);
  
  // // MatrixXd OutputVol(vOutHeightVol * vOutWidthVol, vOutFieldsVol);
  // // dlfunctions::im2col(tFILTERSIZE,tFILTERSIZE,InputVol.data(), OutputVol.data(), vOutHeightVol, vOutWidthVol, vOutFieldsVol, vImageHeightVol, vImageWidthVol, vNumChannelsVol, vPaddingVol, vPaddingVol, vStrideVol, vNumSamples);
  // // MatrixXd OutputConv = OutputVol * FilterVol;

  // // Now using the conv function
  // MatrixXd OutputVolCnv(vOutHeightVol * vOutWidthVol, vFiltersNumber);
  // dlfunctions::convolution(OutputVolCnv, vOutHeightVol, vOutWidthVol, FilterVol, tFILTERSIZE, tFILTERSIZE, InputVol, vImageHeightVol, vImageWidthVol, vNumChannelsVol, vPaddingVol, vPaddingVol, vStrideVol, vNumSamples);

  // std::cout << OutputVolCnv << std::endl;
  // //Input
  const size_t vInputSampleNumber = 1;

  const size_t vInputDepth1 = 3;
  const size_t vInputHeight1 = 7;
  const size_t vInputWidth1 = 5;
  MatrixXd Input = MatrixXd::Random(vInputHeight1 * vInputWidth1,vInputDepth1);
  std::cout << "Input" << std::endl;
  std::cout << Input << std::endl;

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
  dlfunctions::im2col(vFilterHeight1,vFilterWidth1, Input.data(), im2ColImage.data(),vOutputHeight1,vOutputWidth1, vOutFields,vInputHeight1,vInputWidth1,vInputDepth1,
            vPaddingHeight1,vPaddingWidth1,vStride1, 1);
  std::cout << im2ColImage.data()[107] << std::endl;
  std::cout << im2ColImage << std::endl;
  // ConvLayer firstConvLayer(vFilterHeight1,
  //                          vFilterWidth1,
  //                          vPaddingHeight1,
  //                          vPaddingWidth1,
  //                          vStride1,
  //                          vInputDepth1,
  //                          vInputHeight1,
  //                          vInputWidth1,
  //                          vOutputDepth1,
  //                          vOutputHeight1,
  //                          vOutputWidth1,
  //                          vInputSampleNumber);
  // firstConvLayer.SetInput(Input);

  // firstConvLayer.ForwardPass();
  // std::cout << *(firstConvLayer.GetOutput()) << std::endl;
}
