#include <iostream>
#include <type_traits>
#include <libdl/dlfunctions.h>
#include <libdl/MaxPoolLayer.h>
#include <math.h>

using Eigen::MatrixXd;
int main()
{
  MatrixXd InputVol1(16, 1);
  MatrixXd InputVol2(16, 1);
  MatrixXd InputVol3(16, 1);
  InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316;
  InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
  InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216;
  MatrixXd InputVol(16, 3);
  InputVol << -1 * InputVol1, InputVol2, InputVol3;
//   MatrixXd InputVol = MatrixXd::Random(16,3);
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

  MaxPoolLayer maxP(vNumChannelsVol, vImageHeightVol, vImageWidthVol, tFILTERSIZE, vStrideVol, vNumSamples);
  maxP.SetInput(InputVol);
  maxP.ForwardPass();
  std::cout << "*(maxP.GetOutput())" << std::endl;
  std::cout << *(maxP.GetOutput()) << std::endl;
  maxP.SetBackpropInput(maxP.GetOutput());
  maxP.BackwardPass();
  std::cout << "*(maxP.GetBackpropOutput())" << std::endl;
  std::cout << *(maxP.GetBackpropOutput()) << std::endl;

  
  
  MatrixXd ShouldVol1(16, 1);
  MatrixXd ShouldVol2(16, 1);
  MatrixXd ShouldVol3(16, 1);
  ShouldVol3 << 0, 0, 0, 0, 0, 306, 0, 308, 0, 0, 0, 0, 0, 314, 0, 316;
  ShouldVol1 << -101, 0, 0, 0, 0, 106, 0, 108, 0, 0, 0, 0, 0, 114, 0, 116;
  ShouldVol2 << 0, 0, 0, 0, 0, 206, 0, 208, 0, 0, 0, 0, 0, 214, 0, 216;
  MatrixXd ShouldVol(16, 3);
  ShouldVol << ShouldVol1, ShouldVol2, ShouldVol3;

}
