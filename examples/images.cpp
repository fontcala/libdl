#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/ConvLayer.h>

int main()
{
  MatrixXd InputVol1(16, 1);
  MatrixXd InputVol2(16, 1);
  MatrixXd InputVol3(16, 1);
  InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 330, 311, 312, 313, 314, 315, 316;
  InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
  InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 211, 212, 213, 214, 215, 216;
  MatrixXd InputVol(16, 3);
  InputVol << InputVol1, InputVol2, InputVol3;


  std::cout << "Input" << std::endl;
  std::cout << InputVol << std::endl;
  
  size_t vInputDepth = 4;
  MatrixXd vFlipped = dlfunctions::flip(InputVol,vInputDepth);
  // size_t vInputDepth = 4;
  // size_t vOutputDepth = 3;
  // size_t vFilterSize = InputVol.rows();
  // size_t v2DFilterSize = vFilterSize / vInputDepth;

  // for(int i = 0; i < vInputDepth; ++i){
  //   InputVol.block(v2DFilterSize * i,0,v2DFilterSize,vOutputDepth).colwise().reverseInPlace();
  // }
  std::cout << "Input reverse" << std::endl;
  std::cout << vFlipped << std::endl;
}
