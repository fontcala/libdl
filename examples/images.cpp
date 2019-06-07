#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/ConvLayer.h>
#include <libdl/SigmoidActivationLayer.h>

MatrixXd flattenz(const MatrixXd &aInput, const size_t aNumberCuts)
{
    size_t vInputDim1 = aInput.rows();
    size_t vInputDim2 = aInput.cols();
    size_t vBlockSize = vInputDim1 / aNumberCuts;
    size_t vOutputCols = vInputDim2 * vBlockSize;
    MatrixXd vToBeReturned(aNumberCuts,vOutputCols);

    for (int i = 0; i < aNumberCuts; ++i)
    { 
        MatrixXd vSampleBlock = aInput.block(i*vBlockSize,0,vBlockSize,vInputDim2);
        vSampleBlock.resize(1,vOutputCols);
        vToBeReturned.row(i) = vSampleBlock;
    }
    return vToBeReturned;
}
MatrixXd unflatten(const MatrixXd &aInput,const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
{
    size_t vNumberSamples = aInput.rows();
    size_t vUnflatSampleSize = aInputHeight * aInputWidth / vNumberSamples;
    MatrixXd vToBeReturned(vNumberSamples * vUnflatSampleSize,aInputDepth);
    for (int i = 0; i < vNumberSamples; ++i)
    {
        MatrixXd vSampleBlock = aInput.row(i);
        vSampleBlock.resize(vUnflatSampleSize,aInputDepth);
        vToBeReturned.block(i*vUnflatSampleSize,0,vUnflatSampleSize,aInputDepth) = vSampleBlock;
    }
    return vToBeReturned;
}

int main()
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


  MatrixXd flattened = flattenz(InputVol,2);
  std::cout << flattened << std::endl;
  // MatrixXd aa = flattened.row(0);
  // aa.resize(8,3);
  // std::cout << aa << std::endl;
  MatrixXd unflattened = unflatten(InputVol,3,4,4);
  std::cout << unflattened << std::endl;

  // Input
  // const size_t vInputSampleNumber = 1;

  // const size_t vInputDepth1 = 3;
  // const size_t vInputHeight1 = 7;
  // const size_t vInputWidth1 = 5;
  // MatrixXd Input = MatrixXd::Random(7 * 5, 3);

  // //Params
  // const size_t vFilterHeight1 = 5;
  // const size_t vFilterWidth1 = 2;
  // const size_t vPaddingHeight1 = 1;
  // const size_t vPaddingWidth1 = 1;
  // const size_t vStride1 = 2;

  // const size_t vOutputDepth1 = 6;
  // const size_t vOutputHeight1 = (vInputHeight1 - vFilterHeight1 + 2 * vPaddingHeight1) / vStride1 + 1;
  // const size_t vOutputWidth1 = (vInputWidth1 - vFilterWidth1 + 2 * vPaddingWidth1) / vStride1 + 1;

  // ConvLayer firstLayer(vFilterHeight1,
  //                      vFilterWidth1,
  //                      vPaddingHeight1,
  //                      vPaddingWidth1,
  //                      vStride1,
  //                      vInputDepth1,
  //                      vInputHeight1,
  //                      vInputWidth1,
  //                      vOutputDepth1,
  //                      vOutputHeight1,
  //                      vOutputWidth1,
  //                      vInputSampleNumber);
}
