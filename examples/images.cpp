#include <iostream>
#include <libdl/dlfunctions.h>
int main()
{
    MatrixXd imageData = MatrixXd::Random(21,7);
    int vImageSizeX = 7;
    int vImageSizeY = 7;
    int vCurrentDim = 1;
    int vKernelSizeX = 3;
    int vKernelSizeY = 3;
    int vConvIndx = 2;
    int vConvIndy = 1;
    //3*7*7
    std::cout << imageData.block<7,7>(vCurrentDim*7,0).block<3,3>(vConvIndx,vConvIndy) << std::endl;
    std::cout << imageData.block<3,3>(vCurrentDim * vImageSizeX + vConvIndx,vConvIndy) << std::endl;
    std::cout << imageData.block(vCurrentDim * vImageSizeX + vConvIndx,vConvIndy,3,3) << std::endl;
}
