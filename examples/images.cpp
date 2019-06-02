#include <iostream>
#include <libdl/dlfunctions.h>

void im2col(MatrixXd *aOutput, MatrixXd *aInput, size_t aFilterSize, size_t aStride)
{
    int limitRow = (*aInput).rows() - aFilterSize + 1;
    int limitCol = (*aInput).cols() - aFilterSize + 1;
    int rowIndex = 0;
    for (int row = 0; row < limitRow; row = row + aStride)
    {
        for (int col = 0; col < limitCol; col = col + aStride)
        {
            (*aOutput).row(rowIndex) = (*aInput).block(row, col, aFilterSize, aFilterSize).reshaped<Eigen::RowMajor>().transpose();
            rowIndex++;
        }
    }
}
int main()
{
    // MatrixXd imageData = MatrixXd::Random(21,7);
    // int vImageSizeX = 7;
    // int vImageSizeY = 7;
    // int vCurrentDim = 1;
    // int vKernelSizeX = 3;
    // int vKernelSizeY = 3;
    // int vConvIndx = 2;
    // int vConvIndy = 1;
    // //3*7*7
    // std::cout << imageData.block<7,7>(vCurrentDim*7,0).block<3,3>(vConvIndx,vConvIndy) << std::endl;
    // std::cout << imageData.block<3,3>(vCurrentDim * vImageSizeX + vConvIndx,vConvIndy) << std::endl;
    // std::cout << imageData.block(vCurrentDim * vImageSizeX + vConvIndx,vConvIndy,3,3) << std::endl;
    
    Eigen::MatrixXd Input(4, 4);
    Input << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    std::cout << "Input" << std::endl;
    std::cout << Input << std::endl;
    // std::cout << "magic:" << std::endl;
    // std::cout << Input.block(0,0,2,2).reshaped<Eigen::RowMajor>().transpose() << std::endl;
    // Input.row(0) = Input.block(0,0,2,2).reshaped<Eigen::RowMajor>().transpose();
    // std::cout << Input << std::endl;
    size_t vFilterSize = 3;
    size_t vImageHeight = Input.rows();
    size_t vImageWidth = Input.cols();
    size_t vStride = 1;
    size_t vOutHeight = (vImageHeight - vFilterSize) / vStride + 1;
    size_t vOutWidth = (vImageWidth - vFilterSize) / vStride + 1;
    Eigen::MatrixXd Output(vOutHeight * vOutWidth, vFilterSize * vFilterSize);

    im2col(&Output,&Input,vFilterSize,vStride);
    std::cout << "Output" << std::endl;
    std::cout << Output << std::endl;
}
