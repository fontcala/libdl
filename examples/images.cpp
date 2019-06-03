#include <iostream>
#include <libdl/dlfunctions.h>
//#include <libdl/ConvLayer.h>

// template <int size> // size is the non-type parameter
// class someClass
// {
// public:
//     MatrixXd * someM;
//     void someF();
// };
// template<int size>
// void someClass<size>::someF(){
//     auto aa = (*someM).block<size,size>(0, 0).reshape(1,size*size).transpose();
// }

template <int FilterSize>
void im2col(MatrixXd *aOutput, MatrixXd *aInput, size_t aStride, size_t aNumChannels)
{
    // TODO Check dims
    size_t limitRow = (*aInput).rows() - FilterSize + 1;
    size_t imWidth = (*aInput).cols() / aNumChannels;
    size_t limitCol = imWidth - FilterSize + 1;
    size_t rowIndex = 0;
    for (size_t row = 0; row < limitRow; row = row + aStride)
    {
        for (size_t col = 0; col < limitCol; col = col + aStride)
        {
            for (size_t chan = 0; chan < aNumChannels; chan++)
            {
                size_t coloffset = chan * imWidth;
                (*aOutput).block<1, FilterSize * FilterSize>(rowIndex, 0 + coloffset) = (*aInput).block<FilterSize, FilterSize>(row, col + coloffset).transpose().reshaped();
            }
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

    MatrixXd Input(4, 4);
    Input << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    std::cout << "Input" << std::endl;
    std::cout << Input << std::endl;
    // std::cout << "magic:" << std::endl;
    // std::cout << Input.block(0,0,2,2).reshaped<Eigen::RowMajor>().transpose() << std::endl;
    // Input.row(0) = Input.block(0,0,2,2).reshaped<Eigen::RowMajor>().transpose();
    // std::cout << Input << std::endl;
    size_t vFilterSize = 2;
    size_t vImageHeight = Input.rows();
    size_t vImageWidth = Input.cols();
    size_t vStride = 2;
    size_t vNumChannels = 1;
    size_t vOutHeight = (vImageHeight - vFilterSize) / vStride + 1;
    size_t vOutWidth = (vImageWidth - vFilterSize) / vStride + 1;
    MatrixXd Output(vOutHeight * vOutWidth, vFilterSize * vFilterSize);

    im2col<2>(&Output, &Input, vStride, vNumChannels);
    std::cout << "Output" << std::endl;
    std::cout << Output << std::endl;

    MatrixXd InputVol1(4, 4);
    MatrixXd InputVol2(4, 4);
    MatrixXd InputVol3(4, 4);
    MatrixXd InputVol(4, 4 * 3);
    InputVol3 << 301, 302, 303, 304, 305, 306, 307, 308, 309, 330, 311, 312, 313, 314, 315, 316;
    InputVol1 << 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116;
    InputVol2 << 201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 211, 212, 213, 214, 215, 216;
    InputVol << InputVol1, InputVol2, InputVol3;
    size_t vNumChannelsVol = 3;
    size_t vOutHeightVol = (InputVol.rows() - vFilterSize) / vStride + 1;
    size_t vOutWidthVol = (InputVol.cols() / vNumChannelsVol - vFilterSize) / vStride + 1;
    MatrixXd OutputVol(vOutHeightVol * vOutWidthVol, vFilterSize * vFilterSize * vNumChannelsVol);
    std::cout << "InputVol" << std::endl;
    std::cout << InputVol << std::endl;
    im2col<2>(&OutputVol, &InputVol, vStride, vNumChannelsVol);
    std::cout << "Output:" << std::endl;
    std::cout << OutputVol << std::endl;

    // someClass<3> sc;
    // sc.someM = &Input;
    // sc.someF();
}
